"""Per-metric fire/quiet analysis for the collapse-instrument calibration (P0.2.2).

P0.2.1 calibrated a single instrument (RankMe) with a bespoke gate. P0.2.2 maps the *whole*
collapse suite — RankMe, VICReg variance/covariance, Wang & Isola uniformity/alignment — read
at every embedding surface the method exposes (``backbone`` and ``proj``). This module turns
the two arms' health series (``PC`` = forced collapse, ``healthy`` = intact SimSiam) into a
per-``(instrument × surface)`` **separation** and a two-sided **fires / quiet** reading, so we
can see *where each instrument responds and where it does not* across regimes.

It is a **reporting layer**, not a hard gate: the certified RankMe gate stays in
``scripts/positive_control.py``. Thresholds here are provisional mapping aids (the point of
P0.2.2 is to *find* each instrument's envelope), so the raw finals are always reported
alongside the booleans.

Each instrument carries its collapse **direction** and **scale**:

* ``rankme`` — ``down`` (collapse lowers effective rank), ratio ``healthy/PC``. The reference.
* ``mean_feature_var`` — ``down`` (dimensions → constant), ratio ``healthy/PC``.
* ``offdiag_cov`` — ``up`` (redundancy raises off-diagonal mass), ratio ``PC/healthy``. NB:
  point-collapse (SimSiam's mode) drives *all* variance → 0, so covariance → 0 too — this
  instrument targets a *different* (redundancy) sub-mode and is expected to stay quiet here.
* ``uniformity`` — ``up`` (embeddings clump, the potential climbs toward 0), gap ``PC−healthy``
  (values are negative, so a ratio is ill-defined).
* ``alignment`` — **not standalone**: under point-collapse positives map to the same constant,
  so alignment → 0 ("perfectly aligned"), which looks *healthy* in isolation. Reported for
  interpretation, read jointly with uniformity, never gated alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    """Collapse semantics of one instrument."""

    base: str
    direction: str  # "down" | "up": which way the value moves under collapse
    scale: str  # "ratio" | "gap": how separation between arms is summarized
    standalone: bool  # whether it is a standalone collapse detector (alignment is not)
    note: str = ""


# Ordered so the rendered table reads reference-first.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("rankme", "down", "ratio", True, "reference (calibrated in P0.2.1)"),
    MetricSpec("mean_feature_var", "down", "ratio", True, "VICReg variance; L2-normalized"),
    MetricSpec("offdiag_cov", "up", "ratio", True, "VICReg redundancy; expected null under point-collapse"),
    MetricSpec("uniformity", "up", "gap", True, "W&I uniformity (negative-valued)"),
    MetricSpec("alignment", "down", "gap", False, "read jointly with uniformity; not a standalone detector"),
)


def _series(records: list[dict[str, Any]], key: str) -> list[float]:
    """Values of ``key`` across health records that carry it, in order."""
    return [r[key] for r in records if isinstance(r.get(key), int | float)]


def _surfaces_for(base: str, records: list[dict[str, Any]]) -> list[str]:
    """Surface suffixes present for ``base`` in the records (``""`` = backbone, ``"_proj"`` …)."""
    suffixes: list[str] = []
    for suffix in ("", "_proj"):
        if any(f"{base}{suffix}" in r for r in records):
            suffixes.append(suffix)
    return suffixes


def _separation(spec: MetricSpec, healthy_final: float, pc_final: float) -> float:
    """Signed separation oriented so a larger value = more separated in the collapse direction.

    For ``ratio`` metrics this is the collapse-direction ratio (``healthy/PC`` for ``down``,
    ``PC/healthy`` for ``up``); for ``gap`` metrics it is the collapse-direction difference.
    """
    if spec.scale == "ratio":
        hi, lo = (healthy_final, pc_final) if spec.direction == "down" else (pc_final, healthy_final)
        return hi / lo if lo else float("inf")
    # gap: positive when PC is further in the collapse direction than healthy.
    return pc_final - healthy_final if spec.direction == "up" else healthy_final - pc_final


def metric_envelope(
    pc: list[dict[str, Any]],
    healthy: list[dict[str, Any]],
    *,
    min_ratio: float = 2.0,
    min_gap: float = 0.5,
) -> list[dict[str, Any]]:
    """Build the per-``(instrument × surface)`` separation + fires/quiet rows.

    Args:
        pc: The PC (forced-collapse) arm's health records.
        healthy: The intact arm's health records.
        min_ratio: Separation-ratio bar for ``ratio``-scale instruments (provisional).
        min_gap: Separation-gap bar for ``gap``-scale instruments (provisional).

    Returns:
        One row dict per present ``(base, surface)``, with the arms' finals, the oriented
        separation, and — for standalone instruments — a ``fires`` boolean.
    """
    rows: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        for suffix in _surfaces_for(spec.base, healthy):
            key = f"{spec.base}{suffix}"
            hc_series, pc_series = _series(healthy, key), _series(pc, key)
            if not hc_series or not pc_series:
                continue
            hc_final, pc_final = hc_series[-1], pc_series[-1]
            sep = _separation(spec, hc_final, pc_final)
            bar = min_ratio if spec.scale == "ratio" else min_gap
            row: dict[str, Any] = {
                "metric": spec.base,
                "surface": "backbone" if suffix == "" else suffix.lstrip("_"),
                "direction": spec.direction,
                "scale": spec.scale,
                "healthy_final": hc_final,
                "pc_final": pc_final,
                "separation": sep,
                "threshold": bar,
                "standalone": spec.standalone,
                "note": spec.note,
            }
            # `fires` only for standalone detectors, and only when separated in the RIGHT
            # direction (sep > bar, not merely |sep| large the wrong way).
            row["fires"] = bool(spec.standalone and sep >= bar)
            rows.append(row)
    return rows


def quiet_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """False-positive (P0.2.4) verdict: the quiet side holds iff **no** instrument fires.

    P0.2.4 stresses the *quiet* side — it feeds an **atypical-healthy** arm into the ``pc`` slot of
    :func:`metric_envelope` (paired against a canonical-healthy reference) and asks the *inverse* of
    the collapse gate: a healthy-but-atypical representation must **not** trip any detector. Each
    standalone instrument that separates past its bar is a **false fire** — a documented
    false-positive boundary of that (instrument × surface), not a failure to engineer away.

    Args:
        rows: The envelope rows from :func:`metric_envelope` (``atypical`` in the ``pc`` slot).

    Returns:
        A dict with the fired rows (metric/surface/separation/threshold) and ``passed`` — ``True``
        iff the quiet side held (nothing fired).
    """
    fired = [
        {
            "metric": r["metric"],
            "surface": r["surface"],
            "separation": r["separation"],
            "threshold": r["threshold"],
        }
        for r in rows
        if r["fires"]
    ]
    return {"mode": "quiet", "num_fired": len(fired), "fired": fired, "passed": not fired}


def render_envelope_table(rows: list[dict[str, Any]]) -> str:
    """Render the envelope rows as a fixed-width table (one line per instrument × surface)."""
    cols = ("metric", "surface", "direction", "healthy_final", "pc_final", "separation", "fires")
    widths = {"metric": 16, "surface": 8, "direction": 9, "fires": 7}
    header = "  ".join(f"{c:>{widths.get(c, 14)}}" for c in cols)
    lines = [header, "  ".join("-" * widths.get(c, 14) for c in cols)]
    for r in rows:
        cells = []
        for c in cols:
            w = widths.get(c, 14)
            v = r[c]
            if isinstance(v, bool):
                cells.append(f"{('yes' if v else 'no') if r['standalone'] else '(paired)':>{w}}")
            elif isinstance(v, float):
                cells.append(f"{v:>{w}.4f}")
            else:
                cells.append(f"{v:>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
