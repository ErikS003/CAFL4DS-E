"""The P1.0.2 hand-off corpus — turn logged health arms into a wide, metadata-tagged time series.

Where :mod:`cafl4ds.deploy` assembles the run-time deploy *report* (the trust-annotated channels + the
Tier-A verdict), this module turns that report into the **corpus a partner actually mines**: a flat,
wide table with one row per ``(seed, arm, checkpoint)``, every health signal a column, and each row
tagged with the **event-relative coordinates** that make a stateful trajectory comparable across seeds
— the leg (``era_index`` / ``era_name``) it was read in, whether it sits on a regime **transition**,
and how many steps into the leg it is. Alongside the table it writes the **leg map** (``segments.json``
— the ordered regime blocks, derived from the same era sequence), the **trust map**, and a provenance
**manifest**.

The transforms are pure (no torch, no harness) and fully testable; the writer emits ``health.csv`` via
the stdlib and ``health.parquet`` via a **lazy** ``pyarrow`` import, so the wheel never hard-requires
it. The corpus is the deliverable; the health series it reads come from :mod:`cafl4ds.harness` arms,
projected through :func:`cafl4ds.deploy.build_deploy_report`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from cafl4ds.health_trust import Backbone

# The corpus schema version (independent of the deploy-report schema).
CORPUS_SCHEMA_VERSION = 1

# Health-record bookkeeping fields that are not instrument signals (mirrors cafl4ds.deploy).
_BOOKKEEPING = frozenset({"step", "era", "loss", "run", "series"})

# The fixed leading columns of the corpus table; instrument signals follow, sorted.
_CONTEXT_COLUMNS = [
    "backbone",
    "seed",
    "arm",
    "checkpoint_index",
    "step",
    "era_index",
    "era_name",
    "is_transition",
    "steps_since_transition",
    "leg_ordinal",
    "loss",
]


def event_coordinates(health: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive per-checkpoint event-relative coordinates from the era sequence.

    A stateful health trajectory is only comparable across seeds when each reading is located
    *relative to the regime transitions* rather than by absolute step. This computes, per checkpoint:
    the leg it belongs to (``era_index``), whether it opens a new leg (``is_transition``), how many
    stream steps into the leg it sits (``steps_since_transition`` — the axis to event-lock on), and its
    ordinal within the leg (``leg_ordinal``).

    Args:
        health: The arm's health records, in checkpoint order (each carries ``era`` and ``step``).

    Returns:
        One coordinate dict per record, aligned to ``health``.
    """
    coords: list[dict[str, Any]] = []
    prev_era: int | None = None
    leg_start_step = 0
    leg_ordinal = 0
    for i, record in enumerate(health):
        era = int(record["era"])
        step = int(record["step"])
        transition = prev_era is None or era != prev_era
        if transition:
            leg_start_step = step
            leg_ordinal = 0
        coords.append(
            {
                "checkpoint_index": i,
                "era_index": era,
                "is_transition": transition,
                "steps_since_transition": step - leg_start_step,
                "leg_ordinal": leg_ordinal,
            }
        )
        prev_era = era
        leg_ordinal += 1
    return coords


def health_rows(
    health: list[dict[str, Any]],
    *,
    family: Backbone,
    seed: int,
    arm: str,
    era_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten one arm's health series into wide corpus rows (context columns + every signal).

    Args:
        health: The arm's health records, in checkpoint order.
        family: The backbone family (stamped on every row).
        seed: The run seed (the ensemble axis — one trajectory per seed).
        arm: The arm role (``live`` / ``pc`` / ``b5``).
        era_names: Optional map from era (walk position) to a human name; falls back to ``era{idx}``.

    Returns:
        One wide row per checkpoint.
    """
    names = era_names or {}
    rows: list[dict[str, Any]] = []
    for record, coord in zip(health, event_coordinates(health), strict=True):
        era = coord["era_index"]
        row: dict[str, Any] = {
            "backbone": family.value,
            "seed": seed,
            "arm": arm,
            "checkpoint_index": coord["checkpoint_index"],
            "step": int(record["step"]),
            "era_index": era,
            "era_name": names.get(era, f"era{era}"),
            "is_transition": coord["is_transition"],
            "steps_since_transition": coord["steps_since_transition"],
            "leg_ordinal": coord["leg_ordinal"],
            "loss": record.get("loss"),
        }
        row.update({k: v for k, v in record.items() if k not in _BOOKKEEPING})
        rows.append(row)
    return rows


def rows_from_report(
    report: dict[str, Any], *, seed: int, era_names: dict[int, str] | None = None
) -> list[dict[str, Any]]:
    """Build the corpus rows for every arm in a deploy report.

    Args:
        report: A :func:`cafl4ds.deploy.build_deploy_report` result.
        seed: The seed that produced this report.
        era_names: Optional era (walk position) to human-name map.

    Returns:
        The concatenated wide rows across the report's arms.
    """
    family = Backbone(report["backbone_family"])
    rows: list[dict[str, Any]] = []
    for arm, payload in report["arms"].items():
        rows.extend(health_rows(payload["health"], family=family, seed=seed, arm=arm, era_names=era_names))
    return rows


def segment_map(
    health: list[dict[str, Any]],
    *,
    era_names: dict[int, str] | None = None,
    composition: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Derive the leg map — the ordered contiguous regime blocks — from a health series.

    Each segment is one leg of the drive (a maximal run of a single era), with its step and checkpoint
    span. This is the ``sunny → rain → …`` marker set a consumer aligns dynamics against. When a
    per-era ``composition`` is supplied (:meth:`cafl4ds.data.regime.RegimeStream.era_composition`),
    each leg is also tagged with the label aggregates of the images it streamed.

    Args:
        health: The (live) arm's health records, in checkpoint order.
        era_names: Optional era (walk position) to human-name map.
        composition: Optional per-era label composition (image count + scene / category histograms).

    Returns:
        The ordered segments.
    """
    names = era_names or {}
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for record, coord in zip(health, event_coordinates(health), strict=True):
        era = coord["era_index"]
        step = int(record["step"])
        checkpoint = coord["checkpoint_index"]
        if coord["is_transition"]:
            if current is not None:
                segments.append(current)
            current = {
                "era_index": era,
                "era_name": names.get(era, f"era{era}"),
                "start_step": step,
                "end_step": step,
                "start_checkpoint": checkpoint,
                "end_checkpoint": checkpoint,
                "n_checkpoints": 1,
            }
        elif current is not None:
            current["end_step"] = step
            current["end_checkpoint"] = checkpoint
            current["n_checkpoints"] += 1
    if current is not None:
        segments.append(current)
    if composition:
        for segment in segments:
            leg = composition.get(segment["era_index"])
            if leg is not None:
                segment["n_images"] = leg["n_images"]
                segment["scene_hist"] = leg["scene_hist"]
                if leg["category_hist"]:
                    segment["category_hist"] = leg["category_hist"]
    return segments


def normalize_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Compute the unified column set (context-first, signals sorted) and fill gaps with ``None``.

    Rows from different backbones carry different signal columns; a single flat table needs one schema,
    so every row is widened to the column union (missing cells become ``None``).

    Args:
        rows: The wide rows (possibly with heterogeneous signal columns).

    Returns:
        ``(columns, normalized_rows)`` — the ordered columns and the gap-filled rows.
    """
    signal_columns: set[str] = set()
    for row in rows:
        signal_columns |= {k for k in row if k not in _CONTEXT_COLUMNS}
    columns = [*_CONTEXT_COLUMNS, *sorted(signal_columns)]
    normalized = [{c: row.get(c) for c in columns} for row in rows]
    return columns, normalized


def _write_json(path: Path, payload: object) -> Path:
    """Write ``payload`` as indented JSON."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    """Write the wide table as CSV (stdlib — always available)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _maybe_write_parquet(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path | None:
    """Write the wide table as parquet if ``pyarrow`` is importable, else skip it.

    The parquet twin is a convenience binary; ``health.csv`` is the portable primary. The wheel does
    not hard-require ``pyarrow`` (it is a dev/analysis extra, absent e.g. from the Gaudi runtime image),
    so a missing dependency is a graceful skip — not a failure that would discard an expensive drive's
    corpus at the final write. The parquet can always be regenerated from the CSV where pyarrow exists.

    Returns:
        The written path, or ``None`` when ``pyarrow`` is unavailable.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415 - lazy: the wheel does not hard-require parquet
        import pyarrow.parquet as pq  # noqa: PLC0415 - lazy: the wheel does not hard-require parquet
    except ImportError:  # pragma: no cover - exercised only without the optional dep
        logger.warning("pyarrow not installed — skipping the health.parquet twin (health.csv is the portable primary).")
        return None
    table = pa.Table.from_pylist(rows, schema=pa.schema([(c, pa.infer_type([r[c] for r in rows])) for c in columns]))
    pq.write_table(table, path)
    return path


def write_corpus(
    out_dir: str | Path,
    *,
    reports: list[tuple[int, dict[str, Any]]],
    era_names: dict[int, str] | None = None,
    composition: dict[int, dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the full hand-off corpus for an ensemble of seed reports into ``out_dir``.

    Emits ``health.csv`` + ``health.parquet`` (the wide time series across all seeds and arms),
    ``segments.json`` (the leg map), ``trust.json`` (the per-signal calibrated status), and
    ``manifest.json`` (provenance + the per-seed Tier-A verdicts, and the Tier-B sanity verdicts when
    they were computed). The segment map and trust block are shared across seeds (same config), so they
    are taken from the first report.

    Args:
        out_dir: The directory to write the corpus into (created if absent).
        reports: The ``(seed, report)`` pairs — one deploy report per seed.
        era_names: Optional era (walk position) to human-name map.
        composition: Optional per-era label composition to tag each leg in ``segments.json``.
        manifest: Optional extra provenance to merge into ``manifest.json`` (git sha, image count, …).

    Returns:
        A map of artifact name to written path.

    Raises:
        ValueError: If ``reports`` is empty.
    """
    if not reports:
        raise ValueError("write_corpus needs at least one (seed, report) pair.")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for seed, report in reports:
        all_rows.extend(rows_from_report(report, seed=seed, era_names=era_names))
    columns, rows = normalize_rows(all_rows)

    first = reports[0][1]
    live_health = (
        first["arms"]["live"]["health"] if "live" in first["arms"] else next(iter(first["arms"].values()))["health"]
    )
    study = str((manifest or {}).get("study", "P1.0.2"))
    provenance: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "study": study,
        "backbone_family": first["backbone_family"],
        "config": first["config"],
        "seeds": [seed for seed, _ in reports],
        "n_rows": len(rows),
        "channels": first["channels"],
        "tier_a_passed": {str(seed): report["validation"]["passed"] for seed, report in reports},
    }
    if any("tier_b" in report for _, report in reports):
        provenance["tier_b_passed"] = {
            str(seed): report["tier_b"]["passed"] for seed, report in reports if "tier_b" in report
        }
    if manifest:
        provenance.update(manifest)

    paths = {
        "health_csv": _write_csv(directory / "health.csv", columns, rows),
        "segments": _write_json(
            directory / "segments.json",
            {"study": study, "segments": segment_map(live_health, era_names=era_names, composition=composition)},
        ),
        "trust": _write_json(directory / "trust.json", first["trust"]),
        "manifest": _write_json(directory / "manifest.json", provenance),
    }
    parquet = _maybe_write_parquet(directory / "health.parquet", columns, rows)
    if parquet is not None:
        paths["health_parquet"] = parquet
    return paths
