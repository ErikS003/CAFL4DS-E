r"""Phase-0 partial-collapse dose-response (P0.2.5) — does the collapse suite read the MIDDLE?

Every P0.2.x calibration drives SimSiam to a mathematically-forced trivial optimum (the "boiling
water" end). None checks whether the projector detectors read a graded scale *in between* — the
incipient / gradual degradation Phase-1 early-warning actually needs. This harness builds a
**partial-collapse continuum** at the certified P0.2.1 corner (IID x 40ep) and reads it three ways,
answering three audit gaps off ONE experiment:

* **C1 (monotonicity)** — as ``severity`` climbs from healthy to collapsed, does each projector
  instrument move *monotonically* in the collapse direction? Scored by the Spearman rank
  correlation of ``severity`` vs. the reading along the ladder.
* **C2 (surface coupling)** — does projector geometry *predict* the backbone's downstream utility?
  Scored by the Spearman correlation of RankMe@proj vs. the backbone probe accuracy across rungs.
* **C4 (independent anchor)** — the probe-accuracy contrast (healthy vs. collapsed) grounds
  "healthy" in real downstream usefulness, not just in the geometry instruments vouching for
  themselves.

TWO knobs reach collapse by mechanistically distinct paths, so agreement between them is a genuine
robustness test (pre-registered: both-monotone => C1 passes robustly; both-a-step => the
instruments cannot read the incipient range at toy scale; disagree => monotonicity is knob-
dependent). ``severity`` ``s`` maps: ``blend`` -> ``collapse_alpha = s`` (interpolate the whole
objective to the exact PC); ``softsg`` -> ``stopgrad_beta = 1 - s`` (weaken only the essential
mechanism, predictor kept on). Both share the healthy pole; their collapsed poles differ by design.

One session runs the whole severity ladder from an *identical* init (the seed is reset per rung), so
the ladder is a clean within-seed trajectory; distinct seeds give the min/mean-across-seed
robustness the rest of P0.2 uses. This is a **falsification / go-no-go probe**, not a pass/fail
gate: the output is the monotonicity verdict, read in the doc across knobs and seeds.

Examples:
    Default (STL-10, CPU), loss-blend knob::

        uv run python scripts/partial_collapse.py sweep.knob=blend seed=0

    Fast network-free smoke::

        uv run python scripts/partial_collapse.py data=synthetic img_size=16 epochs=6 \
            stream.support_per_class=8 stream.query_per_class=8 stream.era_eval_per_class=5

    On the Gaudi HPU (inside the container; see docs/developing.md)::

        ./scripts/run_gaudi_dev.sh -m /mnt/stl10 gaudi-env-cafl4ds:latest 0 \
            python scripts/partial_collapse.py sweep.knob=softsg seed=0 device=hpu
"""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.loop import StreamingLoop
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")

# Instruments scored for monotonicity, with the sign of a *collapse-consistent* trend as severity
# rises (the reading moves toward the collapse pole). Projector surface first (the calibrated core),
# then the backbone geometry, then the backbone probe accuracy (the C4 downstream anchor).
_PROJ_INSTRUMENTS = ("rankme_proj", "mean_feature_var_proj", "uniformity_proj", "offdiag_cov_proj")
_BACKBONE_INSTRUMENTS = ("rankme", "mean_feature_var", "uniformity")
_PROBES = ("knn_acc", "linear_acc")
# Expected collapse-direction sign of Spearman(severity, reading): -1 = the reading should FALL with
# severity (rank/variance/probe-accuracy collapse), +1 = it should RISE (uniformity -> 0, redundancy up).
_COLLAPSE_SIGN = {
    "rankme_proj": -1,
    "mean_feature_var_proj": -1,
    "uniformity_proj": +1,
    "offdiag_cov_proj": +1,
    "rankme": -1,
    "mean_feature_var": -1,
    "uniformity": +1,
    "knn_acc": -1,
    "linear_acc": -1,
}


def _knob_kwargs(knob: str, severity: float) -> dict[str, float]:
    """Map a ``severity`` in ``[0, 1]`` onto the selected knob's SimSiam constructor kwargs.

    ``blend`` interpolates the whole objective (``collapse_alpha = severity``); ``softsg`` weakens
    only the stop-gradient (``stopgrad_beta = 1 - severity``). Both are healthy at ``severity=0``.

    Args:
        knob: ``"blend"`` or ``"softsg"``.
        severity: The ladder position, ``0`` (healthy) .. ``1`` (collapsed).

    Returns:
        Keyword arguments for :func:`~cafl4ds.ssl.factory.build_simsiam`.

    Raises:
        ValueError: If ``knob`` is unknown.
    """
    if knob == "blend":
        return {"collapse_alpha": float(severity), "stopgrad_beta": 1.0}
    if knob == "softsg":
        return {"collapse_alpha": 0.0, "stopgrad_beta": 1.0 - float(severity)}
    raise ValueError(f"unknown sweep.knob {knob!r}; expected 'blend' or 'softsg'.")


def _run_rung(config: DictConfig, severity: float, *, run_name: str, out_dir: Path) -> dict[str, Any]:
    """Train one rung of the ladder at the given ``severity``; return its final readings.

    Mirrors ``scripts/positive_control.py``'s ``_run_arm`` but with the anti-collapse mechanism
    *softened* (not toggled off) by the selected knob. The global seed is reset so every rung in a
    seed starts from a bit-identical init and augmentation RNG — the knob value is the only variable
    across the ladder.

    Args:
        config: The composed ``partial_collapse`` config.
        severity: The ladder position (mapped to a knob value via :func:`_knob_kwargs`).
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        The rung record: ``severity``, the applied knob kwargs, the FINAL health record (all
        instruments at both surfaces + probe accuracy), and the min-over-steps SSL loss floor.
    """
    knob_kwargs = _knob_kwargs(config.sweep.knob, severity)
    torch.manual_seed(config.seed)  # identical init + augmentation RNG across rungs of a seed
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder, anti_collapse=True, **knob_kwargs)
    apply_encoder_init(method.encoder, "from_scratch")

    stream = instantiate(config.stream)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"rung '{run_name}': severity={severity} knob={config.sweep.knob} {knob_kwargs}")

    loop = StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=instantiate(config.filter),
        monitor=monitor,
        run_logger=run_logger,
        eval_every=eval_every,
        epochs=config.epochs,
        scheduler=scheduler,
        device=config.device,
    )
    loop.run()

    records = read_run(run_log_path)
    health = [r for r in records if r.get("series") == "health"]
    loss_floor = min(r["loss"] for r in records if r.get("series") == "loss")
    return {"severity": float(severity), "knob_kwargs": knob_kwargs, "final": health[-1], "loss_floor": loss_floor}


def _rank(values: list[float]) -> list[float]:
    """Return 1-based ranks of ``values``, averaging ties (for a Spearman correlation)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float | None], ys: list[float | None]) -> float:
    """Spearman rank correlation of paired series, dropping any pair with a missing value.

    Pure-Python (no numpy/scipy in this env): Pearson correlation of the average ranks.

    Args:
        xs: First series (``None`` entries drop the pair).
        ys: Second series (``None`` entries drop the pair).

    Returns:
        The Spearman correlation, or ``nan`` if fewer than two valid pairs / zero variance.
    """
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if isinstance(x, int | float) and isinstance(y, int | float)]
    if len(pairs) < 2:
        return float("nan")
    rx, ry = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def _series(ladder: list[dict[str, Any]], key: str) -> list[float | None]:
    """Extract the per-rung final reading of ``key`` along the ladder (``None`` where absent)."""
    out: list[float | None] = []
    for rung in ladder:
        value = rung["final"].get(key)
        out.append(float(value) if isinstance(value, int | float) else None)
    return out


def _monotonicity(ladder: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the three P0.2.5 reads off a finished ladder.

    Args:
        ladder: The per-rung records (must be ordered by ascending severity).

    Returns:
        ``{"per_instrument", "coupling", "probe_contrast"}``:
        * ``per_instrument`` (C1) — for each instrument present, the Spearman of severity vs. the
          reading, the healthy/collapsed endpoint values, and a ``monotone`` flag (the correlation
          has the collapse-consistent sign and magnitude >= 0.9).
        * ``coupling`` (C2) — Spearman of RankMe@proj vs. each backbone probe accuracy (does
          projector geometry track backbone utility?) and vs. RankMe@backbone.
        * ``probe_contrast`` (C4) — healthy vs. collapsed backbone probe accuracy (the independent
          downstream anchor).
    """
    severity = [rung["severity"] for rung in ladder]
    per_instrument: dict[str, Any] = {}
    for key in (*_PROJ_INSTRUMENTS, *_BACKBONE_INSTRUMENTS, *_PROBES):
        readings = _series(ladder, key)
        if all(v is None for v in readings):
            continue
        rho = _spearman(severity, readings)
        sign = _COLLAPSE_SIGN[key]
        valid = [v for v in readings if v is not None]
        per_instrument[key] = {
            "spearman_vs_severity": rho,
            "expected_sign": sign,
            "healthy": readings[0],
            "collapsed": readings[-1],
            "monotone": bool(rho == rho and rho * sign >= 0.9),  # rho==rho drops nan
            "range": (max(valid) - min(valid)) if valid else float("nan"),
        }

    rankme_proj = _series(ladder, "rankme_proj")
    coupling = {
        "rankme_proj_vs_linear_acc": _spearman(rankme_proj, _series(ladder, "linear_acc")),
        "rankme_proj_vs_knn_acc": _spearman(rankme_proj, _series(ladder, "knn_acc")),
        "rankme_proj_vs_rankme_backbone": _spearman(rankme_proj, _series(ladder, "rankme")),
    }
    probe_contrast = {
        probe: {"healthy": _series(ladder, probe)[0], "collapsed": _series(ladder, probe)[-1]}
        for probe in _PROBES
        if not all(v is None for v in _series(ladder, probe))
    }
    return {"per_instrument": per_instrument, "coupling": coupling, "probe_contrast": probe_contrast}


def _render_ladder_table(ladder: list[dict[str, Any]]) -> str:
    """Render the severity ladder: final projector geometry + backbone probe accuracy per rung."""
    cols = ("severity", "rankme_proj", "var_proj", "unif_proj", "rankme_bb", "knn_acc", "linear_acc", "loss")
    keys = ("rankme_proj", "mean_feature_var_proj", "uniformity_proj", "rankme", "knn_acc", "linear_acc")
    lines = ["  ".join(f"{c:>12}" for c in cols), "  ".join("-" * 12 for _ in cols)]
    for rung in ladder:
        vals = [rung["severity"], *(rung["final"].get(k, float("nan")) for k in keys), rung["loss_floor"]]
        lines.append("  ".join(f"{v:>12.4f}" for v in vals))
    return "\n".join(lines)


def _render_monotonicity_summary(mono: dict[str, Any], knob: str) -> str:
    """Render the C1/C2/C4 reads (per-instrument monotonicity, coupling, probe contrast)."""
    lines = [f"MONOTONICITY [knob={knob}] — Spearman(severity, reading); |rho|>=0.9 w/ collapse sign => monotone"]
    for key, row in mono["per_instrument"].items():
        flag = "MONOTONE ✅" if row["monotone"] else "not monotone"
        lines.append(
            f"  {key:>22}: rho={row['spearman_vs_severity']:+.3f} (want sign {row['expected_sign']:+d})  "
            f"{row['healthy']:.4f} -> {row['collapsed']:.4f}  [{flag}]"
        )
    c = mono["coupling"]
    lines.append("  C2 coupling — RankMe@proj vs. backbone utility / geometry:")
    lines.append(
        f"    vs linear_acc rho={c['rankme_proj_vs_linear_acc']:+.3f}  "
        f"vs knn_acc rho={c['rankme_proj_vs_knn_acc']:+.3f}  "
        f"vs rankme@bb rho={c['rankme_proj_vs_rankme_backbone']:+.3f}"
    )
    if mono["probe_contrast"]:
        lines.append("  C4 anchor — backbone probe accuracy (healthy -> collapsed):")
        for probe, vals in mono["probe_contrast"].items():
            lines.append(f"    {probe}: {vals['healthy']:.4f} -> {vals['collapsed']:.4f}")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="partial_collapse")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the severity ladder for one knob + seed, then compute the three P0.2.5 reads."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    knob = config.sweep.knob
    severities = list(config.sweep.severities)

    ladder = [_run_rung(config, s, run_name=f"{knob}_s{str(s).replace('.', '')}", out_dir=out_dir) for s in severities]
    mono = _monotonicity(ladder)

    logger.info(f"partial-collapse ladder [knob={knob}, seed={config.seed}]\n" + _render_ladder_table(ladder))
    logger.info(_render_monotonicity_summary(mono, knob))

    payload = {
        "sweep": {"knob": knob, "severities": severities, "seed": int(config.seed)},
        "ladder": ladder,
        "monotonicity": mono,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"wrote ladder + monotonicity to {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
