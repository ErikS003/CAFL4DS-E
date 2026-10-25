r"""Phase-0 false-positive stress (P0.2.4) — the quiet-side calibration of the collapse suite.

P0.2/P0.2.1/P0.2.2/P0.2.3 all calibrate how a forced-collapse (PC) arm **separates** from a
healthy arm. None stresses the *quiet* side: the "healthy" arm is always a well-behaved SimSiam,
so the suite's **false-positive rate** — a detector firing on a healthy-but-atypical
representation — is untested. In the coupled active-learning loop a false fire mis-aims selection,
so this is the failure mode that most hurts.

This harness has **no PC arm**. It runs two **healthy** SimSiam arms (``anti_collapse=True``) over
the SAME session:

* **reference** — the canonical-healthy baseline (the base config; byte-identical to the positive
  control's healthy arm, so readings are directly comparable).
* **atypical** — the same healthy SimSiam pushed into a regime that *superficially resembles*
  collapse by the selected ``stressor`` (low-diversity data / heavy augmentation / undertraining),
  applied to this arm only via ``stressor.overrides`` merged over the base config.

The atypical arm is fed into the ``pc`` slot of :func:`~cafl4ds.metric_envelope.metric_envelope`,
so the existing separation machinery is reused unchanged. A **false fire** = the atypical arm
separates from the reference in the collapse direction past its bar. The verdict is **inverted**
relative to the positive control: the run PASSES iff **no** instrument fires (:func:`quiet_verdict`).

Unlike the positive control, a *fire here is a legitimate scientific outcome* (a documented
false-positive boundary of that instrument), not a wiring bug — so this script logs the fired
instruments but does **not** exit non-zero. See ``docs/experiments/phase0/P0.2.4.md``.

Examples:
    Default (STL-10, CPU), low-diversity stressor::

        uv run python scripts/false_positive_stress.py stressor=low_diversity

    Fast network-free smoke (synthetic has 100 imgs/class, so shrink the reservations + volume)::

        uv run python scripts/false_positive_stress.py data=synthetic img_size=16 epochs=4 \\
            stream.support_per_class=8 stream.query_per_class=8 stream.era_eval_per_class=5 \\
            monitor.run_knn=false monitor.run_linear=false

    Option B (ablation) — add a matched forced-collapse arm in the SAME stressed regime::

        uv run python scripts/false_positive_stress.py stressor=low_diversity collapse_pole=true
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.loop import StreamingLoop
from cafl4ds.metric_envelope import metric_envelope, quiet_verdict, render_envelope_table
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")


def _run_arm(
    config: DictConfig,
    *,
    run_name: str,
    out_dir: Path,
    anti_collapse: bool = True,
    eval_classes: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build and run one arm, returning its health series.

    Mirrors ``scripts/positive_control.py``'s ``_run_arm``. The default (``anti_collapse=True``) is
    a **healthy** arm — the P0.2.4 variable is the *config* (reference vs. stressor-perturbed), not
    the anti-collapse switch. Option B (``collapse_pole=true``) flips ``anti_collapse=False`` to run
    a matched **forced-collapse** arm inside a stressed regime. The global seed is reset so the run
    is reproducible.

    Args:
        config: The (base or stressor-merged) composed config for this arm.
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.
        anti_collapse: SimSiam anti-collapse toggle (``True`` = healthy, ``False`` = collapse pole).
        eval_classes: If set, restrict this arm's monitor to the given class subset (*matched-class
            eval*, A2). The canonical reference arm trains on all classes but is *evaluated* on the
            atypical arm's few-class subset, so the effective-rank cap is matched across arms and the
            low-diversity RankMe false fire — an eval-set artifact — cancels. ``None`` leaves the
            arm's full eval set in place (byte-identical to the pre-A2 behaviour).

    Returns:
        The per-checkpoint health records.
    """
    torch.manual_seed(config.seed)
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder, anti_collapse=anti_collapse)
    apply_encoder_init(method.encoder, "from_scratch")

    stream = instantiate(config.stream)
    eval_sets = stream.eval_sets if eval_classes is None else stream.eval_sets.restrict_to_classes(eval_classes)
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=eval_sets)

    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(
        f"arm '{run_name}': {stream.num_eras} eras, "
        f"{batches_per_epoch} batches x {config.epochs} epochs = {total_steps} steps"
    )

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
    return [r for r in records if r.get("series") == "health"]


def _apply_stressor(config: DictConfig) -> DictConfig:
    """Return a copy of ``config`` with the selected stressor's overrides merged in (test arm).

    The stressor config carries an ``overrides`` mapping (e.g. ``stream.class_order``, ``ssl.*``
    augmentation knobs, ``epochs``/``optim.lr``); it is merged over a deep copy of the base config
    with struct mode off so *new* keys (a class subset, augmentation params absent from the base)
    are allowed. The reference arm is left untouched.
    """
    test = copy.deepcopy(config)
    OmegaConf.set_struct(test, False)
    overrides = config.stressor.get("overrides", {}) or {}
    return OmegaConf.merge(test, overrides)


def _evaluate_quiet_gate(config: DictConfig, envelope: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap :func:`quiet_verdict` with the run's stressor label and fire bars for the record."""
    verdict = quiet_verdict(envelope)
    verdict["stressor"] = config.stressor.name
    verdict["fire_ratio"] = config.fire_ratio
    verdict["fire_gap"] = config.fire_gap
    verdict["match_eval_classes"] = bool(config.get("match_eval_classes", False))
    return verdict


def _render_quiet_summary(gate: dict[str, Any]) -> str:
    """Render the inverted (quiet) verdict — pass iff no instrument false-fired."""
    verdict = "PASS ✅ (quiet side holds)" if gate["passed"] else "FALSE FIRE ⚠️"
    lines = [
        f"FALSE-POSITIVE STRESS [stressor={gate['stressor']}, "
        f"bars ratio>={gate['fire_ratio']} / gap>={gate['fire_gap']}]: {verdict}"
    ]
    if gate["fired"]:
        for f in gate["fired"]:
            lines.append(
                f"  FALSE FIRE: {f['metric']} @ {f['surface']} — "
                f"separation {f['separation']:.2f} >= bar {f['threshold']} "
                f"(atypical-healthy tripped a collapse detector: a false-positive boundary)"
            )
    else:
        lines.append("  no instrument separated the atypical-healthy arm from the reference past its bar")
    return "\n".join(lines)


def _instrument_positions(
    reference: list[dict[str, Any]],
    atypical: list[dict[str, Any]],
    collapse: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Locate the atypical-healthy arm on the [collapse-pole, canonical-healthy] axis (Option B).

    For each ``(instrument × surface)`` present in the envelope, take the three arms' final
    readings — ``h`` (canonical-healthy reference), ``a`` (atypical-healthy), ``c`` (matched
    collapse-in-regime) — and compute the linear position ``pos = (a - c) / (h - c)``. This is
    direction-agnostic: ``pos = 1`` sits at the healthy pole, ``pos = 0`` at the collapse floor, so
    ``pos > 0.5`` means the atypical arm reads *nearer healthy than collapse*. The ``h`` pole is the
    canonical (out-of-regime) baseline, so for the low-diversity regime it carries the eval-set-cap
    caveat — the confound-free readout is the in-regime discrimination envelope, not this position.

    Args:
        reference: The canonical-healthy arm's health series (the ``h`` pole).
        atypical: The atypical-healthy arm's health series (the point being located, ``a``).
        collapse: The matched collapse-in-regime arm's health series (the ``c`` pole).

    Returns:
        One row per ``(metric, surface)`` with the three finals, ``pos``, and ``nearer_healthy``.
    """
    # Reuse the envelope's (metric, surface) enumeration; the separation column is irrelevant here.
    rows: list[dict[str, Any]] = []
    for spec_row in metric_envelope(atypical, reference):
        key = spec_row["metric"] + ("" if spec_row["surface"] == "backbone" else "_proj")
        h = [r[key] for r in reference if isinstance(r.get(key), int | float)]
        a = [r[key] for r in atypical if isinstance(r.get(key), int | float)]
        c = [r[key] for r in collapse if isinstance(r.get(key), int | float)]
        if not (h and a and c):
            continue
        h_f, a_f, c_f = h[-1], a[-1], c[-1]
        span = h_f - c_f
        pos = (a_f - c_f) / span if span else float("nan")
        rows.append(
            {
                "metric": spec_row["metric"],
                "surface": spec_row["surface"],
                "healthy_pole": h_f,
                "atypical": a_f,
                "collapse_pole": c_f,
                "pos": pos,
                "nearer_healthy": bool(pos > 0.5),
            }
        )
    return rows


def _render_pole_crosstab(
    envelope: list[dict[str, Any]],
    pole_envelope: list[dict[str, Any]],
    positions: list[dict[str, Any]],
) -> str:
    """Cross-tab Option A (apparent FP) against Option B (in-regime discrimination + position).

    Per ``(instrument × surface)``: A = does the atypical arm separate from the canonical reference
    (the Option A false-fire reading)? B = does a matched collapse arm still separate from the
    atypical arm *within the regime* (confound-free)? Plus the position of the atypical arm on the
    [collapse, healthy] axis. The verdict disambiguates each Option-A fire: ``A fires & B fires`` =>
    the instrument keeps its discriminative power in-regime, so the Option-A fire is a reference /
    eval-set artifact; ``A fires & B quiet`` => a genuine in-regime blind spot.
    """
    pole_fires = {(r["metric"], r["surface"]): r["fires"] for r in pole_envelope}
    pole_sep = {(r["metric"], r["surface"]): r["separation"] for r in pole_envelope}
    pos_by = {(r["metric"], r["surface"]): r for r in positions}
    cols = ("metric", "surface", "A:atyp/ref", "A?", "B:coll/atyp", "B?", "pos", "verdict")
    widths = {"metric": 16, "surface": 8, "A?": 5, "B?": 5, "pos": 6, "verdict": 34}
    lines = ["  ".join(f"{c:>{widths.get(c, 12)}}" for c in cols), "  ".join("-" * widths.get(c, 12) for c in cols)]
    for row in envelope:
        if not row["standalone"]:
            continue
        k = (row["metric"], row["surface"])
        a_fires, b_fires = row["fires"], pole_fires.get(k, False)
        pos = pos_by.get(k, {}).get("pos", float("nan"))
        if not a_fires:
            verdict = "A quiet (no false fire)"
        elif b_fires:
            verdict = "A=eval-set/ref artifact (B discriminates)"
        else:
            verdict = "genuine in-regime blind spot"
        cells = [
            f"{row['metric']:>16}",
            f"{row['surface']:>8}",
            f"{row['separation']:>12.2f}",
            f"{('yes' if a_fires else 'no'):>5}",
            f"{pole_sep.get(k, float('nan')):>12.2f}",
            f"{('yes' if b_fires else 'no'):>5}",
            f"{pos:>6.2f}",
            f"{verdict:>34}",
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="false_positive_stress")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the reference + atypical healthy arms, then apply the inverted (quiet) verdict."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    test_config = _apply_stressor(config)

    # A2 matched-class eval: when the stressor subsets classes (low-diversity) and
    # `match_eval_classes` is on, evaluate the full-class reference arm on the *same* few-class
    # subset the atypical arm sees, so the effective-rank cap cancels and the RankMe eval-set
    # false fire disappears. A no-op for stressors that do not subset classes (heavy_aug /
    # undertrained keep all 10), and off by default (byte-identical to the pre-A2 Option-A run).
    eval_classes: list[int] | None = None
    if config.get("match_eval_classes", False):
        subset = test_config.stream.get("class_order")
        eval_classes = list(OmegaConf.to_container(subset)) if subset is not None else None

    reference = _run_arm(config, run_name="reference_healthy", out_dir=out_dir, eval_classes=eval_classes)
    atypical = _run_arm(test_config, run_name=f"atypical_{config.stressor.name}", out_dir=out_dir)

    # Feed the atypical arm into the `pc` slot: a false fire = it separates from the canonical
    # healthy reference in the collapse direction past its bar (same envelope machinery as P0.2.2).
    envelope = metric_envelope(atypical, reference, min_ratio=config.fire_ratio, min_gap=config.fire_gap)
    gate = _evaluate_quiet_gate(config, envelope)

    logger.info(_render_quiet_summary(gate))
    logger.info(
        "collapse-suite envelope (atypical-in-PC-slot vs. reference — separation + fire/quiet per "
        "instrument × surface):\n" + render_envelope_table(envelope)
    )

    payload: dict[str, Any] = {"gate": gate, "envelope": envelope, "atypical": atypical, "reference": reference}

    # Option B (ablation): a matched forced-collapse arm in the SAME stressed regime. Feeding it
    # into the `pc` slot vs. the atypical-healthy arm gives the confound-free in-regime separation;
    # `_instrument_positions` places the atypical arm on the [collapse, canonical-healthy] axis.
    if config.get("collapse_pole", False):
        collapse = _run_arm(
            test_config, run_name=f"collapse_{config.stressor.name}", out_dir=out_dir, anti_collapse=False
        )
        pole_envelope = metric_envelope(collapse, atypical, min_ratio=config.fire_ratio, min_gap=config.fire_gap)
        positions = _instrument_positions(reference, atypical, collapse)
        payload.update({"pole_envelope": pole_envelope, "positions": positions, "collapse": collapse})
        logger.info(
            "OPTION B — matched collapse-pole per regime (A: atypical-vs-reference false fire; "
            "B: collapse-vs-atypical in-regime discrimination; pos on [collapse, healthy] axis):\n"
            + _render_pole_crosstab(envelope, pole_envelope, positions)
        )

    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"wrote comparison + quiet gate + envelope to {out_dir / 'comparison.json'}")

    # NB: a false fire is a documented calibration outcome (a false-positive boundary), NOT a wiring
    # bug — so, unlike the positive control, this script does not exit non-zero on a fire.
    if not gate["passed"]:
        logger.warning(
            f"{gate['num_fired']} instrument(s) false-fired on the '{gate['stressor']}' regime — "
            "document as a known false-positive boundary (see docs/experiments/phase0/P0.2.4.md)."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
