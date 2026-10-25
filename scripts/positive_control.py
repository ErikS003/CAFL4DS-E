"""Phase-0 positive control (P0.2, recalibrated by P0.2.1) — the collapse-instrument gate.

Runs, in **one session**, two arms of the SAME SimSiam over the SAME **IID** STL-10 stream,
differing only by SimSiam's ``anti_collapse`` toggle:

* **PC** — anti-collapse DISABLED (predictor bypassed to ``p = z`` + stop-gradient off). Its trivial global
  optimum maps every input to one constant vector, so collapse is *mathematically forced*
  (loss → −1, RankMe → the ~1.9 projector-BatchNorm floor).
* **healthy** — SimSiam intact (predictor + stop-gradient ON). In the P0.2.1 fair-training
  regime its RankMe dips early then **re-expands and holds** (~5.8) — a genuine healthy
  baseline, cleanly separated from the collapse floor.

The P0.2.1 regime (why this is not P0.2's single ~25-step class-blocked pass): P0.2 found the
intact arm's RankMe decayed *indistinguishably* from the forced collapse at that toy horizon
(only the loss separated them). The fix is a fair training regime — the SAME tiny ViT and
img_size, but the full STL-10 train split, **IID** ordering, **multiple epochs**, and a
**warmup+cosine LR** — not a bigger model. IID isolates the ablation as the only collapse
cause (correlated-stream degradation is Phase 1). In this regime the discriminator flips:
both arms reach a low loss, so the gate separates on **RankMe**, not loss.

Both arms are **from-scratch**. The toggle is the *only* difference: the seed is reset before
building each arm so the two encoders start bit-identical and the augmentation RNG stays in
lockstep, so any divergence is attributable to anti-collapse alone. The gate is the
**contrast** between the two RankMe curves, checked numerically — see
``docs/experiments/phase0/P0.2.1.md``.

Examples:
    Default (STL-10, CPU)::

        uv run python scripts/positive_control.py

    Fast network-free smoke (synthetic has 100 imgs/class, so shrink the probe reservations)::

        uv run python scripts/positive_control.py data=synthetic img_size=16 epochs=8 \
            stream.support_per_class=8 stream.query_per_class=8 stream.era_eval_per_class=5

    On the Gaudi HPU (inside the container; see docs/developing.md)::

        ./scripts/run_gaudi_dev.sh -m /mnt/stl10 gaudi-env-cafl4ds:latest 0 \
            python scripts/positive_control.py device=hpu
"""

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from cafl4ds.jsonio import dumps_valid
from cafl4ds.loop import StreamingLoop
from cafl4ds.metric_envelope import metric_envelope, render_envelope_table
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")

_SPARK = "▁▂▃▄▅▆▇█"


def _run_arm(
    config: DictConfig, *, anti_collapse: bool, run_name: str, out_dir: Path
) -> tuple[list[dict[str, Any]], float]:
    """Build and run one arm of the positive control, returning its health series + loss floor.

    The global seed is reset here so both arms start from a bit-identical encoder init and
    draw the same augmentation sequence — the ``anti_collapse`` toggle is the only variable.
    Runs the P0.2.1 regime: ``config.epochs`` passes over the IID stream with a warmup+cosine
    LR schedule (the fair training horizon a healthy SimSiam baseline needs).

    Args:
        config: The composed ``positive_control`` config.
        anti_collapse: SimSiam anti-collapse toggle (``False`` = PC, ``True`` = healthy).
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        A ``(health_records, loss_floor)`` pair: the per-checkpoint health records, and the
        minimum SSL loss over *all* steps (robust to batch-to-batch noise — the "did the loss
        reach its −1 constant-solution floor?" signal).
    """
    torch.manual_seed(config.seed)  # identical init + augmentation RNG across arms
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder, anti_collapse=anti_collapse)
    apply_encoder_init(method.encoder, "from_scratch")  # PC must NOT load a warm start

    stream = instantiate(config.stream)  # same seed -> identical splits/order as the other arm
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    # One eval per `eval_every_epochs` epochs — probes (kNN/linear) are expensive, so a
    # per-step cadence over a multi-epoch run would dominate the wall-clock for no benefit.
    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(
        f"arm '{run_name}' (anti_collapse={anti_collapse}): {stream.num_eras} eras, "
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
    health = [r for r in records if r.get("series") == "health"]
    loss_floor = min(r["loss"] for r in records if r.get("series") == "loss")
    return health, loss_floor


def _spark(values: list[float], lo: float, hi: float) -> str:
    """Render values as a unicode sparkline scaled to the shared ``[lo, hi]`` range.

    Args:
        values: The series to render.
        lo: Lower bound of the shared scale (maps to the lowest block).
        hi: Upper bound of the shared scale (maps to the highest block).

    Returns:
        A one-line sparkline string.
    """
    span = hi - lo or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, max(0, int((v - lo) / span * (len(_SPARK) - 1))))] for v in values)


def _combined_table(pc: list[dict[str, Any]], hc: list[dict[str, Any]]) -> str:
    """Render both arms' loss + RankMe side by side, one row per aligned checkpoint.

    Args:
        pc: The PC arm's health records.
        hc: The healthy arm's health records.

    Returns:
        A fixed-width table string.
    """
    cols = ("step", "era", "pc_loss", "pc_rankme", "hc_loss", "hc_rankme")
    lines = ["  ".join(f"{c:>12}" for c in cols), "  ".join("-" * 12 for _ in cols)]
    for p, h in zip(pc, hc, strict=False):
        row = [p["step"], p["era"], p["loss"], p["rankme"], h["loss"], h["rankme"]]
        lines.append("  ".join(f"{v:>12.4f}" for v in row))
    return "\n".join(lines)


def _evaluate_point_collapse_gate(
    config: DictConfig,
    pc: list[dict[str, Any]],
    hc: list[dict[str, Any]],
    pc_loss_floor: float,
    hc_loss_floor: float,
) -> dict[str, Any]:
    """Apply the numeric pass criterion (the gate) — a RankMe-separation contrast (P0.2.1).

    Passes iff, in the P0.2.1 baseline regime: the PC's final RankMe drops to at most
    ``gate.pc_rankme_drop_frac`` of its OWN random-init RankMe (a large relative collapse); the
    PC's loss floor reaches ``<= gate.pc_loss_floor`` (rides to its −1 constant-solution floor —
    the "right reason" fingerprint); the intact control's final RankMe clears the absolute floor
    ``gate.healthy_rankme_min`` (a genuine healthy baseline — it re-expanded, not collapsed); and
    the two arms are separated on RankMe by ``healthy_final / pc_final >= gate.min_rankme_ratio``.

    Note the discriminator flip from P0.2: in this fair-training regime BOTH arms reach a low
    SSL loss (~−0.9), so the loss no longer separates them — RankMe does. Loss is used only for
    the PC's "right reason" fingerprint, never to gate the healthy arm.

    Args:
        config: The composed config (its ``gate`` block holds the thresholds).
        pc: The PC arm's health records.
        hc: The healthy arm's health records.
        pc_loss_floor: Minimum SSL loss the PC reached over all steps.
        hc_loss_floor: Minimum SSL loss the intact control reached over all steps.

    Returns:
        A dict of the measured numbers, per-condition booleans, and the overall ``passed``.
    """
    g = config.gate
    pc_init, pc_final = pc[0]["rankme"], pc[-1]["rankme"]
    hc_final = hc[-1]["rankme"]
    pc_drop_frac = pc_final / pc_init if pc_init else 1.0
    rankme_ratio = hc_final / pc_final if pc_final else float("inf")

    checks = {
        "pc_collapses_relative": pc_drop_frac <= g.pc_rankme_drop_frac,
        "pc_right_reason": pc_loss_floor <= g.pc_loss_floor,
        "healthy_holds": hc_final >= g.healthy_rankme_min,
        "rankme_separated": rankme_ratio >= g.min_rankme_ratio,
    }
    return {
        "pc_rankme_init": pc_init,
        "pc_rankme_final": pc_final,
        "pc_rankme_drop_frac": pc_drop_frac,
        "pc_loss_floor": pc_loss_floor,
        "healthy_rankme_final": hc_final,
        "healthy_loss_floor": hc_loss_floor,
        "rankme_ratio": rankme_ratio,
        "loss_separation": hc_loss_floor - pc_loss_floor,  # reported only — no longer a discriminator
        "mode": "point_collapse",
        "thresholds": {
            "pc_rankme_drop_frac": g.pc_rankme_drop_frac,
            "pc_loss_floor": g.pc_loss_floor,
            "healthy_rankme_min": g.healthy_rankme_min,
            "min_rankme_ratio": g.min_rankme_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _envelope_row(envelope: list[dict[str, Any]], metric: str, surface: str) -> dict[str, Any] | None:
    """Return the collapse-suite envelope row for ``(metric, surface)`` if present, else ``None``."""
    for row in envelope:
        if row["metric"] == metric and row["surface"] == surface:
            return row
    return None


def _evaluate_redundancy_gate(config: DictConfig, envelope: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the redundancy-/dimensional-collapse gate (P0.2.3) off the collapse-suite envelope.

    The point-collapse gate above is the wrong instrument for a decorrelation-method vehicle
    (Barlow Twins with its redundancy-reduction term ablated): the projector BatchNorm holds
    per-feature variance at unit and the loss floor is not the SimSiam −1 constant-solution
    value, so a RankMe-drop + loss-floor read misfires. Redundancy collapse leaves a *different*
    fingerprint, read here from the method-agnostic envelope at one surface (``gate.surface``):

    * ``offdiag_fires`` — off-diagonal covariance **fires**: PC/healthy separation
      ``>= gate.min_offdiag_ratio``. The headline test — the *fire* side of ``offdiag_cov`` that
      P0.2.2's point-collapse vehicle could never reach.
    * ``variance_quiet`` — per-dimension variance **stays quiet**: the healthy/PC separation is
      *below* ``gate.max_variance_ratio`` (it does NOT clear the fire bar). This is the direct
      discriminator between redundancy and *point* collapse — under point collapse variance would
      crater; here BatchNorm preserves it.
    * ``rankme_corroborates`` — RankMe **fires** too (``>= gate.min_rankme_ratio``): rank-based
      detectors see dimensional collapse, so the low-rank subspace should register here as well.

    Args:
        config: The composed config (its ``gate`` block holds the thresholds + ``surface``).
        envelope: The per-``(instrument × surface)`` collapse-suite envelope (``metric_envelope``).

    Returns:
        A dict of the measured separations, per-condition booleans, and the overall ``passed``
        (all three checks — the two-sided ``offdiag_cov`` verdict plus the variance discriminator
        and the rank corroboration).
    """
    g = config.gate
    surface = g.surface
    offdiag = _envelope_row(envelope, "offdiag_cov", surface)
    variance = _envelope_row(envelope, "mean_feature_var", surface)
    rankme = _envelope_row(envelope, "rankme", surface)

    # Absent row => the instrument could not be read at this surface: score it as non-firing
    # (offdiag/rankme) or as "not proven quiet" (variance) so a missing signal never passes.
    offdiag_sep = offdiag["separation"] if offdiag else 0.0
    variance_sep = variance["separation"] if variance else float("inf")
    rankme_sep = rankme["separation"] if rankme else 0.0

    checks = {
        "offdiag_fires": offdiag_sep >= g.min_offdiag_ratio,
        "variance_quiet": variance_sep < g.max_variance_ratio,
        "rankme_corroborates": rankme_sep >= g.min_rankme_ratio,
    }
    return {
        "mode": "redundancy_collapse",
        "surface": surface,
        "offdiag_separation": offdiag_sep,
        "offdiag_healthy_final": offdiag["healthy_final"] if offdiag else None,
        "offdiag_pc_final": offdiag["pc_final"] if offdiag else None,
        "variance_separation": variance_sep,
        "variance_healthy_final": variance["healthy_final"] if variance else None,
        "variance_pc_final": variance["pc_final"] if variance else None,
        "rankme_separation": rankme_sep,
        "thresholds": {
            "min_offdiag_ratio": g.min_offdiag_ratio,
            "max_variance_ratio": g.max_variance_ratio,
            "min_rankme_ratio": g.min_rankme_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_gate(
    config: DictConfig,
    pc: list[dict[str, Any]],
    hc: list[dict[str, Any]],
    pc_loss_floor: float,
    hc_loss_floor: float,
    envelope: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dispatch to the failure-mode-appropriate gate (``config.gate.mode``, default point).

    ``point_collapse`` (P0.2.1) reads the RankMe-separation + loss-floor-to-−1 fingerprint of
    SimSiam point collapse; ``redundancy_collapse`` (P0.2.3) reads the collapse-suite envelope
    for the off-diagonal-covariance fire + feature-variance-quiet fingerprint a decorrelation
    vehicle leaves. Both return a dict tagged with its ``mode`` for the summary renderer.
    """
    if config.gate.get("mode", "point_collapse") == "redundancy_collapse":
        return _evaluate_redundancy_gate(config, envelope)
    return _evaluate_point_collapse_gate(config, pc, hc, pc_loss_floor, hc_loss_floor)


def _render_point_collapse_summary(gate: dict[str, Any]) -> str:
    """Render the P0.2.1 point-collapse gate result (RankMe separation + loss-floor fingerprint)."""
    t = gate["thresholds"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    return (
        f"POSITIVE-CONTROL GATE [point collapse]: {verdict}\n"
        f"  PC RankMe    {gate['pc_rankme_init']:.3f} (init) -> {gate['pc_rankme_final']:.3f} "
        f"= {gate['pc_rankme_drop_frac'] * 100:.1f}% of init "
        f"(<= {t['pc_rankme_drop_frac'] * 100:.0f}%?  {gate['checks']['pc_collapses_relative']})\n"
        f"  PC loss floor  = {gate['pc_loss_floor']:.4f} "
        f"(<= {t['pc_loss_floor']} -> rides to -1 constant-solution floor?  {gate['checks']['pc_right_reason']})\n"
        f"  healthy RankMe final = {gate['healthy_rankme_final']:.3f} "
        f"(>= {t['healthy_rankme_min']} absolute floor?  {gate['checks']['healthy_holds']})\n"
        f"  RankMe separation (healthy / PC) = {gate['rankme_ratio']:.2f}x "
        f"(>= {t['min_rankme_ratio']}x?  {gate['checks']['rankme_separated']})\n"
        f"  [reported] loss-floor gap (healthy - PC) = {gate['loss_separation']:.3f} "
        f"(NOT gated — both arms reach a low loss in this regime)"
    )


def _render_redundancy_summary(gate: dict[str, Any]) -> str:
    """Render the P0.2.3 redundancy-collapse gate result (offdiag fires, variance stays quiet)."""
    t, c = gate["thresholds"], gate["checks"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    return (
        f"POSITIVE-CONTROL GATE [redundancy collapse, surface={gate['surface']}]: {verdict}\n"
        f"  offdiag_cov separation (PC / healthy) = {gate['offdiag_separation']:.2f}x "
        f"(>= {t['min_offdiag_ratio']}x -> FIRES?  {c['offdiag_fires']})  "
        f"[PC {gate['offdiag_pc_final']:.4f} vs healthy {gate['offdiag_healthy_final']:.4f}]\n"
        f"  mean_feature_var separation (healthy / PC) = {gate['variance_separation']:.2f}x "
        f"(< {t['max_variance_ratio']}x -> STAYS QUIET (discriminator vs point collapse)?  {c['variance_quiet']})\n"
        f"  RankMe separation (healthy / PC) = {gate['rankme_separation']:.2f}x "
        f"(>= {t['min_rankme_ratio']}x -> corroborates (rank sees dimensional collapse)?  {c['rankme_corroborates']})"
    )


def _render_gate_summary(gate: dict[str, Any]) -> str:
    """Render whichever gate ran (``gate['mode']``) into its human-readable verdict block."""
    if gate["mode"] == "redundancy_collapse":
        return _render_redundancy_summary(gate)
    return _render_point_collapse_summary(gate)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run both arms, render the contrast, and apply the numeric gate."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    hc, hc_loss_floor = _run_arm(config, anti_collapse=True, run_name="simsiam_healthy", out_dir=out_dir)
    pc, pc_loss_floor = _run_arm(config, anti_collapse=False, run_name="simsiam_pc", out_dir=out_dir)

    table = _combined_table(pc, hc)
    # Shared scale so the two RankMe sparklines are directly comparable.
    all_rankme = [r["rankme"] for r in pc] + [r["rankme"] for r in hc]
    lo, hi = min(all_rankme), max(all_rankme)
    curves = (
        f"RankMe curves (shared scale {lo:.2f}..{hi:.2f}):\n"
        f"  PC (collapse) : {_spark([r['rankme'] for r in pc], lo, hi)}\n"
        f"  healthy       : {_spark([r['rankme'] for r in hc], lo, hi)}\n"
        f"PC loss (toward -1 floor, scale -1..0):\n"
        f"  PC (collapse) : {_spark([r['loss'] for r in pc], -1.0, 0.0)}"
    )

    # P0.2.2 reporting layer: the per-(instrument × surface) separation + fire/quiet map for the
    # WHOLE collapse suite. Under `gate.mode=redundancy_collapse` (P0.2.3) it also FEEDS the gate,
    # so it is computed before the gate; otherwise it is the calibration map alongside the RankMe
    # gate. Reads whatever surface metrics the monitor logged into the health series.
    envelope = metric_envelope(pc, hc)

    gate = _evaluate_gate(config, pc, hc, pc_loss_floor, hc_loss_floor, envelope)
    summary = _render_gate_summary(gate)

    logger.info("positive control — side-by-side (aligned checkpoints)\n" + table)
    logger.info(curves)
    logger.info(summary)
    logger.info(
        "collapse-suite envelope (P0.2.2 — separation + fire/quiet per instrument × surface):\n"
        + render_envelope_table(envelope)
    )

    (out_dir / "comparison.json").write_text(
        dumps_valid({"gate": gate, "envelope": envelope, "pc": pc, "healthy": hc}),
        encoding="utf-8",
    )
    logger.info(f"wrote comparison + gate + envelope to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass. Per P0.2: if the PC does not leave its expected collapse fingerprint (or "
            "not for the right reason), the instrument/wiring is suspect — investigate before trusting "
            "downstream numbers."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
