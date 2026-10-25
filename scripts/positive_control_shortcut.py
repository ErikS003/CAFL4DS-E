"""Phase-0 positive control (P0.5) — the MAE shortcut / stable-loss degradation gate.

Runs, in **one session**, two arms of the SAME from-scratch MAE over the SAME **IID** stream,
differing only by ONE shortcut lever:

* **shortcut** — the lever relaxed so the encoder can take a low-level shortcut (a **low mask
  ratio**, P0.5.0; or an **over-powered decoder**, P0.5.1). Its reconstruction loss stays
  low/healthy-looking while its frozen-probe accuracy **craters** — the "loss lies" case.
* **healthy** — MAE at the standard operating point (mask 0.75, lightweight decoder), which
  reaches a **good** frozen-probe accuracy (real headroom above chance).

The lever is the *only* difference: the seed is reset before building each arm so the two
encoders start bit-identical and their augmentation RNG stays in lockstep, so any divergence is
attributable to the lever alone. The init factor (``init=``) selects the vehicle: **from-scratch**
(P0.5.0/P0.5.1 — the clean producer of a full-rank-but-useless rep) or **pretrained-continue** from
a competent, capacity-gated well (P0.5.2 — degrade-*from-competent*, where a smaller competent model
has less slack so the shortcut bites; pretrain the well first with ``scripts/pretrain.py`` at the
same encoder capacity).

**What the gate is, and is not.** The failure mode is a representation that is **full-rank but
useless**, so RankMe staying high is the mode's **precondition**, not a fire (a RankMe crater would
mean P0.2 collapse, not this mode). The gate certifies the mode was *induced* (precondition +
probe crater + the loss does not alarm + contrast). It deliberately does **not** gate on the
label-free reader: the study's real deliverable is the **reader verdict** — does `alignment` /
`uniformity` track the probe's crater on an MAE backbone, or does the P0.2 suite have a quality
blind spot? — and a fire *or* a clean null is a legitimate result, so the verdict is **reported**
either way. On the P0.5.2 degrade-from-competent vehicle the panel also carries the MAE-native
candidates (`clusterability`, `attn_distance`, `alignment_strong`); a fire closes the mode's
portfolio gap, a clean null against a genuine crater is a strong earned negative. See
``docs/experiments/phase0/P0.5.0.md``, ``P0.5.1.md`` and ``P0.5.2.md``.

Examples:
    Fast network-free smoke (synthetic; 4 patches at img16, so keep the shortcut ratio >= ~0.25)::

        uv run python scripts/positive_control_shortcut.py data=synthetic img_size=16 epochs=6 \
            stream.support_per_class=8 stream.query_per_class=8 stream.era_eval_per_class=5 \
            'shortcut.mask_ratio=0.25'

    On the Gaudi HPU (inside the container; see docs/developing.md)::

        ./scripts/run_gaudi_dev.sh -m /mnt gaudi-env-cafl4ds:latest 0 \
            python scripts/positive_control_shortcut.py device=hpu
"""

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.jsonio import dumps_valid
from cafl4ds.loop import StreamingLoop
from cafl4ds.metric_envelope import metric_envelope, render_envelope_table
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")

# The label-free quality-reader candidates ON TRIAL in P0.5 (read at the backbone — MAE's only
# surface), each paired with the SIGN of its degradation direction. A reader "fires" when the
# sign-normalized gap ``sign * (shortcut_final - healthy_final)`` clears the reader's threshold —
# i.e. the shortcut arm is worse in that reader's own worsening direction:
#   * +1 (climbs when worse): `uniformity` / `alignment` / `alignment_strong` — a shortcut rep is
#     less uniform (uniformity climbs toward 0) and less augmentation-invariant (alignment distance
#     grows). uniformity/alignment are the P0.5.0/P0.5.1 carry-ins (the P0.2 suite, expected blind);
#     alignment_strong is the P0.5.2 stronger-augment repurpose.
#   * -1 (drops when worse): `clusterability` (silhouette falls) / `attn_distance` (heads go local)
#     — the P0.5.2 candidate MAE-native quality readers.
_QUALITY_READERS: tuple[tuple[str, float], ...] = (
    ("uniformity", +1.0),
    ("alignment", +1.0),
    ("alignment_strong", +1.0),
    ("clusterability", -1.0),
    ("attn_distance", -1.0),
)


def _arm_overrides(config: DictConfig, arm_name: str) -> dict[str, Any]:
    """Return the per-arm ``build_mae`` keyword overrides (the shortcut lever) as a plain dict.

    Args:
        config: The composed config (its ``healthy`` / ``shortcut`` blocks hold the lever).
        arm_name: ``"healthy"`` or ``"shortcut"``.

    Returns:
        A ``{param: value}`` dict merged as kwargs into ``instantiate(config.ssl, ...)`` — e.g.
        ``{"mask_ratio": 0.1}`` (P0.5.0) or ``{"decoder_dim": 768, "decoder_depth": 12}`` (P0.5.1).
    """
    block = config.get(arm_name)
    if block is None:
        return {}
    return dict(OmegaConf.to_container(block, resolve=True))


def _run_arm(
    config: DictConfig, *, arm_name: str, run_name: str, out_dir: Path
) -> tuple[list[dict[str, Any]], float, dict[str, float] | None]:
    """Build and run one arm of the positive control, returning its health series + loss floor.

    The global seed is reset here so both arms start from a bit-identical encoder init and draw the
    same augmentation sequence — the shortcut lever (``config[arm_name]``) is the only variable.
    The init factor (``config.init.mode``) selects the vehicle: ``from_scratch`` (P0.5.0/P0.5.1) or
    ``pretrained`` continue-from-a-competent-well (P0.5.2).

    Args:
        config: The composed ``positive_control_shortcut`` config.
        arm_name: ``"healthy"`` or ``"shortcut"`` — selects the per-arm lever overrides.
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        A ``(health_records, loss_floor, well_baseline)`` triple: the per-checkpoint health records;
        the minimum reconstruction loss over *all* steps (the "how low did the loss look" summary the
        anti-instrument check reads); and — for the ``pretrained`` vehicle only — the **pristine
        well baseline**, a full health read of the loaded well taken *before* the first continue
        gradient (``None`` for ``from_scratch``, where a pre-training baseline is just the random
        floor). See the warm-up-confound note below.
    """
    torch.manual_seed(config.seed)  # identical init + augmentation RNG across arms
    encoder = instantiate(config.encoder)
    overrides = _arm_overrides(config, arm_name)
    method = instantiate(config.ssl, encoder=encoder, **overrides)
    # Init factor (I): from_scratch (P0.5.0/P0.5.1, the default) or pretrained-continue from a
    # competent, capacity-gated well (P0.5.2 degrade-from-competent). Both arms load the SAME
    # encoder checkpoint (seed reset keeps their fresh decoders bit-identical too), so the shortcut
    # lever stays the only variable. The checkpoint defaults to <pretrain_dir>/<method_name>.pt.
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)

    stream = instantiate(config.stream)  # same seed -> identical splits/order as the other arm
    optimizer = instantiate(config.optim, params=method.parameters())
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    # Warm-up-confound guard (the P0.3.9 lesson). The checkpoint is encoder-only, so a `pretrained`
    # continue splices a FRESH decoder onto the pretrained encoder; its early gradients perturb the
    # encoder — a warm-up transient that drops the probe at the START of the continue, on the SAME
    # data. The final-step cross-arm crater is robust to it (the transient is shared, and we do NOT
    # swap datasets), but the ABSOLUTE healthy-headroom bar is NOT: a low healthy_final could be an
    # incompetent well OR a competent well the continue damaged. So probe the PRISTINE well once,
    # pre-gradient (step -1), on a throwaway monitor — the real run's monitor state (drift refs) is
    # untouched — and require the healthy arm to RECOVER to this baseline for its headroom to count.
    well_baseline: dict[str, float] | None = None
    if config.init.mode == "pretrained":
        method.to(config.device)
        well_baseline = instantiate(config.monitor, eval_sets=stream.eval_sets).measure(method, -1)
        logger.info(
            f"arm '{run_name}' pristine well baseline (pre-continue): knn={well_baseline['knn_acc']:.3f} "
            f"linear={well_baseline['linear_acc']:.3f} rankme={well_baseline['rankme']:.3f}"
        )

    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"arm '{run_name}' (overrides={overrides}): {batches_per_epoch} batches x {config.epochs} epochs")

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
    return health, loss_floor, well_baseline


def _reader_verdict(
    shortcut: list[dict[str, Any]],
    healthy: list[dict[str, Any]],
    reader_min_gap: float,
    reader_min_gaps: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The DELIVERABLE, reported either way: does any label-free quality reader track the crater?

    For each candidate reader (:data:`_QUALITY_READERS`) at the backbone, report both arms' final
    value and the **sign-normalized** degradation gap ``sign * (shortcut - healthy)`` — positive
    when the shortcut arm is worse in that reader's own worsening direction (see
    :data:`_QUALITY_READERS`). A reader **fires** if that gap clears its threshold (per-reader from
    ``reader_min_gaps``, else the ``reader_min_gap`` default — readers live on different scales, so
    a single bar does not fit all). This never gates ``passed`` — a clean null ("nothing fires →
    the P0.2 suite has a quality blind spot on MAE and the MAE-native candidates do not close it")
    is a legitimate result.

    Args:
        shortcut: The shortcut arm's health records.
        healthy: The healthy arm's health records.
        reader_min_gap: Default degradation-gap threshold for readers without a specific entry.
        reader_min_gaps: Optional per-reader threshold overrides (readers on different scales).

    Returns:
        A dict with per-reader ``{reader, healthy_final, shortcut_final, degrade_gap, threshold,
        fires}`` rows, the list of readers that fired, and a one-line ``headline``.
    """
    per_reader = reader_min_gaps or {}
    rows: list[dict[str, Any]] = []
    for reader, sign in _QUALITY_READERS:
        if reader not in shortcut[-1] or reader not in healthy[-1]:
            continue  # reader not measured for this method / config (e.g. no positive pair)
        hc_final, sc_final = healthy[-1][reader], shortcut[-1][reader]
        gap = sign * (sc_final - hc_final)  # sign-normalized: > 0 => shortcut worse
        threshold = float(per_reader.get(reader, reader_min_gap))
        rows.append(
            {
                "reader": reader,
                "healthy_final": hc_final,
                "shortcut_final": sc_final,
                "degrade_gap": gap,
                "threshold": threshold,
                "fires": bool(gap >= threshold),
            }
        )
    fired = [r["reader"] for r in rows if r["fires"]]
    headline = (
        f"reader(s) fired: {', '.join(fired)}"
        if fired
        else "NO reader fired -> no label-free reader tracks the crater (P0.2 suite blind + MAE-native candidates)"
    )
    return {"candidates": rows, "fired": fired, "any_reader_fired": bool(fired), "headline": headline}


def _evaluate_shortcut_gate(
    config: DictConfig,
    shortcut: list[dict[str, Any]],
    healthy: list[dict[str, Any]],
    shortcut_loss_floor: float,
    healthy_loss_floor: float,
    well_baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the mode-induction gate (P0.5) and attach the reported reader verdict.

    ``passed`` reads ONLY the four mode-induction conditions — the reader verdict is reported, never
    gated (a fire and a clean null are both legitimate):

    * ``healthy_has_headroom`` — the healthy arm's frozen **kNN** probe (primary) clears ``healthy_probe_min``
      (a real representation, not a floor-bound one — the P0.3.0 headroom lesson).
    * ``rankme_preserved`` — the **precondition**: the shortcut rep is full-rank, i.e. its backbone
      RankMe is at least ``precondition_rankme_frac`` of the healthy arm's (a large drop would mean
      collapse, not this mode).
    * ``probe_craters`` — the ground-truth quality read (**kNN**, primary) falls: ``healthy_probe - shortcut_probe >=
      min_probe_gap``.
    * ``loss_does_not_alarm`` — the anti-instrument: the shortcut recon-loss floor does not exceed
      the healthy arm's by more than ``loss_alarm_margin`` (the loss lies — read within-arm for the
      mask-ratio lever, per the P0.5.0 caveat).
    * ``contrast`` — the crater rests on the lever's difference: ``healthy_probe / shortcut_probe >=
      min_probe_ratio``.

    Args:
        config: The composed config (its ``gate`` block holds the thresholds).
        shortcut: The shortcut arm's health records.
        healthy: The healthy arm's health records.
        shortcut_loss_floor: Minimum reconstruction loss the shortcut arm reached.
        healthy_loss_floor: Minimum reconstruction loss the healthy arm reached.
        well_baseline: The pristine pre-continue well health read (pretrained vehicle only; ``None``
            for from_scratch). Feeds the reported warm-up-recovery diagnostic (P0.3.9).

    Returns:
        A dict of the measured numbers, per-condition booleans, the reported reader verdict, and
        the overall ``passed`` (mode induced).
    """
    g = config.gate
    sc_rankme, hc_rankme = shortcut[-1]["rankme"], healthy[-1]["rankme"]
    # kNN is the primary ground-truth probe: it registers local class structure at a lower
    # quality threshold than the linear probe, which floors at chance on a weak rep (P0.5.0).
    # Linear is kept as a reported secondary; a kNN/linear divergence is itself diagnostic.
    sc_probe, hc_probe = shortcut[-1]["knn_acc"], healthy[-1]["knn_acc"]
    sc_linear, hc_linear = shortcut[-1]["linear_acc"], healthy[-1]["linear_acc"]
    linear_probe_gap = hc_linear - sc_linear
    rankme_frac = sc_rankme / hc_rankme if hc_rankme else 0.0
    probe_gap = hc_probe - sc_probe
    probe_ratio = hc_probe / sc_probe if sc_probe else float("inf")

    # Warm-up-confound diagnostic (P0.3.9), REPORTED not gated. For the pretrained vehicle the
    # continue splices a fresh decoder, so the encoder takes a warm-up dip at the start. The healthy
    # arm should RECOVER to the pristine well's probe by the tail; if it does NOT, a low healthy_final
    # is warm-up damage (or an incompetent well), not real headroom — read the crater with that caveat.
    well_probe = float(well_baseline["knn_acc"]) if well_baseline else None
    healthy_drop_from_well = (well_probe - hc_probe) if well_probe is not None else None
    healthy_recovers_baseline = well_probe is None or hc_probe >= well_probe - g.get("recovery_margin", 0.05)

    checks = {
        "healthy_has_headroom": hc_probe >= g.healthy_probe_min,
        "rankme_preserved": rankme_frac >= g.precondition_rankme_frac,
        "probe_craters": probe_gap >= g.min_probe_gap,
        "loss_does_not_alarm": shortcut_loss_floor <= healthy_loss_floor + g.loss_alarm_margin,
        "contrast": probe_ratio >= g.min_probe_ratio,
    }
    reader_min_gaps = OmegaConf.to_container(g.reader_min_gaps, resolve=True) if g.get("reader_min_gaps") else None
    return {
        "mode": "shortcut",
        "shortcut_rankme_final": sc_rankme,
        "healthy_rankme_final": hc_rankme,
        "rankme_frac": rankme_frac,
        "shortcut_probe_final": sc_probe,
        "healthy_probe_final": hc_probe,
        "well_baseline_probe": well_probe,
        "healthy_drop_from_well": healthy_drop_from_well,
        "healthy_recovers_baseline": healthy_recovers_baseline,
        "probe_gap": probe_gap,
        "probe_ratio": probe_ratio,
        "shortcut_linear_final": sc_linear,
        "healthy_linear_final": hc_linear,
        "linear_probe_gap": linear_probe_gap,
        "shortcut_loss_floor": shortcut_loss_floor,
        "healthy_loss_floor": healthy_loss_floor,
        "thresholds": {
            "precondition_rankme_frac": g.precondition_rankme_frac,
            "healthy_probe_min": g.healthy_probe_min,
            "min_probe_gap": g.min_probe_gap,
            "min_probe_ratio": g.min_probe_ratio,
            "loss_alarm_margin": g.loss_alarm_margin,
        },
        "checks": checks,
        "reader_verdict": _reader_verdict(shortcut, healthy, g.reader_min_gap, reader_min_gaps),
        "passed": all(checks.values()),
    }


def _render_shortcut_summary(gate: dict[str, Any]) -> str:
    """Render the P0.5 shortcut gate result — mode-induction verdict + the reported reader verdict."""
    t, c = gate["thresholds"], gate["checks"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    lines = [
        f"POSITIVE-CONTROL GATE [shortcut degradation]: {verdict}  "
        f"(mode induced?; reader verdict is REPORTED, not gated)",
        f"  healthy probe (kNN, primary) = {gate['healthy_probe_final']:.3f} "
        f"(>= {t['healthy_probe_min']} real headroom?  {c['healthy_has_headroom']})",
        *(
            [
                f"  [warm-up guard, P0.3.9] pristine well kNN = {gate['well_baseline_probe']:.3f}; "
                f"healthy drop-from-well = {gate['healthy_drop_from_well']:+.3f}  "
                f"(healthy RECOVERS to well?  {gate['healthy_recovers_baseline']}"
                + (
                    ""
                    if gate["healthy_recovers_baseline"]
                    else " -- LOW healthy_final may be warm-up damage, NOT real headroom"
                )
                + ")"
            ]
            if gate.get("well_baseline_probe") is not None
            else []
        ),
        f"  PRECONDITION RankMe: shortcut {gate['shortcut_rankme_final']:.3f} / healthy "
        f"{gate['healthy_rankme_final']:.3f} = {gate['rankme_frac']:.2f} "
        f"(>= {t['precondition_rankme_frac']} -> full-rank, NOT collapse?  {c['rankme_preserved']})",
        f"  probe crater (kNN) = {gate['probe_gap']:.3f} (healthy {gate['healthy_probe_final']:.3f} - "
        f"shortcut {gate['shortcut_probe_final']:.3f}) (>= {t['min_probe_gap']}?  {c['probe_craters']})",
        f"  contrast (healthy / shortcut probe) = {gate['probe_ratio']:.2f}x "
        f"(>= {t['min_probe_ratio']}x?  {c['contrast']})",
        f"  loss does NOT alarm: shortcut floor {gate['shortcut_loss_floor']:.4f} <= healthy "
        f"{gate['healthy_loss_floor']:.4f} + {t['loss_alarm_margin']} (the loss lies?  {c['loss_does_not_alarm']})",
        f"  [reported] linear probe (secondary): shortcut {gate['shortcut_linear_final']:.3f} "
        f"vs healthy {gate['healthy_linear_final']:.3f} (gap {gate['linear_probe_gap']:+.3f}; "
        f"kNN/linear divergence is diagnostic)",
        f"  READER VERDICT (the deliverable): {gate['reader_verdict']['headline']}",
    ]
    for r in gate["reader_verdict"]["candidates"]:
        fired = "FIRES" if r["fires"] else "quiet"
        lines.append(
            f"    {r['reader']:>11}: healthy {r['healthy_final']:.4f} -> shortcut {r['shortcut_final']:.4f} "
            f"(degrade gap {r['degrade_gap']:+.4f}) [{fired}]"
        )
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control_shortcut")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run both arms, render the contrast + the P0.2 suite envelope, and apply the mode-induction gate."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    hc, hc_loss_floor, hc_baseline = _run_arm(config, arm_name="healthy", run_name="mae_healthy", out_dir=out_dir)
    sc, sc_loss_floor, sc_baseline = _run_arm(config, arm_name="shortcut", run_name="mae_shortcut", out_dir=out_dir)

    # The full P0.2 collapse suite on trial: per-(instrument × surface) separation with the SHORTCUT
    # arm in the `pc` slot. Reported as the calibration map; the quality-reader verdict is read
    # separately (metric_envelope orients alignment for *collapse*, not for this mode).
    envelope = metric_envelope(sc, hc)

    gate = _evaluate_shortcut_gate(config, sc, hc, sc_loss_floor, hc_loss_floor, well_baseline=hc_baseline)
    logger.info(_render_shortcut_summary(gate))
    logger.info(
        "P0.2 suite on trial (separation + fire/quiet per instrument × surface; shortcut in the pc slot):\n"
        + render_envelope_table(envelope)
    )

    (out_dir / "comparison.json").write_text(
        dumps_valid(
            {
                "gate": gate,
                "envelope": envelope,
                "shortcut": sc,
                "healthy": hc,
                # Pristine pre-continue well baselines (pretrained vehicle only; None for from_scratch).
                # The two should be ~identical — a sanity check that both arms loaded the same well.
                "well_baseline_healthy": hc_baseline,
                "well_baseline_shortcut": sc_baseline,
            }
        ),
        encoding="utf-8",
    )
    logger.info(f"wrote comparison + gate + envelope to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass: the shortcut mode was not cleanly induced (no full-rank probe crater "
            "with a quiet loss). Before reading the reader verdict, fix the vehicle — escalate scale "
            "for healthy headroom (P0.3.0), or push the lever harder — a null reader is only meaningful "
            "once the mode is genuinely induced."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
