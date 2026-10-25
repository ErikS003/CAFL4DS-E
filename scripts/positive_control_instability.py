"""Phase-0 positive control (P0.4.0) — the INSTABILITY / DIVERGENCE-instrument gate.

The collapse gate (``positive_control.py``, P0.2.x) and the forgetting gate
(``positive_control_forgetting.py``, P0.3.x) say nothing about whether the *instability*
detector — the **gradient norm** — works. This harness is their divergence analogue: it
drives a deliberately pathological learning rate until the loss runs to non-finite and checks
that the grad norm **fires** (blows up, ideally *ahead* of the loss NaN), while a matched
sane-LR baseline stays **quiet** (bounded, ~stationary) *and* genuinely learns — the same
two-sided discipline, a third failure mode.

Vehicle (P0.4.0): the **same MAE** over the **same IID STL-10 stream** from a seed-reset,
bit-identical start, with **warmup OFF and grad-clip OFF** on every arm so the learning rate is
the sole difference and clipping (the standard mitigation) cannot mask the blow-up. MAE is the
vehicle because divergence needs an **unbounded** loss — its per-patch MSE can genuinely blow up
to inf/NaN, whereas SimSiam's bounded negative-cosine loss *collapses* instead. Divergence is an
**optimization** phenomenon (a numerical positive-feedback runaway), so it reproduces faithfully
at toy scale — unlike forgetting, this study is seconds-to-minutes on CPU.

The harness runs one **healthy** arm at ``lr_healthy`` (the invariant quiet reference) and one
**PC** arm per LR in ``lr_sweep`` (the dose-response). The instrument is read from the per-step
``{step, loss, grad_norm, finite}`` trace the instrumented :class:`~cafl4ds.loop.StreamingLoop`
now logs. The gate (see :func:`_evaluate_instability_gate`) is the contrast between the healthy
and the highest-LR PC traces, plus the **lead time** between the grad-norm spike and the loss
going non-finite — the leading-indicator payoff that would license grad norm as a live signal.

Examples:
    Default (STL-10, CPU — the whole dose-response in one session)::

        uv run python scripts/positive_control_instability.py

    A finer / higher sweep, or a single operating point::

        uv run python scripts/positive_control_instability.py 'lr_sweep=[1e-3,2e-3,4e-3,8e-3]'
        uv run python scripts/positive_control_instability.py 'lr_sweep=[3e-3]'

    Batch-size ablation (raises gradient *variance*, a route to divergence other than LR magnitude)::

        uv run python scripts/positive_control_instability.py batch_size=16
"""

import math
import statistics
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.jsonio import dumps_valid
from cafl4ds.loop import StreamingLoop
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import apply_encoder_init

logger.remove()
logger.add(sys.stdout, level="INFO")

_SPARK = "▁▂▃▄▅▆▇█"


def _run_arm(config: DictConfig, *, lr: float, run_name: str, out_dir: Path) -> list[dict[str, Any]]:
    """Build and run one arm (a single learning rate), returning its per-step trace.

    The global seed is reset here so every arm starts from a bit-identical from-scratch MAE and
    draws the same augmentation sequence — the learning rate is the only variable. Warmup and
    grad-clip are off (from the config), so the arm is the raw optimization dynamics at ``lr``.

    Args:
        config: The composed ``positive_control_instability`` config.
        lr: The learning rate for this arm.
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        The per-step loss series (each record carries ``loss``, ``grad_norm``, ``finite``).
    """
    torch.manual_seed(config.seed)  # identical init + augmentation RNG across arms
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)
    apply_encoder_init(method.encoder, "from_scratch")  # both arms from-scratch, bit-identical

    stream = instantiate(config.stream)  # same seed -> identical splits/order as the other arms
    optimizer = instantiate(config.optim, params=method.parameters(), lr=lr)
    monitor = instantiate(config.monitor, eval_sets=stream.eval_sets)

    batches_per_epoch = len(stream)
    eval_every = max(1, config.eval_every_epochs * batches_per_epoch)
    total_steps = config.epochs * batches_per_epoch
    scheduler = instantiate(config.schedule, optimizer=optimizer, total_steps=total_steps)

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"arm '{run_name}' (lr={lr:g}): {batches_per_epoch} batches x {config.epochs} epochs = {total_steps}")

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
        grad_clip=config.grad_clip,
        device=config.device,
    )
    loop.run()
    return [r for r in read_run(run_log_path) if r.get("series") == "loss"]


def _summarize(trace: list[dict[str, Any]], init_skip: int) -> dict[str, Any]:
    """Reduce a per-step trace to the divergence summary statistics.

    Grad-norm stats are read over the **steady state** — the finite steps at/after ``init_skip`` —
    so the no-warmup first-gradient transient (present in every arm) never counts as a blow-up. An
    arm that goes non-finite before ``init_skip`` has no steady window, so it falls back to all its
    finite steps (it diverges via the non-finite event regardless).

    Args:
        trace: The arm's per-step loss records (``step``, ``loss``, ``grad_norm``, ``finite``).
        init_skip: Number of leading steps to exclude from the steady-state grad-norm stats.

    Returns:
        A dict with the first non-finite step (or ``None``), the steady-state peak / median grad
        norm, and the first / final finite loss (the descent read).
    """
    steps = [int(r["step"]) for r in trace]
    losses = [r["loss"] for r in trace]
    gnorms = [r["grad_norm"] for r in trace]
    finite = [bool(r.get("finite", True)) for r in trace]

    first_nonfinite = next((s for s, f in zip(steps, finite, strict=False) if not f), None)
    finite_gnorms = [g for g, f in zip(gnorms, finite, strict=False) if f]
    steady = [g for s, g, f in zip(steps, gnorms, finite, strict=False) if f and s >= init_skip]
    band = steady or finite_gnorms  # fall back to all finite steps if divergence pre-empted the window
    finite_losses = [x for x, f in zip(losses, finite, strict=False) if f]
    return {
        "n_steps": len(trace),
        "first_nonfinite_step": first_nonfinite,
        "peak_grad_norm": max(band) if band else float("inf"),
        "median_grad_norm": statistics.median(band) if band else float("nan"),
        "init_loss": finite_losses[0] if finite_losses else float("nan"),
        "final_loss": finite_losses[-1] if finite_losses else float("nan"),
    }


def _lead_time(trace: list[dict[str, Any]], threshold: float, init_skip: int) -> dict[str, Any]:
    """Lead time between the grad-norm spike and the loss going non-finite (the payoff read).

    The spike search starts at ``init_skip`` so the no-warmup first-gradient transient (which is
    the same magnitude as a runaway's early steps) is never mistaken for the divergence spike.

    Args:
        trace: The (PC) arm's per-step trace.
        threshold: The grad-norm blow-up level (``divergence_ratio`` x the healthy steady peak)
            whose *first* crossing at/after ``init_skip`` marks the spike.
        init_skip: Leading steps excluded from the spike search (the init transient).

    Returns:
        ``{spike_step, nonfinite_step, lead}`` where ``lead = nonfinite_step - spike_step`` (steps
        of warning the grad norm gave). ``lead`` is ``None`` if either event never occurred; a
        positive value is the win (grad norm crossed *before* the loss NaN).
    """
    steps = [int(r["step"]) for r in trace if int(r["step"]) >= init_skip]
    gnorms = [r["grad_norm"] for r in trace if int(r["step"]) >= init_skip]
    finite = [bool(r.get("finite", True)) for r in trace if int(r["step"]) >= init_skip]
    # First step whose grad norm crosses the threshold OR which is itself non-finite (a blown grad).
    spike = next(
        (s for s, g, f in zip(steps, gnorms, finite, strict=False) if (not f) or g >= threshold),
        None,
    )
    nonfinite = next((s for s, f in zip(steps, finite, strict=False) if not f), None)
    lead = (nonfinite - spike) if (nonfinite is not None and spike is not None) else None
    return {"spike_step": spike, "nonfinite_step": nonfinite, "lead": lead}


def _logspark(values: list[float], lo: float, hi: float) -> str:
    """Render values as a unicode sparkline on a **log10** scale over ``[lo, hi]`` (log-space bounds).

    Grad norms span orders of magnitude across a dose-response, so a linear scale hides the runaway
    behind its own peak; the log scale keeps the healthy band and the blow-up both legible. A
    non-finite value (the divergence event) renders as a full block.

    Args:
        values: The series to render (raw grad norms).
        lo: Lower bound in log10-space (maps to the lowest block).
        hi: Upper bound in log10-space (maps to the highest block).

    Returns:
        A one-line sparkline string.
    """
    span = hi - lo or 1.0
    cells = []
    for v in values:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf -> full block (the blow-up)
            cells.append(_SPARK[-1])
            continue
        lv = math.log10(max(v, 1e-12))
        cells.append(_SPARK[min(len(_SPARK) - 1, max(0, int((lv - lo) / span * (len(_SPARK) - 1))))])
    return "".join(cells)


def _evaluate_instability_gate(
    config: DictConfig, healthy: dict[str, Any], sweep: list[dict[str, Any]]
) -> dict[str, Any]:
    """Two-sided divergence gate: the top-LR PC must diverge, the healthy arm must hold & learn.

    Passes iff: the healthy arm is **stable** (never goes non-finite); the healthy arm genuinely
    **learns** (final loss ≤ ``gate.healthy_descends_frac`` x its initial loss — the quiet baseline
    is not a degenerate artifact); the **highest-LR PC arm diverges** (its loss/grad goes non-finite
    by the horizon — the unambiguous divergence fingerprint); and the grad norm **separates** —
    the smallest diverging arm's peak grad norm is ≥ ``gate.min_separation`` x the healthy steady
    peak (the two-sided contrast; the healthy grad norm is *bounded*, not stationary, so the gate
    reads this orders-of-magnitude gap, not a peak/median stationarity test).

    Reported but not gated: the full LR **dose-response** (divergence onset per LR — the threshold,
    and that the runaway speeds up with LR) and the **lead time** at the just-past-threshold
    operating point (grad-norm blow-up vs. loss NaN).

    Args:
        config: The composed config (``gate`` block holds the thresholds).
        healthy: The healthy arm's summary.
        sweep: The PC arms' summaries, in ascending-LR order (each carries its ``lr``).

    Returns:
        A dict of the measured numbers, per-condition ``checks``, the ``dose_response`` +
        ``lead_time`` observations, and the overall ``passed`` (the three hard checks).
    """
    g = config.gate
    h_med = healthy["median_grad_norm"]
    h_peak = healthy["peak_grad_norm"]
    # The grad-norm "blow-up" level, for the lead-time spike search. Anchored to the healthy STEADY
    # peak (not the median) so a runaway's early steps clear it but the healthy band never does.
    # Divergence ITSELF is the non-finite event — an unambiguous fingerprint independent of a ratio.
    blowup_level = g.divergence_ratio * h_peak

    def _fires(arm: dict[str, Any]) -> bool:
        return arm["first_nonfinite_step"] is not None

    dose = [
        {
            "lr": a["lr"],
            "peak_grad_norm": a["peak_grad_norm"],
            "nonfinite": a["first_nonfinite_step"],
            "fires": _fires(a),
        }
        for a in sweep
    ]
    firing = [d for d in dose if d["fires"]]
    threshold_lr = (
        firing[0]["lr"] if firing else None
    )  # smallest LR that diverges (the just-past-threshold operating point)
    top_pc = sweep[-1]

    # The two-sided CONTRAST (the headline, analogue of the collapse RankMe ratio / forgetting
    # forget ratio): the smallest diverging arm's peak grad norm vs. the healthy steady peak. A
    # healthy grad norm is BOUNDED, not stationary — it fluctuates with batch noise — so the gate
    # reads this orders-of-magnitude separation, not a fragile peak/median stationarity test.
    diverging_peaks = [d["peak_grad_norm"] for d in dose if d["fires"]]
    min_diverging_peak = min(diverging_peaks) if diverging_peaks else 0.0
    separation = (min_diverging_peak / h_peak) if h_peak else float("inf")

    healthy_stable = healthy["first_nonfinite_step"] is None  # the healthy arm never diverges
    healthy_learns = healthy["final_loss"] <= g.healthy_descends_frac * healthy["init_loss"]

    checks = {
        "healthy_stable": bool(healthy_stable),
        "healthy_learns": bool(healthy_learns),
        "pc_diverges": bool(_fires(top_pc)),
        "grad_norm_separates": separation >= g.min_separation,
    }
    return {
        "mode": "instability",
        "healthy_median_grad_norm": h_med,
        "healthy_peak_grad_norm": h_peak,
        "healthy_init_loss": healthy["init_loss"],
        "healthy_final_loss": healthy["final_loss"],
        "blowup_level": blowup_level,
        "min_diverging_peak_grad_norm": min_diverging_peak,
        "grad_norm_separation": separation,
        "top_pc_lr": top_pc["lr"],
        "top_pc_peak_grad_norm": top_pc["peak_grad_norm"],
        "top_pc_nonfinite_step": top_pc["first_nonfinite_step"],
        "threshold_lr": threshold_lr,
        "dose_response": dose,
        "thresholds": OmegaConf.to_container(g),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _render_summary(gate: dict[str, Any], lead: dict[str, Any]) -> str:
    """Render the instability-gate verdict block (two-sided gate + dose-response + lead time)."""
    c, t = gate["checks"], gate["thresholds"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"

    def _dose_cell(d: dict[str, Any]) -> str:
        # lr:<peak grad norm, or NaN@step if it diverged><🔥 fires | · quiet>
        reading = f"NaN@{d['nonfinite']}" if d["nonfinite"] is not None else f"{d['peak_grad_norm']:.2g}"
        return f"{d['lr']:g}:{reading}{'🔥' if d['fires'] else '·'}"

    dose = "  ".join(_dose_cell(d) for d in gate["dose_response"])
    threshold_lr = gate["threshold_lr"]
    lead_str = (
        f"spike@{lead['spike_step']} -> NaN@{lead['nonfinite_step']} = {lead['lead']} steps lead"
        if lead["lead"] is not None
        else "no gradual runaway captured at the threshold LR (NaN in one step, or none fired)"
    )
    return (
        f"POSITIVE-CONTROL GATE [instability / divergence]: {verdict}\n"
        f"  healthy grad norm  steady median={gate['healthy_median_grad_norm']:.3g} "
        f"peak={gate['healthy_peak_grad_norm']:.3g}; stayed finite -> STABLE?  {c['healthy_stable']}\n"
        f"  healthy loss  {gate['healthy_init_loss']:.4f} -> {gate['healthy_final_loss']:.4f} "
        f"(<= {t['healthy_descends_frac']}x init -> genuinely LEARNS?  {c['healthy_learns']})\n"
        f"  top-LR PC (lr={gate['top_pc_lr']:g})  peak grad norm={gate['top_pc_peak_grad_norm']:.3g} "
        f"nonfinite@{gate['top_pc_nonfinite_step']} (loss/grad -> non-finite -> DIVERGES?  {c['pc_diverges']})\n"
        f"  grad-norm separation (min diverging peak {gate['min_diverging_peak_grad_norm']:.3g} / healthy peak) "
        f"= {gate['grad_norm_separation']:.3g}x (>= {t['min_separation']}x -> SEPARATES?  {c['grad_norm_separates']})\n"
        f"  dose-response (lr:peak/NaN-step, 🔥=diverges): {dose}\n"
        f"  threshold LR (smallest diverging) = {threshold_lr if threshold_lr is not None else 'none'}; "
        f"lead time @ threshold: {lead_str}"
    )


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control_instability")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run the healthy arm + the PC LR sweep, render the traces, and apply the divergence gate."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    init_skip = int(config.init_skip)
    healthy_trace = _run_arm(config, lr=config.lr_healthy, run_name="mae_healthy", out_dir=out_dir)
    healthy = _summarize(healthy_trace, init_skip)

    sweep_traces: dict[str, list[dict[str, Any]]] = {}
    sweep: list[dict[str, Any]] = []
    for lr in config.lr_sweep:
        name = f"mae_pc_lr{float(lr):g}"
        trace = _run_arm(config, lr=float(lr), run_name=name, out_dir=out_dir)
        summary = _summarize(trace, init_skip)
        summary["lr"] = float(lr)
        sweep_traces[name] = trace
        sweep.append(summary)

    gate = _evaluate_instability_gate(config, healthy, sweep)

    # Lead time at the just-past-threshold operating point (the gradual runaway), if any diverged.
    threshold_lr = gate["threshold_lr"]
    blowup_level = gate["blowup_level"]
    lead = {"spike_step": None, "nonfinite_step": None, "lead": None}
    if threshold_lr is not None:
        op_name = f"mae_pc_lr{threshold_lr:g}"
        lead = _lead_time(sweep_traces[op_name], blowup_level, init_skip)

    # Side-by-side grad-norm sparklines on a shared log10 scale (norms span orders of magnitude).
    all_g = [r["grad_norm"] for r in healthy_trace if r.get("finite", True)]
    for tr in sweep_traces.values():
        all_g += [r["grad_norm"] for r in tr if r.get("finite", True)]
    lo, hi = (math.log10(max(min(all_g), 1e-12)), math.log10(max(all_g))) if all_g else (0.0, 1.0)
    curves = [
        f"grad-norm curves (shared log10 scale 1e{lo:.1f}..1e{hi:.1f}, █=non-finite):",
        f"  healthy (lr={config.lr_healthy:g}) : {_logspark([r['grad_norm'] for r in healthy_trace], lo, hi)}",
    ]
    for name, tr in sweep_traces.items():
        curves.append(f"  {name:<18}: {_logspark([r['grad_norm'] for r in tr], lo, hi)}")

    logger.info("\n".join(curves))
    logger.info(_render_summary(gate, lead))

    comparison = {
        "gate": gate,
        "lead_time": lead,
        "healthy": {"summary": healthy, "trace": healthy_trace},
        "sweep": [{"summary": s, "trace": sweep_traces[f"mae_pc_lr{s['lr']:g}"]} for s in sweep],
    }
    (out_dir / "comparison.json").write_text(dumps_valid(comparison), encoding="utf-8")
    logger.info(f"wrote comparison + gate to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass. Either the healthy arm did not stay quiet / genuinely learn, or the "
            "top-LR PC did not diverge — investigate the regime before trusting the grad-norm "
            "instability instrument downstream."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
