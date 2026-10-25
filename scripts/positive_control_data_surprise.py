"""Phase-0 positive control (P0.4.1) — the DATA-driven SURPRISE / SHOCK instability read.

The deployment-relevant companion to :mod:`scripts.positive_control_instability` (P0.4.0). Where
P0.4.0 forces divergence with a pathological learning rate — the clean, controllable lever —
P0.4.1 asks the question that matters for cafl4ds's stream regime: does the **gradient-norm**
instrument fire on **data surprise** (an out-of-distribution burst arriving mid-stream), and can it
tell a *transient, recoverable* shock from a *runaway* one?

Vehicle: the **same instrumented** :class:`~cafl4ds.loop.StreamingLoop` and the **same MAE + SGD**
as P0.4.0 — but with **raw (un-normalized) reconstruction MSE** (``norm_pix_loss=false``), because
surprise must register in the loss *magnitude* and ``norm_pix`` caps the per-patch loss near 1.0 for
any content (a P0.4.1 vehicle finding). Over a hand-built stream: ``warm_steps`` of a **base**
distribution (STL-10) so the MAE is genuinely competent, then a ``burst_steps`` shock, then
``tail_steps`` of the base again so a transient spike's *recovery* is observable. Two arms from a
seed-reset, bit-identical start differing ONLY by whether the burst is injected:

* **shock** — base -> burst -> base: the grad norm **spikes** at the burst (a change point), then
  either **recovers** to the pre-burst band (transient) or **runs away** (divergence).
* **control** — the base throughout (no burst): the grad norm stays **in band** (quiet reference).

The gate (see :func:`_evaluate_surprise_gate`) is the **two-sided (shock vs. control at the same
steps) change-point contrast** — the shock's ``burst peak / control same-step peak`` grad norm
firing while the control's within-arm band stays flat — plus a base-competence precondition (the
warm loss must descend) and that the burst is genuinely surprising (a higher raw loss than the
control's same steps).

Key finding: the grad norm reads reconstruction-error **magnitude**, i.e. a **content** shock, NOT
distributional novelty. The positive control that fires is a ``noise`` burst scaled **out of the
trained ``[0, 1]`` range** (``burst_scale`` > 1, a saturating sensor-corruption shock). An in-range
distributional-OOD burst (``burst_kind=dataset``: CIFAR / grayscale / phase-scrambled) does **not**
fire — its raw reconstruction error is comparable to the base, below the grad-norm variance floor —
so grad norm is not a distribution-shift detector for MAE. The ``burst_scale`` amplitude is a clean
data-shock **severity dose-response** at fixed ``lr_op``: below threshold the spike recovers
(transient — severity ≠ permanence), above it the shock *triggers* a divergence (the burst completes
its runaway in the base tail), with the lead time **shrinking** as severity grows — the data-driven
analogue of P0.4.0's lead-time / operating-envelope finding. The ``lr_sweep`` ladder is the negative
control: LR-driven divergence fires during *warm* (shock ≡ control NaN step), so it is
shock-independent.

Examples:
    Default (STL-10 base, saturating-noise shock, CPU — the whole read in one session)::

        uv run python scripts/positive_control_data_surprise.py

    The data-shock severity dose-response (no-fire -> transient -> runaway)::

        uv run python scripts/positive_control_data_surprise.py 'burst_scale=1'  # in-range: no fire
        uv run python scripts/positive_control_data_surprise.py 'burst_scale=8'  # over threshold: runaway

    The distributional-OOD non-fire ladder (the refuted naive vehicle)::

        uv run python scripts/positive_control_data_surprise.py burst_kind=dataset data_b=cifar100_scrambled
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

from cafl4ds.data.streams import EvalSet, EvalSets, StreamBatch
from cafl4ds.jsonio import dumps_valid
from cafl4ds.loop import _MIN_BATCH, StreamingLoop
from cafl4ds.run_log import RunLogger, read_run

logger.remove()
logger.add(sys.stdout, level="INFO")

_SPARK = "▁▂▃▄▅▆▇█"


def _iid_batches(images: torch.Tensor, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Shuffle ``images`` once and chunk into fixed-size batches (sub-minimum tail dropped).

    Args:
        images: All source images ``[N, C, H, W]`` in ``[0, 1]``.
        batch_size: Images per batch.
        generator: RNG for the (single) shuffle — fixed so both arms see the same base order.

    Returns:
        The list of image batches (each ``>= _MIN_BATCH`` samples).
    """
    perm = torch.randperm(images.shape[0], generator=generator)
    shuffled = images[perm]
    batches = [shuffled[i : i + batch_size] for i in range(0, shuffled.shape[0], batch_size)]
    return [b for b in batches if b.shape[0] >= _MIN_BATCH]


def _burst_batches(config: DictConfig, base_imgs: torch.Tensor, generator: torch.Generator) -> list[torch.Tensor]:
    """Build the OOD burst batches per ``config.burst_kind``.

    ``noise`` synthesizes uniform-random ``[0, 1]`` images of the base shape — content that is
    genuinely hard to *inpaint*, the guaranteed positive control for a content shock. ``dataset``
    loads ``config.data_b`` (the CIFAR distributional-OOD variants), the naive-surprise vehicle the
    study refutes. Both are chunked to ``batch_size``.

    Args:
        config: The composed config (``burst_kind``, ``data_b``, ``batch_size``, ``seed``).
        base_imgs: The base images (for the noise burst's ``[C, H, W]`` shape).
        generator: RNG for the noise draw / OOD shuffle (fixed, shared across arms).

    Returns:
        The list of burst image batches.

    Raises:
        ValueError: If ``burst_kind`` is neither ``"noise"`` nor ``"dataset"``.
    """
    kind = config.burst_kind
    if kind == "noise":
        _, c, h, w = base_imgs.shape
        n = 32  # a pool of distinct noise batches the burst cycles through
        scale = float(config.get("burst_scale", 1.0))  # amplitude: > 1 saturates out of the trained [0,1] range
        return [scale * torch.rand(config.batch_size, c, h, w, generator=generator) for _ in range(n)]
    if kind == "dataset":
        ood_imgs, _ = instantiate(config.data_b).load()
        return _iid_batches(ood_imgs, config.batch_size, generator)
    raise ValueError(f"unknown burst_kind {kind!r}; expected 'noise' or 'dataset'.")


def _shock_sequence(
    base: list[torch.Tensor], ood: list[torch.Tensor], *, warm: int, burst: int, tail: int, inject: bool
) -> list[StreamBatch]:
    """Build the base -> (OOD) burst -> base step sequence as label-free :class:`StreamBatch` es.

    Base batches are indexed by ``step`` (cycled with a modulo over the available base batches), so
    the warm and tail base steps are **identical across arms** regardless of ``inject`` — the only
    difference the toggle makes is the *content* of the burst window (OOD when injected, base
    otherwise). Eras tag the phase: 0 = warm base, 1 = OOD burst (injected arm only), 2 = tail base.

    Args:
        base: The base-distribution image batches (cycled).
        ood: The OOD image batches (used only in the burst window when ``inject``).
        warm: Number of warm (base) steps before the burst.
        burst: Number of burst steps.
        tail: Number of tail (base) steps after the burst.
        inject: Whether to inject the OOD burst (``True`` = shock arm, ``False`` = control arm).

    Returns:
        The ordered stream of ``warm + burst + tail`` batches.
    """
    seq: list[StreamBatch] = []
    for step in range(warm + burst + tail):
        if inject and warm <= step < warm + burst:
            seq.append(StreamBatch(images=ood[(step - warm) % len(ood)], era=1, step=step))
        else:
            era = 2 if step >= warm + burst else 0
            seq.append(StreamBatch(images=base[step % len(base)], era=era, step=step))
    return seq


def _run_arm(
    config: DictConfig,
    *,
    lr: float,
    inject: bool,
    base: list[torch.Tensor],
    ood: list[torch.Tensor],
    eval_sets: EvalSets,
    run_name: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Build and run one arm (a fixed LR, burst injected or not), returning its per-step trace.

    The global seed is reset here so every arm starts from a bit-identical from-scratch MAE and
    draws the same masking sequence over the shared base batches — the burst (and the LR) are the
    only variables. Flat LR (no scheduler) and grad-clip off, so the burst is the sole perturbation
    and a blow-up is not masked.

    Args:
        config: The composed ``positive_control_data_surprise`` config.
        lr: The (flat) learning rate for this arm.
        inject: Whether to inject the OOD burst (shock vs. control).
        base: The base-distribution image batches.
        ood: The OOD image batches.
        eval_sets: Held-out base eval sets for the (cheap, probe-off) health monitor.
        run_name: Name recorded on the run log and used for its filename.
        out_dir: Directory the run log is written to.

    Returns:
        The per-step loss series (each record carries ``loss``, ``grad_norm``, ``finite``, ``era``).
    """
    torch.manual_seed(config.seed)  # identical init + masking RNG across arms
    encoder = instantiate(config.encoder)
    method = instantiate(config.ssl, encoder=encoder)
    method.encoder.reset_parameters_from_scratch()  # both arms from-scratch, bit-identical
    method.to(torch.device(config.device))

    optimizer = instantiate(config.optim, params=method.parameters(), lr=lr)
    monitor = instantiate(config.monitor, eval_sets=eval_sets)
    seq = _shock_sequence(
        base, ood, warm=config.warm_steps, burst=config.burst_steps, tail=config.tail_steps, inject=inject
    )

    run_log_path = out_dir / f"{run_name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=run_name)
    logger.info(f"arm '{run_name}' (lr={lr:g}, inject={inject}): {len(seq)} steps")

    StreamingLoop(
        stream=seq,
        method=method,
        optimizer=optimizer,
        selection_filter=instantiate(config.filter),
        monitor=monitor,
        run_logger=run_logger,
        eval_every=len(seq) + 1,  # health only at the (cheap) closing read; grad norm + loss are the readouts
        epochs=1,
        scheduler=None,  # flat LR: the burst is the sole perturbation
        grad_clip=config.grad_clip,
        device=config.device,
    ).run()
    return [r for r in read_run(run_log_path) if r.get("series") == "loss"]


def _window(trace: list[dict[str, Any]], lo: int, hi: int) -> list[dict[str, Any]]:
    """Return the trace records whose step is in ``[lo, hi)``."""
    return [r for r in trace if lo <= int(r["step"]) < hi]


def _peak(records: list[dict[str, Any]]) -> float:
    """Peak finite grad norm over ``records`` (``0.0`` if none finite)."""
    finite = [r["grad_norm"] for r in records if r.get("finite", True)]
    return max(finite) if finite else 0.0


def _mean_loss(records: list[dict[str, Any]]) -> float:
    """Mean finite loss over ``records`` (``nan`` if none finite)."""
    finite = [r["loss"] for r in records if r.get("finite", True)]
    return sum(finite) / len(finite) if finite else float("nan")


def _summarize(config: DictConfig, trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce an arm's per-step trace to the change-point summary over its three windows.

    Windows: **pre-burst base** ``[init_skip, warm)`` (the quiet band the change point is read
    against — the init transient is excluded, as in P0.4.0), the **burst** ``[warm, warm+burst)``,
    and the **tail** ``[warm+burst, end)`` (recovery). Reads the within-arm change-point ratio
    (burst peak / base peak), whether the burst was genuinely surprising (burst vs. base loss), and
    whether the grad norm recovered to the base band in the tail.

    Args:
        config: The composed config (window sizes + ``init_skip``).
        trace: The arm's per-step loss records.

    Returns:
        A dict of the per-window peaks / losses, the change-point + recovery ratios, the base-loss
        descent, and the first non-finite step (``None`` unless the shock ran away).
    """
    warm, burst = int(config.warm_steps), int(config.burst_steps)
    init_skip = int(config.init_skip)
    base_win = _window(trace, init_skip, warm)
    burst_win = _window(trace, warm, warm + burst)
    tail_win = _window(trace, warm + burst, len(trace) + 1)

    base_peak = _peak(base_win)
    burst_peak = _peak(burst_win)
    tail_finite = [r["grad_norm"] for r in tail_win if r.get("finite", True)]
    tail_min = min(tail_finite) if tail_finite else float("inf")
    first_nonfinite = next((int(r["step"]) for r in trace if not r.get("finite", True)), None)

    base_losses = [r["loss"] for r in base_win if r.get("finite", True)]
    return {
        "n_steps": len(trace),
        "base_peak_grad_norm": base_peak,
        "base_median_grad_norm": statistics.median([r["grad_norm"] for r in base_win]) if base_win else float("nan"),
        "burst_peak_grad_norm": burst_peak,
        "tail_min_grad_norm": tail_min,
        "within_arm_ratio": (burst_peak / base_peak) if base_peak else float("inf"),
        "recover_ratio": (tail_min / base_peak) if base_peak else float("inf"),
        "base_mean_loss": _mean_loss(base_win),
        "burst_mean_loss": _mean_loss(burst_win),
        "base_init_loss": base_losses[0] if base_losses else float("nan"),
        "base_final_loss": base_losses[-1] if base_losses else float("nan"),
        "first_nonfinite_step": first_nonfinite,
    }


def _lead_time(config: DictConfig, trace: list[dict[str, Any]], blowup_level: float) -> dict[str, Any]:
    """Lead time between the grad-norm blow-up and the loss NaN, for an arm that ran away.

    The spike search starts at the burst onset (``warm_steps``) — the shock triggers the runaway, so
    the pre-burst base band is irrelevant here. Analogous to the P0.4.0 lead-time read.

    Args:
        config: The composed config (``warm_steps`` = burst onset).
        trace: The (shock) arm's per-step trace.
        blowup_level: The grad-norm level whose first post-onset crossing marks the spike.

    Returns:
        ``{spike_step, nonfinite_step, lead}`` (``lead = nonfinite - spike``); ``None`` if either
        event never occurred.
    """
    onset = int(config.warm_steps)
    post = [r for r in trace if int(r["step"]) >= onset]
    spike = next((int(r["step"]) for r in post if (not r.get("finite", True)) or r["grad_norm"] >= blowup_level), None)
    nonfinite = next((int(r["step"]) for r in post if not r.get("finite", True)), None)
    lead = (nonfinite - spike) if (nonfinite is not None and spike is not None) else None
    return {"spike_step": spike, "nonfinite_step": nonfinite, "lead": lead}


def _logspark(values: list[float], lo: float, hi: float) -> str:
    """Render values as a unicode sparkline on a **log10** scale over ``[lo, hi]`` (log-space bounds).

    A non-finite value renders as a full block (the divergence event).

    Args:
        values: The series to render (raw grad norms).
        lo: Lower bound in log10-space.
        hi: Upper bound in log10-space.

    Returns:
        A one-line sparkline string.
    """
    span = hi - lo or 1.0
    cells = []
    for v in values:
        if v != v or v in (float("inf"), float("-inf")):
            cells.append(_SPARK[-1])
            continue
        lv = math.log10(max(v, 1e-12))
        cells.append(_SPARK[min(len(_SPARK) - 1, max(0, int((lv - lo) / span * (len(_SPARK) - 1))))])
    return "".join(cells)


def _cross_ratios(shock: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    """The two-sided burst-window contrasts: shock vs. control read at the **same** steps.

    Comparing the shock's burst to the control's *same* steps (not to its own pre-burst base band)
    removes the training-progress confound — the base loss is still descending across the burst, so
    the counterfactual "no shock" reading is the control at those exact steps, not an earlier band.

    Args:
        shock: The shock arm's summary.
        control: The control arm's summary (same LR, no burst).

    Returns:
        ``surprise_ratio`` (shock/control burst mean loss) and ``fire_ratio`` (shock/control burst
        peak grad norm).
    """
    c_loss, c_peak = control["burst_mean_loss"], control["burst_peak_grad_norm"]
    return {
        "surprise_ratio": (shock["burst_mean_loss"] / c_loss) if c_loss else float("inf"),
        "fire_ratio": (shock["burst_peak_grad_norm"] / c_peak) if c_peak else float("inf"),
    }


def _evaluate_surprise_gate(config: DictConfig, shock: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Two-sided change-point gate at ``lr_op``: the shock must fire on the burst, control must hold.

    Passes iff: the base is **competent** (the shock arm's warm loss descended — an under-trained
    MAE finds everything surprising, voiding the contrast); the burst is genuinely **surprising**
    (shock burst mean loss ≥ ``surprise_ratio`` x the **control's same-step** burst-window loss); the
    grad norm **fires** (the CROSS-ARM ``fire_ratio`` = shock burst peak / control same-step burst
    peak ≥ ``gate.fire_ratio`` — the two-sided change point, the headline gate); and the **control is
    flat** (the base-only arm's own WITHIN-ARM ``within_arm_ratio`` = burst peak / pre-burst base peak
    < ``control_quiet_ratio`` and it never diverges — no spurious change without the injection). Note
    the two ratios are distinct quantities: ``fire_ratio`` is cross-arm (the gate); ``within_arm_ratio``
    is within-arm (the control's own change-point sanity check).

    Reported but not gated: whether the shock **recovers** to the base band in the tail (transient,
    the P0.3 severity ≠ permanence analogue) — the runaway pole + its lead time is read from the
    ``lr_sweep`` ladder, not here.

    Args:
        config: The composed config (``gate`` block holds the thresholds).
        shock: The shock (burst-injected) arm's summary at ``lr_op``.
        control: The control (no-burst) arm's summary at ``lr_op``.

    Returns:
        A dict of the measured numbers, per-condition ``checks``, the non-gating ``reported``
        recovery read, and the overall ``passed``.
    """
    g = config.gate
    x = _cross_ratios(shock, control)
    base_competent = shock["base_final_loss"] <= g.base_descends_frac * shock["base_init_loss"]
    surprise_real = x["surprise_ratio"] >= g.surprise_ratio
    fires = x["fire_ratio"] >= g.fire_ratio
    control_flat = control["within_arm_ratio"] < g.control_quiet_ratio and control["first_nonfinite_step"] is None

    checks = {
        "base_competent": bool(base_competent),
        "surprise_real": bool(surprise_real),
        "shock_fires": bool(fires),
        "control_flat": bool(control_flat),
    }
    reported = {
        "shock_recovers": shock["recover_ratio"] <= g.recover_ratio and shock["first_nonfinite_step"] is None,
        "shock_recover_ratio": shock["recover_ratio"],
        "shock_first_nonfinite_step": shock["first_nonfinite_step"],
    }
    return {
        "mode": "data_surprise",
        "lr_op": float(config.lr_op),
        "base_init_loss": shock["base_init_loss"],
        "base_final_loss": shock["base_final_loss"],
        "shock_burst_mean_loss": shock["burst_mean_loss"],
        "control_burst_mean_loss": control["burst_mean_loss"],
        "surprise_ratio": x["surprise_ratio"],
        "shock_burst_peak_grad_norm": shock["burst_peak_grad_norm"],
        "control_burst_peak_grad_norm": control["burst_peak_grad_norm"],
        "fire_ratio": x["fire_ratio"],
        "control_within_arm_ratio": control["within_arm_ratio"],
        "thresholds": OmegaConf.to_container(g),
        "checks": checks,
        "reported": reported,
        "passed": all(checks.values()),
    }


def _render_summary(gate: dict[str, Any], ladder: list[dict[str, Any]]) -> str:
    """Render the surprise-gate verdict block (two-sided change point + the runaway ladder)."""
    c, t, r = gate["checks"], gate["thresholds"], gate["reported"]
    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"

    def _rung(x: dict[str, Any]) -> str:
        shock_nf = x["shock"]["first_nonfinite_step"]
        ctrl_nf = x["control"]["first_nonfinite_step"]
        fire = _cross_ratios(x["shock"], x["control"])["fire_ratio"]
        shock_read = f"NaN@{shock_nf}" if shock_nf is not None else f"fire x{fire:.2g}"
        ctrl_read = f"NaN@{ctrl_nf}" if ctrl_nf is not None else "finite"
        lead = x.get("lead", {}).get("lead")
        lead_str = f" lead={lead}" if lead is not None else ""
        return (
            f"lr={x['lr']:g}: shock {shock_read}{'🔥' if shock_nf is not None else ''} vs control {ctrl_read}{lead_str}"
        )

    rungs = "\n    ".join(_rung(x) for x in ladder)
    return (
        f"POSITIVE-CONTROL GATE [data surprise / shock] @ lr_op={gate['lr_op']:g}: {verdict}\n"
        f"  base loss  {gate['base_init_loss']:.4f} -> {gate['base_final_loss']:.4f} "
        f"(<= {t['base_descends_frac']}x init -> base COMPETENT?  {c['base_competent']})\n"
        f"  surprise  shock burst loss {gate['shock_burst_mean_loss']:.4f} / control same-step "
        f"{gate['control_burst_mean_loss']:.4f} = {gate['surprise_ratio']:.3g}x "
        f"(>= {t['surprise_ratio']}x -> SURPRISE real?  {c['surprise_real']})\n"
        f"  change point  shock burst peak {gate['shock_burst_peak_grad_norm']:.3g} / control same-step "
        f"{gate['control_burst_peak_grad_norm']:.3g} = {gate['fire_ratio']:.3g}x "
        f"(>= {t['fire_ratio']}x -> FIRES?  {c['shock_fires']})\n"
        f"  control flat  within-arm burst/base = {gate['control_within_arm_ratio']:.3g}x "
        f"(< {t['control_quiet_ratio']}x & finite -> FLAT?  {c['control_flat']})\n"
        f"  [reported] shock recovers to base band in tail (recover ratio {r['shock_recover_ratio']:.3g} "
        f"<= {t['recover_ratio']}x)?  {r['shock_recovers']}  (transient shock; severity != permanence)\n"
        f"  runaway ladder (shock vs base-only control per LR; 🔥 = shock ran away):\n    {rungs}"
    )


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="positive_control_data_surprise")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Warm on the base, splice an OOD burst, and read the grad-norm change point over the LR ladder."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    gen = torch.Generator().manual_seed(config.seed)
    base_imgs, _ = instantiate(config.data).load()
    base = _iid_batches(base_imgs, config.batch_size, gen)
    ood = _burst_batches(config, base_imgs, gen)
    logger.info(f"base: {len(base)} batches (cycled); burst ({config.burst_kind}): {len(ood)} batches")
    # A small held-out base slice for the (probe-off) monitor's construction — labels unused.
    hold = base_imgs[: 2 * config.batch_size]
    zeros = torch.zeros(hold.shape[0], dtype=torch.long)
    half = config.batch_size
    eval_sets = EvalSets(
        probe_support=EvalSet(hold[:half], zeros[:half]), probe_query=EvalSet(hold[half:], zeros[half:])
    )

    ladder: list[dict[str, Any]] = []
    op: dict[str, Any] = {}
    for raw_lr in config.lr_sweep:
        lr = float(raw_lr)
        tag = f"lr{lr:g}"
        shock_trace = _run_arm(
            config,
            lr=lr,
            inject=True,
            base=base,
            ood=ood,
            eval_sets=eval_sets,
            run_name=f"shock_{tag}",
            out_dir=out_dir,
        )
        control_trace = _run_arm(
            config,
            lr=lr,
            inject=False,
            base=base,
            ood=ood,
            eval_sets=eval_sets,
            run_name=f"control_{tag}",
            out_dir=out_dir,
        )
        shock, control = _summarize(config, shock_trace), _summarize(config, control_trace)
        rung: dict[str, Any] = {"lr": lr, "shock": shock, "control": control}
        # Lead time only where the shock actually ran away (the runaway pole).
        if shock["first_nonfinite_step"] is not None:
            blowup = config.gate.fire_ratio * shock["base_peak_grad_norm"]
            rung["lead"] = _lead_time(config, shock_trace, blowup)
        ladder.append(rung)
        if lr == float(config.lr_op):
            op = {"shock": shock, "control": control, "shock_trace": shock_trace, "control_trace": control_trace}

    gate = _evaluate_surprise_gate(config, op["shock"], op["control"])

    # Grad-norm sparklines at lr_op (shared log10 scale), with the burst window bracketed.
    warm, burst = int(config.warm_steps), int(config.burst_steps)
    all_g = [r["grad_norm"] for tr in (op["shock_trace"], op["control_trace"]) for r in tr if r.get("finite", True)]
    lo, hi = (math.log10(max(min(all_g), 1e-12)), math.log10(max(all_g))) if all_g else (0.0, 1.0)
    logger.info(
        f"grad-norm @ lr_op={config.lr_op:g} (log10 {lo:.1f}..{hi:.1f}; burst = steps [{warm},{warm + burst})):\n"
        f"  shock  : {_logspark([r['grad_norm'] for r in op['shock_trace']], lo, hi)}\n"
        f"  control: {_logspark([r['grad_norm'] for r in op['control_trace']], lo, hi)}"
    )
    logger.info(_render_summary(gate, ladder))

    comparison = {
        "gate": gate,
        "ladder": [
            {"lr": x["lr"], "shock": x["shock"], "control": x["control"], "lead": x.get("lead")} for x in ladder
        ],
        "op_traces": {"shock": op["shock_trace"], "control": op["control_trace"]},
    }
    (out_dir / "comparison.json").write_text(dumps_valid(comparison), encoding="utf-8")
    logger.info(f"wrote comparison + gate to {out_dir / 'comparison.json'}")

    if not gate["passed"]:
        logger.error(
            "Gate did NOT pass. Either the base was not competent / the burst was not genuinely "
            "surprising, the shock arm did not fire a grad-norm change point, or the control arm did "
            "not stay quiet — investigate the regime before trusting the surprise read downstream."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
