"""The Phase-1 multi-arm harness — live / PC / B5 co-logged into one canonical artifact.

Phase 0 left the loop's pieces scattered: the adapting backbone ran in one place
(:mod:`cafl4ds.loop`), a separate positive-control *script* manufactured each failure mode,
and the frozen-backbone floor (B5) was measured only at run-end. Phase 1 reads every live
health trajectory *against its gate* (the load-bearing Phase-0 lesson — a signal is only
interpretable against its gate), so the three arms have to be one loop, co-logged on a shared
step index, under one schema.

This module is that orchestration, kept Hydra-free (the wiring lives in
``scripts/run_harness.py``) so the arms, the alignment, the gate, and the canonical
``comparison.json`` schema are all unit-testable:

* **live** — the adapting backbone under the base filter (a :class:`~cafl4ds.loop.StreamingLoop`
  run, reused untouched).
* **PC** — a *manufactured* pathology whose instruments must fire (the gate). The shipped smoke
  PC is Phase-0's cheapest single-pass calibrated fire: SimSiam with anti-collapse **off**
  (``anti_collapse=False``), whose SSL loss rides to its ``-1`` constant-solution floor and whose
  projector RankMe craters. Per-mode PC scripts (forgetting, …) plug into the same arm slot.
* **B5** — the frozen, init-matched backbone floor: measured only, never stepped, stamped onto
  the live arm's step grid so all three arms share one index.

The harness is deliberately **number-free** (P1.0.0): it proves the rig *runs*, the instruments
move in the right *direction*, the PC fires, and nothing goes non-finite. Reading movement *as
degradation* is the mode studies' job (P1.3 / P1.4).
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.optim.lr_scheduler import LRScheduler

from cafl4ds.data.streams import StreamBatch
from cafl4ds.eval import PerEraProbe
from cafl4ds.filters.base import Filter
from cafl4ds.loop import StreamingLoop
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger, read_run
from cafl4ds.ssl.base import SSLMethod

# The canonical `comparison.json` base-schema version. Later studies extend the artifact with a
# mode block (collapse adds `envelope`, forgetting adds `savings`, …) but a reader can always
# fall back to this base; bump only on a breaking change to the base.
SCHEMA_VERSION = 1


@dataclass
class Arm:
    """One arm of the harness — its full run-log records, keyed by its role.

    Attributes:
        name: The run name recorded on the arm's log lines.
        role: The arm's role — ``"live"``, ``"pc"``, or ``"b5"`` (the key it takes in the
            canonical artifact's ``arms`` map).
        records: The arm's full run log (the ``loss`` and ``health`` series, as read back from
            its JSONL); a measured-only arm (B5) carries only the ``health`` series.
    """

    name: str
    role: str
    records: list[dict[str, Any]]

    @property
    def health(self) -> list[dict[str, Any]]:
        """The arm's health series (one record per monitor checkpoint)."""
        return [r for r in self.records if r.get("series") == "health"]

    @property
    def loss_records(self) -> list[dict[str, Any]]:
        """The arm's per-step loss series (empty for a measured-only arm)."""
        return [r for r in self.records if r.get("series") == "loss"]

    @property
    def loss_floor(self) -> float | None:
        """Minimum SSL loss over all steps (the collapse "rode to its floor?" fingerprint)."""
        vals = [r["loss"] for r in self.loss_records if r.get("loss") is not None]
        return min(vals) if vals else None

    @property
    def diverged(self) -> bool:
        """Whether any step went non-finite (the P0.4.0 divergence fingerprint)."""
        return any(r.get("finite") is False for r in self.loss_records)


def run_stream_arm(
    *,
    name: str,
    role: str,
    method: SSLMethod,
    stream: Iterable[StreamBatch],
    optimizer: optim.Optimizer,
    selection_filter: Filter,
    monitor: HealthMonitor,
    out_dir: str | Path,
    eval_every: int = 5,
    update_every: int = 1,
    epochs: int = 1,
    scheduler: LRScheduler | None = None,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
    era_evaluator: PerEraProbe | None = None,
) -> Arm:
    """Run one stepped arm as a :class:`~cafl4ds.loop.StreamingLoop` and read its log back.

    A thin wrapper over the Phase-0 loop (reused untouched) that writes the arm's run log into
    ``out_dir`` and returns it as an :class:`Arm`. This is the ``live`` arm and the ``pc`` arm;
    the frozen ``b5`` arm uses :func:`run_frozen_arm` instead (it is never stepped).

    Args:
        name: Run name (also the log filename ``<name>.jsonl``).
        role: The arm's role in the artifact (``"live"`` or ``"pc"``).
        method: The SSL method to adapt.
        stream: The stream to iterate (the ``F`` factor).
        optimizer: Optimizer over ``method.parameters()``.
        selection_filter: The selection knob (the ``A`` factor).
        monitor: The health monitor read every ``eval_every`` steps.
        out_dir: Directory the arm's run log is written to.
        eval_every: Monitor cadence (steps).
        update_every: Optimizer cadence on the incoming-batch clock. ``1`` updates every batch.
        epochs: Passes over the stream (``>1`` for the multi-epoch collapse-PC horizon).
        scheduler: Optional per-step LR scheduler.
        grad_clip: Global grad-norm clip, or ``None`` to disable.
        device: Torch device.
        era_evaluator: Optional probe-on-past evaluator (unused by the smoke arms).

    Returns:
        The completed :class:`Arm`.
    """
    run_log_path = Path(out_dir) / f"{name}.jsonl"
    run_logger = RunLogger(run_log_path, run_name=name)
    loop = StreamingLoop(
        stream=stream,
        method=method,
        optimizer=optimizer,
        selection_filter=selection_filter,
        monitor=monitor,
        run_logger=run_logger,
        eval_every=eval_every,
        update_every=update_every,
        epochs=epochs,
        scheduler=scheduler,
        grad_clip=grad_clip,
        device=device,
        era_evaluator=era_evaluator,
    )
    loop.run()
    return Arm(name=name, role=role, records=read_run(run_log_path))


def run_frozen_arm(
    *,
    name: str,
    frozen_method: SSLMethod,
    monitor: HealthMonitor,
    grid: list[tuple[int, int]],
    device: str = "cpu",
) -> Arm:
    """Build the frozen B5 floor arm — measured only, stamped onto the live arm's step grid.

    B5 is the init-matched backbone that is *never* updated, so its representation is constant:
    the monitor reads identical values at every checkpoint (drift against its own first read is
    ``0`` throughout — B5 does not drift). It is measured once and replicated across the live
    arm's ``(step, era)`` grid so the three arms share one step index (what makes "read movement
    against the gate" well-defined).

    Args:
        name: Run name recorded on the arm's records.
        frozen_method: The frozen, init-matched twin (a snapshot taken *before* any gradient step).
        monitor: The same monitor the live arm used (a fresh drift reference is fine — a frozen
            model does not drift).
        grid: The live arm's ``(step, era)`` checkpoints to stamp the B5 reading onto.
        device: Torch device.

    Returns:
        The B5 :class:`Arm` (health-only; no loss series, as it is never stepped).
    """
    frozen_method.to(torch.device(device))
    base = {k: v for k, v in monitor.measure(frozen_method, 0).items() if k != "step"}
    records = [
        {"run": name, "series": "health", "step": float(step), "era": era, "loss": None, **base} for step, era in grid
    ]
    return Arm(name=name, role="b5", records=records)


def health_grid(arm: Arm) -> list[tuple[int, int]]:
    """The arm's ``(step, era)`` health checkpoints — the shared index the other arms align to."""
    return [(int(r["step"]), int(r["era"])) for r in arm.health]


def all_finite(*arms: Arm) -> bool:
    """Whether every numeric value logged by every arm is finite (no NaN/Inf).

    Args:
        *arms: The arms to check.

    Returns:
        ``True`` iff no loss, grad-norm, or instrument value in any arm is non-finite.
    """
    for arm in arms:
        for record in arm.records:
            for value in record.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float) and not math.isfinite(value):
                    return False
    return True


def collapse_gate(
    pc: Arm,
    *,
    pc_loss_floor_max: float,
    pc_rankme_drop_frac: float,
    surface: str = "rankme_proj",
) -> dict[str, Any]:
    """The manufactured-collapse PC gate — did the PC fire, for the right reason?

    A self-contained fingerprint robust at the toy smoke horizon (where the P0.2.1 *across-arm*
    RankMe separation has not yet developed; ``test_positive_control.py`` uses the same
    toy-horizon fingerprint): the anti-collapse-off SimSiam (1) rides its SSL loss to the ``-1``
    constant-solution floor, and (2) collapses its projector RankMe to a fraction of its own
    initial value. Both must hold.

    Args:
        pc: The PC arm (SimSiam with ``anti_collapse=False``).
        pc_loss_floor_max: The PC's loss floor must be ``<=`` this (rode toward ``-1``).
        pc_rankme_drop_frac: ``final/init`` projector RankMe must be ``<=`` this (self-collapse).
        surface: The RankMe key read for the collapse fingerprint (the projector surface).

    Returns:
        The gate verdict: ``passed`` plus the measured numbers, per-check booleans, and thresholds.
    """
    health = pc.health
    loss_floor = pc.loss_floor
    init = health[0].get(surface) if health else None
    final = health[-1].get(surface) if health else None
    drop = (final / init) if init else 1.0
    loss_fired = loss_floor is not None and loss_floor <= pc_loss_floor_max
    rank_fired = init is not None and final is not None and drop <= pc_rankme_drop_frac
    return {
        "mode": "collapse_smoke",
        "passed": bool(loss_fired and rank_fired),
        "checks": {"loss_floor_fired": bool(loss_fired), "proj_rankme_collapsed": bool(rank_fired)},
        "thresholds": {"pc_loss_floor_max": pc_loss_floor_max, "pc_rankme_drop_frac": pc_rankme_drop_frac},
        "reported": {
            "pc_loss_floor": loss_floor,
            f"pc_{surface}_init": init,
            f"pc_{surface}_final": final,
            f"pc_{surface}_drop_frac": drop,
        },
    }


def directional_report(live: Arm) -> dict[str, Any]:
    """The live arm's number-free directional check — did the instruments *move* the right way?

    Not a Go/No-Go gate (P1.0.0 owns no degradation numbers), only the harness-validation signal
    P0.1.0 already saw: under a correlated diet RankMe responds and representation drift
    accumulates. Reported alongside the gate.

    Args:
        live: The live arm.

    Returns:
        The init/final RankMe, the final cosine drift, and the two "moved?" booleans.
    """
    health = live.health
    rankme_init = health[0].get("rankme") if health else None
    rankme_final = health[-1].get("rankme") if health else None
    drift_final = health[-1].get("cosine_drift") if health else None
    return {
        "rankme_init": rankme_init,
        "rankme_final": rankme_final,
        "rankme_responded": rankme_init is not None and rankme_final is not None and rankme_final != rankme_init,
        "final_cosine_drift": drift_final,
        "drift_accumulated": drift_final is not None and drift_final > 0.0,
    }


def build_comparison(
    *,
    config_header: dict[str, Any],
    gate: dict[str, Any],
    arms: list[Arm],
    directional: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical ``comparison.json`` (base schema P1.0.0 owns).

    The base is the common core every Phase-1 harness run emits — a header, the gate verdict, and
    the per-role arm health series on a shared step index. A study extends it via ``extensions``
    (a mode block alongside ``arms``), but a reader can always fall back to the base.

    Args:
        config_header: The run's factor levels (``C``, ``I``, ``F``, ``A``, seed, alignment).
        gate: The gate verdict (e.g. from :func:`collapse_gate`).
        arms: The arms to co-log, keyed in the artifact by their ``role``.
        directional: Optional live-arm directional report (:func:`directional_report`).
        extensions: Optional mode-specific blocks merged at the top level.

    Returns:
        The canonical artifact dict, ready to serialize.
    """
    comparison: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": config_header,
        "gate": gate,
        "arms": {arm.role: {"run_name": arm.name, "health": arm.health} for arm in arms},
    }
    if directional is not None:
        comparison["directional"] = directional
    if extensions:
        comparison.update(extensions)
    return comparison


def rss_mb() -> float:
    """Current process resident-set size in MiB — the sampler for the long-horizon leak check.

    Reads ``/proc/self/statm`` (Linux; the resident field × page size) so repeated samples reflect
    *current* memory, not a monotonic peak (``ru_maxrss`` would never show a flat plateau). Falls
    back to ``ru_maxrss`` where ``/proc`` is unavailable.

    Returns:
        Resident-set size in MiB.
    """
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            resident_pages = int(f.read().split()[1])
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            raise OSError("os.sysconf is unavailable")
        return float(resident_pages * int(sysconf("SC_PAGE_SIZE")) / (1024 * 1024))
    except (FileNotFoundError, IndexError, ValueError, OSError):
        resource = importlib.import_module("resource")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024


def leak_report(rss_samples: list[float], *, growth_frac_max: float) -> dict[str, Any]:
    """Judge the long-horizon RSS trace — flat (no leak) or still climbing.

    Samples RSS after each of several repeated passes over a resampled-image stream. The right
    diagnostic is whether memory is *still climbing at the end*, not the absolute start-to-end
    growth: a caching allocator legitimately warms up over the first passes (larger tensors,
    first-touch pages) and then **plateaus**, whereas a genuine leak keeps growing pass on pass. So
    the verdict reads the **tail** — the spread across the second half of the passes: a plateau
    leaves it near zero, a leak leaves it still spreading.

    Args:
        rss_samples: RSS (MiB) sampled once per pass, in order.
        growth_frac_max: Allowed tail spread ``(max - min) / min`` over the second half before it
            is called a leak.

    Returns:
        The samples, the tail, its spread fraction, the threshold, and the ``flat`` verdict.
    """
    tail = rss_samples[len(rss_samples) // 2 :] or rss_samples
    lo, hi = min(tail), max(tail)
    spread_frac = (hi - lo) / lo if lo else 0.0
    return {
        "rss_mb": rss_samples,
        "tail_mb": tail,
        "tail_spread_frac": spread_frac,
        "growth_frac_max": growth_frac_max,
        "flat": spread_frac <= growth_frac_max,
    }


# --------------------------------------------------------------------------- P1.0.1: arm-alignment
# The Phase-0 comparison method aligns two arms by re-seeding each — so each rebuilds "the same"
# stream — and then `zip`-ing their step-indexed series post-hoc (`positive_control.py`). That
# rests on an *unverified* assumption: re-seeding reconstructs a **bit-identical** stream and keeps
# the arms in lockstep. P1.0.1 certifies it against the shared-iterator oracle — one materialized
# batch buffer every arm draws from, so the data is *provably* identical by construction — and owns
# the number (the divergence magnitude), attributing any drift to stream *construction* (does the
# rebuild match the oracle's data?) vs. *in-arm* RNG (augmentation, MAE masking, dropout, BatchNorm
# — the draws a single arm makes on top of the data).


def data_identity(rebuilt: Sequence[StreamBatch], oracle: Sequence[StreamBatch]) -> dict[str, Any]:
    """Per-step data identity of a same-seed rebuild against the shared-iterator oracle.

    Certifies the stream-construction half of the Phase-0 alignment: does re-seeding a fresh
    :class:`~cafl4ds.data.streams.EraStream` reconstruct the *same* batches the oracle materialized
    once? Compares the two batch sequences step-for-step — the ``(era, step)`` tags and the image
    tensors — and reports the worst pixel disagreement. Bit-identical (``max_abs_image_diff == 0``)
    means stream construction contributes **zero** divergence.

    Args:
        rebuilt: The same-seed rebuild's batches (a fresh ``list(EraStream(...))``).
        oracle: The oracle's materialized batch buffer (the one every arm draws from).

    Returns:
        The identity report: step count, whether the two are ``(era, step)``-aligned, the max
        absolute per-pixel image difference, and the ``bit_identical`` verdict.
    """
    aligned = len(rebuilt) == len(oracle)
    max_abs = 0.0
    for a, b in zip(rebuilt, oracle, strict=False):
        if a.era != b.era or a.step != b.step or a.images.shape != b.images.shape:
            aligned = False
            continue
        if a.images.numel():
            max_abs = max(max_abs, float((a.images - b.images).abs().max().item()))
    return {
        "num_steps": min(len(rebuilt), len(oracle)),
        "era_step_aligned": aligned,
        "max_abs_image_diff": max_abs,
        "bit_identical": bool(aligned and max_abs == 0.0),
    }


def trajectory_divergence(arm_a: Arm, arm_b: Arm) -> dict[str, Any]:
    """Step-aligned health-trajectory divergence between two arms (the worst instrument gap).

    Aligns the two arms' health series on their shared step index and, at every co-logged
    checkpoint, takes the absolute difference of each instrument value. Reports the per-instrument
    worst gap and the single worst gap over everything — the divergence *magnitude*. A run of the
    *same* arm two ways (same-seed vs oracle) should read ``0`` here; a non-zero value localizes to
    the instrument (and, given identical data, attributes to in-arm RNG).

    Args:
        arm_a: The first arm.
        arm_b: The second arm (compared against ``arm_a`` on the shared step index).

    Returns:
        The divergence report: checkpoint count, whether the step grids match exactly, the worst
        absolute divergence overall and per instrument, and the ``bit_identical`` verdict.
    """
    a_by_step = {int(r["step"]): r for r in arm_a.health}
    b_by_step = {int(r["step"]): r for r in arm_b.health}
    shared_steps = sorted(set(a_by_step) & set(b_by_step))
    per_surface: dict[str, float] = {}
    for step in shared_steps:
        ra, rb = a_by_step[step], b_by_step[step]
        for key, va in ra.items():
            vb = rb.get(key)
            if key in ("step", "era") or isinstance(va, bool) or isinstance(vb, bool):
                continue
            if isinstance(va, int | float) and isinstance(vb, int | float):
                per_surface[key] = max(per_surface.get(key, 0.0), abs(float(va) - float(vb)))
    worst = max(per_surface.values()) if per_surface else 0.0
    grid_a = [int(r["step"]) for r in arm_a.health]
    grid_b = [int(r["step"]) for r in arm_b.health]
    return {
        "num_checkpoints": len(shared_steps),
        "step_grid_matched": grid_a == grid_b,
        "max_abs_divergence": worst,
        "per_surface_max": per_surface,
        "bit_identical": bool(grid_a == grid_b and worst == 0.0),
    }


def build_divergence_report(
    *,
    config_header: dict[str, Any],
    identity: dict[str, Any],
    matched: dict[str, Any],
    desynced: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the P1.0.1 divergence artifact from the three measured components.

    The report certifies the Phase-0 same-seed ``zip`` and decomposes the divergence into its two
    contributors:

    * **stream construction** — the ``data_identity`` pixel gap (0 iff the rebuild matches the
      oracle's data);
    * **in-arm RNG** — the residual trajectory gap *given identical data*, read two ways: with
      Phase-0's per-arm re-seed in force (``matched``, expected ~0) and with the in-arm draws
      deliberately desynced (``desynced``, the lever's magnitude — what the per-arm re-seed buys).

    Args:
        config_header: The run's factor levels (``C``, ``I``, ``F``, seed, alignment).
        identity: The :func:`data_identity` report (rebuild vs. oracle data).
        matched: The :func:`trajectory_divergence` of same-seed vs. oracle (matched in-arm RNG).
        desynced: The :func:`trajectory_divergence` of the in-arm desync control (identical data,
            desynced in-arm RNG).

    Returns:
        The canonical divergence-report dict, ready to serialize.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "P1.0.1",
        "config": config_header,
        "data_identity": identity,
        "trajectory": {"matched_seed": matched, "inarm_rng_desynced": desynced},
        "attribution": {
            "stream_construction": identity["max_abs_image_diff"],
            "in_arm_rng_matched": matched["max_abs_divergence"],
            "in_arm_rng_desynced": desynced["max_abs_divergence"],
        },
        "verdict": {
            "same_seed_certified": bool(identity["bit_identical"] and matched["bit_identical"]),
            "measurement_sensitive": desynced["max_abs_divergence"] > 0.0,
        },
    }
