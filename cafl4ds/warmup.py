"""Warm a "generally competent" well for the P1.0.2 deployment drive.

Phase 0 read health *after* the representation had settled — a competent well, warmed until the loss
plateaus (P0.3.10's reconstruction-competence plateau) — because reading a still-settling from-scratch
model confounds "the model waking up" with "a response to the stream" (P0.3.9's fresh-head transient).
This module reproduces that discipline for the deployment prototype: it trains a backbone on a
**stationary** all-regime sample (the drive's nonstationarity is deliberately withheld) until the SSL
loss stops improving, then saves the **whole method** — encoder *and* head — as a reusable well.

Saving the whole method is the load-bearing detail. The Phase-0 checkpoint utilities are encoder-only;
resuming a warm encoder onto a *fresh* head would replay the very transient P0.3.9 spent a substudy
removing. A drive that loads this well resumes the model intact, so the trajectory starts from genuine
competence with no head warm-up to misread. The well is produced once per backbone and reused across
drives (identical start state → drive-to-drive comparisons carry no per-run warmup variance).

Compute-agnostic: a single ``device`` string, no hardware branching. The stationary diet comes from
:meth:`cafl4ds.data.regime.RegimeStream.stationary_batches` (the reserved training pool, reshuffled
regime-agnostic), so the well is competent on the same probe set the drive reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from loguru import logger

from cafl4ds.data.regime import RegimeStream
from cafl4ds.ssl.base import SSLMethod

# Numerical floor so the plateau ratio is well-defined even at a near-zero loss.
_EPS = 1e-8


@dataclass(frozen=True)
class PlateauCriterion:
    """A loss-plateau stopping rule — stop when the loss stops improving.

    Compares the mean loss over the most recent ``window`` steps against the ``window`` before it; when
    the relative improvement falls below ``tol``, the loss has settled. ``min_steps`` guards against
    calling a plateau before the well has had a chance to learn.

    Attributes:
        window: Steps per comparison window.
        tol: Relative-improvement floor — below this, the loss is considered settled.
        min_steps: Minimum steps before a plateau may be declared.
    """

    window: int = 20
    tol: float = 1e-3
    min_steps: int = 40

    def settled(self, losses: list[float]) -> bool:
        """Whether the loss trace has plateaued.

        Args:
            losses: The per-step loss trace, in order.

        Returns:
            ``True`` once enough steps have run and the recent window improved on the previous one by
            less than ``tol`` (relative). A loss that stops falling — or drifts up in noise — settles.
        """
        if len(losses) < self.min_steps or len(losses) < 2 * self.window:
            return False
        previous = fmean(losses[-2 * self.window : -self.window])
        recent = fmean(losses[-self.window :])
        # Normalize on the combined magnitude so the ratio stays bounded in ~[-1, 1] even for a
        # signed loss (e.g. SimSiam's negative cosine) whose windows can straddle zero.
        relative_improvement = (previous - recent) / (abs(previous) + abs(recent) + _EPS)
        return relative_improvement < self.tol


def warm_until_settled(
    *,
    method: SSLMethod,
    stream: RegimeStream,
    optimizer: torch.optim.Optimizer,
    plateau: PlateauCriterion | None = None,
    max_steps: int = 2000,
    max_passes: int = 50,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train ``method`` on the stream's stationary diet until the loss plateaus (or a cap is hit).

    Args:
        method: The SSL method to warm (updated in place).
        stream: The regime stream whose reserved training pool supplies the stationary diet.
        optimizer: Optimizer over ``method.parameters()``.
        plateau: The plateau stopping rule; defaults to :class:`PlateauCriterion`.
        max_steps: Hard cap on optimizer steps (the plateau usually stops first).
        max_passes: Cap on passes over the stationary pool (each pass is freshly reshuffled).
        grad_clip: Global grad-norm clip, or ``None`` to disable.
        device: Torch device (``cpu`` / ``cuda`` / ``hpu``).

    Returns:
        A warm-up summary: the step count, whether it plateaued, the initial/final smoothed loss, and
        the (possibly truncated) loss trace.
    """
    criterion = plateau or PlateauCriterion()
    method.to(torch.device(device))
    method.train()
    losses: list[float] = []
    settled = False
    for _pass in range(max_passes):
        for batch in stream.stationary_batches():
            optimizer.zero_grad()
            loss = method.training_step(batch.images.to(torch.device(device)))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if len(losses) >= max_steps:
                break
            if criterion.settled(losses):
                settled = True
                break
        if settled or len(losses) >= max_steps:
            break
    window = min(criterion.window, len(losses))
    summary = {
        "steps": len(losses),
        "plateaued": settled,
        "init_loss": fmean(losses[:window]) if losses else None,
        "final_loss": fmean(losses[-window:]) if losses else None,
        "loss_trace": losses,
    }
    logger.info(
        f"warm well: {summary['steps']} steps, plateaued={settled}, "
        f"loss {summary['init_loss']:.4f} → {summary['final_loss']:.4f}"
        if losses
        else "warm well: no steps run"
    )
    return summary


def save_well(method: SSLMethod, path: str | Path) -> Path:
    """Save the **whole method** (encoder + head) as a reusable warm well.

    Unlike :func:`cafl4ds.ssl.base.save_encoder_checkpoint` (encoder only), this preserves the head so
    a drive resumes the model intact — no fresh-head transient to misread (P0.3.9).

    Args:
        method: The warmed method to serialize.
        path: Destination path (parent directories are created).

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # CPU-move before serializing (device-agnostic on disk; avoids the Habana storage-copy bug).
    state = {k: v.detach().contiguous().cpu() for k, v in method.state_dict().items()}
    torch.save(state, destination)
    logger.info(f"saved warm well (full method) to {destination}")
    return destination


def load_well(method: SSLMethod, path: str | Path) -> None:
    """Load a full-method warm well into ``method`` in place.

    Args:
        method: The freshly built method to resume (its architecture must match the saved well).
        path: The well checkpoint written by :func:`save_well`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"warm well not found: {source}")
    method.load_state_dict(torch.load(source, map_location="cpu"))
    logger.info(f"resumed warm well (full method) from {source}")
