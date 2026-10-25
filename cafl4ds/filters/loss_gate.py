"""Loss-gate — keep the high-loss frames (SOFed 2022; loss-based importance).

A loss-based importance knob: score each incoming frame by the SSL loss the **current** model
incurs on it (:meth:`~cafl4ds.ssl.base.SSLMethod.per_sample_loss` — MAE's reconstruction error,
SimSiam's negative cosine) and keep only the fraction with the highest loss. High loss = the
model reconstructs / matches this frame poorly, i.e. it is the informative, not-yet-learned
part of the stream; low-loss frames are already-fit and cheap to skip. This is the "closest
prior art" loss-importance selection (baseline **B3**), run here as a streaming, label-free,
single-pass training-selection signal.

Like SemDeDup this is an *admission* knob (it narrows the batch) and consults only the live
model, never labels or a stored gradient.
"""

from __future__ import annotations

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext


class LossGate(Filter):
    """Admission knob: keep the top ``keep_frac`` of frames by current per-sample SSL loss."""

    def __init__(self, keep_frac: float = 0.5) -> None:
        """Configure the loss-gate.

        Args:
            keep_frac: Fraction of each batch to keep (the highest-loss frames), in ``(0, 1]``.
                At least one frame is always kept. ``1.0`` keeps the whole batch (a no-op gate).

        Raises:
            ValueError: If ``keep_frac`` is not in ``(0, 1]``.
        """
        if not 0.0 < keep_frac <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1]; got {keep_frac}.")
        self.keep_frac = keep_frac

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Return the highest-loss ``keep_frac`` subset of the batch (in stream order).

        Args:
            batch: The incoming label-free stream batch.
            ctx: The live filter context; its ``method`` scores per-frame loss (fast edge).

        Returns:
            The kept images ``[K, C, H, W]`` with ``K = max(1, round(keep_frac * B))``.
        """
        b = batch.images.shape[0]
        keep_n = max(1, round(self.keep_frac * b))
        if keep_n >= b:
            return batch.images
        losses = ctx.method.per_sample_loss(batch.images)  # [B], no grad, higher = keep
        top = torch.topk(losses, keep_n).indices
        top, _ = torch.sort(top)  # preserve stream order among the kept frames
        return batch.images[top]
