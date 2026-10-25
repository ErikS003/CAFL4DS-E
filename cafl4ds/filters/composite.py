"""The composite selector — combine admission filters and a replay buffer (the ``A`` factor).

The cheap Phase-0 knobs split into two composable roles (see :mod:`cafl4ds.filters.base`):
zero or more **admission** :class:`~cafl4ds.filters.base.Filter` s that narrow the incoming
batch, and at most one **replay** :class:`~cafl4ds.filters.base.ReplayBuffer` that stores and
replays. :class:`CompositeSelector` wires them behind the single ``Filter.select`` interface
the loop calls, so *what* combination of knobs is active is purely a config choice:

======================  =========================  ====================
Baseline                admission                  buffer
======================  =========================  ====================
B-floor                 —                          —
B1  (reservoir)         —                          ReservoirReplay
dedup                   SemDeDup                    —
loss-gate  (B3)         LossGate                   —
B1.5  (dedup+replay)    SemDeDup                    ReservoirReplay
======================  =========================  ====================

Admission filters run in listed order (each narrows what the next sees); the buffer always
runs last, on the admitted frames. This ordering is structural: dedup/loss-gate decide what is
*worth admitting*, and the buffer then decides what of the admitted history to *replay* — the
reverse (replaying, then deduping the replay) would be meaningless.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from loguru import logger

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext, ReplayBuffer


class CompositeSelector(Filter):
    """Compose an admission-filter pipeline with an optional replay buffer."""

    def __init__(self, admission: list[Filter] | None = None, buffer: ReplayBuffer | None = None) -> None:
        """Build the composite.

        Args:
            admission: Admission filters applied in order (each narrows the batch). ``None`` or
                an empty list means accept everything (no narrowing) — the B-floor admission.
            buffer: An optional replay buffer run after admission. ``None`` trains on the
                admitted frames directly (no replay).
        """
        self.admission = list(admission or [])
        self.buffer = buffer

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Run the admission pipeline, then the buffer, and return the training batch.

        Args:
            batch: The incoming label-free stream batch.
            ctx: The live filter context (model, step), forwarded to every stage.

        Returns:
            The images the SSL update trains on this step. May be *smaller* than the incoming
            batch (admission narrowed it, possibly to empty) or *larger* (the buffer replayed
            past frames).
        """
        images = batch.images
        for stage in self.admission:
            images = stage.select(replace(batch, images=images), ctx)
            if images.shape[0] == 0:
                logger.debug(f"step {ctx.step}: admission emptied the batch at {type(stage).__name__}.")
                break
        if self.buffer is not None:
            images = self.buffer.observe(images, ctx)
        return images

    def auxiliary_loss(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor | None:
        """Forward an optional regularization loss from the configured replay buffer."""
        if self.buffer is None:
            return None
        return self.buffer.auxiliary_loss(images, ctx)
