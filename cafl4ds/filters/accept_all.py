"""The B-floor filter: accept every batch, no replay, raw stream order.

This is the *only* knob in Phase 0. It is the honest lower bound of the streaming loop — the
model sees the correlated stream exactly as it arrives, with no selection intervention — and
the reference every later filter is measured against.
"""

from __future__ import annotations

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext


class AcceptAll(Filter):
    """B-floor: accepts the entire incoming batch unchanged."""

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Return every image in ``batch`` (no selection).

        Args:
            batch: The incoming label-free stream batch.
            ctx: Unused (B-floor consults neither the model nor any health state).

        Returns:
            The batch's images unchanged.
        """
        del ctx  # B-floor is model-agnostic.
        return batch.images
