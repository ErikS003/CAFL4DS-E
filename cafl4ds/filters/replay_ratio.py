"""Steer current/replay exposure while preserving action-independent memory."""

from __future__ import annotations

import copy
from typing import Any

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.filters.reservoir import ReservoirReplay


class ReplayRatio(Filter):
    """Uniform nested current/replay draws with a fixed32-image optimizer batch."""

    def __init__(self, replay_count: int = 16, capacity: int = 256, seed: int = 0) -> None:
        """Keep selection RNG separate from the stream-wide reservoir RNG."""
        self.reservoir = ReservoirReplay(capacity, 24, seed)
        self.seed = seed
        self.last: dict[str, Any] = {}
        self.set_action(replay_count)

    def set_action(self, replay_count: int) -> None:
        """Select one of three prespecified replay allocations."""
        if replay_count not in (8, 16, 24):
            raise ValueError("Replay count must be8,16 or24")
        self.replay_count = replay_count

    def restore_memory(self, state: dict[str, Any]) -> None:
        """Restore the all-arrival reservoir from a saved SteerableReplay parent."""
        if len(state["buffer"]) < 24 or len(state["buffer"]) > self.reservoir.capacity:
            raise ValueError("Require a populated compatible parent reservoir")
        self.reservoir._buffer = copy.deepcopy(state["buffer"])
        self.reservoir._seen = state["seen"]
        self.reservoir._gen.set_state(state["rng"].clone())

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Draw before ingest; all arrivals enter memory regardless of selected exposure."""
        if len(batch.images) != 32 or len(self.reservoir._buffer) < 24:
            raise ValueError("Require32 arrivals and at least24 historical images")
        order = torch.randperm(len(self.reservoir._buffer), generator=self.reservoir._gen)
        replay_indices = order[: self.replay_count].tolist()
        generator = torch.Generator().manual_seed(self.seed + 104729 * (ctx.step + 1))
        current_indices = torch.randperm(32, generator=generator)[: 32 - self.replay_count].sort().values
        current = batch.images[current_indices.to(batch.images.device)]
        replay = torch.stack([self.reservoir._buffer[i] for i in replay_indices]).to(current.device)
        selected = torch.cat((current, replay))
        for image in batch.images:
            self.reservoir._ingest(image.cpu())
        self.last = {
            "step": ctx.step,
            "current_indices": current_indices.tolist(),
            "replay_indices": replay_indices,
            "current_count": len(current),
            "replay_count": len(replay),
            "reservoir_seen": self.reservoir._seen,
        }
        return selected
