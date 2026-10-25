"""Change replay allocation after health alarms without dropping optimizer updates."""

from __future__ import annotations

import torch

from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.window_replay import WindowReplay
from cafl4ds.natural_health_rate import NaturalHealthRate


class HealthReplayMix(NaturalHealthRate):
    """Use the existing causal alarms to allocate a fixed thirty-two-image update."""

    def split(self, step: int) -> tuple[int, int]:
        """Recover current images in thirty-two-arrival stages after the latest alarm."""
        if step < 1 or (self.last_alarm is not None and step <= self.last_alarm):
            raise ValueError("Request the mix for an arrival after the latest observation")
        stage = 4 if self.last_alarm is None else min(4, 1 + (step - self.last_alarm - 1) // 32)
        current = 8 * stage
        return current, 32 - current


class VariableWindowReplay(WindowReplay):
    """Ingest every arrival while varying the number of past images sampled before ingestion."""

    def __init__(self, capacity: int = 256, seed: int = 73919) -> None:
        """Reuse the verified FIFO and private sampling generator with an initially current-only mix."""
        super().__init__(capacity=capacity, replay_batch=16, seed=seed)
        self.replay_batch = 0

    def observe(self, images: torch.Tensor, ctx: FilterContext, replay_batch: int = 0) -> torch.Tensor:
        """Sample the declared past fraction, then retain every incoming frame."""
        if replay_batch not in (0, 8, 16, 24):
            raise ValueError("Replay allocation must be 0, 8, 16 or 24")
        self.replay_batch = replay_batch
        return super().observe(images, ctx)

    def load_state_dict(self, state: dict) -> None:
        """Restore the last allocation as well as exact pixels, identities and sampling state."""
        if (state["capacity"], state["seed"]) != (self.capacity, self.seed):
            raise ValueError("Replay window configuration differs")
        if state["replay_batch"] not in (0, 8, 16, 24):
            raise ValueError("Invalid saved replay allocation")
        self.replay_batch = state["replay_batch"]
        super().load_state_dict(state)
