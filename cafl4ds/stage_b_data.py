"""Causal memory identities and fixed work schedules for Stage B."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class IndexReservoir:
    """Algorithm-R identity memory, drawing replay strictly before admission."""

    def __init__(self, seed: int, capacity: int = 256) -> None:
        """Use a private RNG so admission cannot alter SSL augmentation draws."""
        self.generator = torch.Generator().manual_seed(seed)
        self.capacity = capacity
        self.buffer: list[int] = []
        self.seen: set[int] = set()

    def batch(self, incoming: list[int]) -> dict[str, Any]:
        """Return32 exposures while admitting all32 unique arrivals."""
        if len(incoming) != 32 or len(set(incoming)) != 32 or self.seen.intersection(incoming):
            raise ValueError("Stage B stream must be single-pass, unique batch32")
        before = self.buffer.copy()
        replay = [before[i] for i in torch.randperm(len(before), generator=self.generator)[:16].tolist()]
        count = 32 - len(replay)
        current = (
            incoming
            if count == 32
            else [incoming[i] for i in torch.randperm(32, generator=self.generator)[:count].tolist()]
        )
        for identity in incoming:
            if len(self.buffer) < self.capacity:
                self.buffer.append(identity)
            else:
                slot = int(torch.randint(0, len(self.seen) + 1, (1,), generator=self.generator))
                if slot < self.capacity:
                    self.buffer[slot] = identity
            self.seen.add(identity)
        return {
            "current": current,
            "replay": replay,
            "buffer_before": before,
            "buffer_after": self.buffer.copy(),
            "seen": len(self.seen),
        }


def online_lr(step: int) -> float:
    """The one frozen24-update schedule, indexed from zero."""
    if not 0 <= step < 24:
        raise ValueError("Online step outside the frozen horizon")
    if step < 4:
        return 1e-5 * (step + 1) / 4
    return 1e-6 + (1e-5 - 1e-6) * (1 + math.cos(math.pi * (step - 3) / 20)) / 2


def repeated_batches(size: int, updates: int, seed: int) -> list[list[int]]:
    """Cycle independently shuffled fitting epochs for declared warm-up/controls."""
    rng = np.random.default_rng(seed)
    chunks: list[int] = []
    while len(chunks) < updates * 32:
        chunks.extend(rng.permutation(size).tolist())
    return [chunks[start : start + 32] for start in range(0, updates * 32, 32)]
