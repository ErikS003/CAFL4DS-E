"""FIFO replay of admitted history with a fixed optimizer-image budget."""

from __future__ import annotations

import hashlib
from collections import deque

import torch

from cafl4ds.filters.base import FilterContext, ReplayBuffer


class WindowReplay(ReplayBuffer):
    """Keep recent admitted frames and draw past frames before ingesting arrivals."""

    def __init__(self, capacity: int = 4096, replay_batch: int = 16, seed: int = 73919) -> None:
        """Configure the admitted-frame window and private sampling generator."""
        if capacity < 32 or capacity % 32 or not 0 < replay_batch < 32:
            raise ValueError("Require a capacity in full 32-frame batches and 1..31 replay images")
        self.capacity, self.replay_batch, self.seed = capacity, replay_batch, seed
        self._images = deque(maxlen=capacity)
        self._ids = deque(maxlen=capacity)
        self._hashes = deque(maxlen=capacity)
        self._gen = torch.Generator().manual_seed(seed)
        self.seen = 0
        self.last = {}

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Mix current/past frames, then store all admitted arrivals; never draw global RNG."""
        if images.ndim != 4 or len(images) != 32 or images.device.type != "cpu":
            raise ValueError("Require exactly 32 host-resident admitted frames")
        ids = list(range(32 * ctx.step, 32 * (ctx.step + 1)))
        if self._ids and ids[0] <= self._ids[-1]:
            raise ValueError("Admitted arrivals must advance strictly")
        count = min(self.replay_batch, len(self._images))
        replay_indices = torch.randperm(len(self._images), generator=self._gen)[:count].tolist() if count else []
        current_indices = (
            torch.randperm(32, generator=self._gen)[: 32 - count].sort().values.tolist() if count else list(range(32))
        )
        current = images[current_indices]
        past = torch.stack([self._images[i] for i in replay_indices]) if count else None
        old_ids = [self._ids[i] for i in replay_indices]
        result = current if past is None else torch.cat([current, past])
        for index, image in zip(ids, images, strict=True):
            item = image.detach().clone()
            self._images.append(item)
            self._ids.append(index)
            self._hashes.append(hashlib.sha256(item.contiguous().numpy().tobytes()).hexdigest())
        self.seen += len(images)
        self.last = {
            "current_ids": [ids[i] for i in current_indices],
            "replay_ids": old_ids,
            "admitted_frames_seen": self.seen,
            "resident_frames": len(self._ids),
            "oldest_resident_id": self._ids[0],
            "replay_arrival_ages": [ctx.step - i // 32 for i in old_ids],
        }
        return result

    def marker(self) -> dict:
        """Describe complete memory identity without repeatedly copying all image bytes."""
        return {
            "capacity": self.capacity,
            "replay_batch": self.replay_batch,
            "seed": self.seed,
            "ids": list(self._ids),
            "hashes": list(self._hashes),
            "rng": self._gen.get_state().clone(),
            "seen": self.seen,
        }

    def state_dict(self) -> dict:
        """Save actual resident pixels as well as identities and private RNG."""
        return {**self.marker(), "images": torch.stack(list(self._images)) if self._images else None}

    def load_state_dict(self, state: dict) -> None:
        """Validate and restore an exact independent replay window."""
        if (state["capacity"], state["replay_batch"], state["seed"]) != (self.capacity, self.replay_batch, self.seed):
            raise ValueError("Replay window configuration differs")
        ids, hashes, images = state["ids"], state["hashes"], state["images"]
        if len(ids) != len(hashes) or len(ids) > self.capacity or ids != sorted(set(ids)):
            raise ValueError("Invalid saved replay identities")
        if (images is None) != (len(ids) == 0) or (images is not None and len(images) != len(ids)):
            raise ValueError("Saved image/identity mismatch")
        self._ids = deque(ids, maxlen=self.capacity)
        self._hashes = deque(hashes, maxlen=self.capacity)
        self._images = deque(maxlen=self.capacity)
        for image, expected in zip([] if images is None else images, hashes, strict=True):
            item = image.detach().cpu().clone()
            if hashlib.sha256(item.contiguous().numpy().tobytes()).hexdigest() != expected:
                raise ValueError("Saved replay pixels differ")
            self._images.append(item)
        self._gen.set_state(state["rng"].cpu())
        self.seen = state["seen"]
