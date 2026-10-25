"""Identity-tracked Algorithm R with replay sampling independent of replacement."""

from __future__ import annotations

import hashlib

import torch

from cafl4ds.filters.base import FilterContext, ReplayBuffer


class HistoricalReservoir(ReplayBuffer):
    """Keep a uniform sample of all arrivals under the existing 32-image update budget."""

    def __init__(self, capacity=256, sample_seed=73919, admission_seed=90101):
        """Separate the matched FIFO sampling RNG from Algorithm R's admission RNG."""
        if capacity < 32 or capacity % 32:
            raise ValueError("Capacity must contain whole 32-image arrivals")
        self.capacity, self.sample_seed, self.admission_seed = capacity, sample_seed, admission_seed
        self._sample = torch.Generator().manual_seed(sample_seed)
        self._admit = torch.Generator().manual_seed(admission_seed)
        self._images, self._ids, self._hashes = [], [], []
        self.seen, self.last = 0, {}

    def observe(self, images, ctx: FilterContext):
        """Sample before ingestion; all incoming frames are eligible for uniform retention."""
        if images.ndim != 4 or len(images) != 32 or images.device.type != "cpu":
            raise ValueError("Require exactly 32 host-resident frames")
        if 32 * ctx.step != self.seen:
            raise ValueError("Require consecutive arrivals from zero")
        count = min(16, len(self._ids))
        old = torch.randperm(len(self._ids), generator=self._sample)[:count].tolist() if count else []
        now = (
            torch.randperm(32, generator=self._sample)[: 32 - count].sort().values.tolist()
            if count
            else list(range(32))
        )
        historical = [self._ids[i] for i in old]
        mixed = torch.cat([images[now], torch.stack([self._images[i] for i in old])]) if count else images
        current = [self.seen + i for i in now]
        retained = 0
        for image in images:
            identity = self.seen
            slot = (
                len(self._ids)
                if self.seen < self.capacity
                else int(torch.randint(self.seen + 1, (1,), generator=self._admit).item())
            )
            if slot < self.capacity:
                item = image.detach().clone()
                checksum = hashlib.sha256(item.contiguous().numpy().tobytes()).hexdigest()
                if len(self._ids) < self.capacity:
                    self._ids.append(identity)
                    self._images.append(item)
                    self._hashes.append(checksum)
                else:
                    self._ids[slot], self._images[slot], self._hashes[slot] = identity, item, checksum
                retained += 1
            self.seen += 1
        self.last = {
            "current_ids": current,
            "replay_ids": historical,
            "admitted_frames_seen": self.seen,
            "resident_frames": len(self._ids),
            "oldest_resident_id": min(self._ids),
            "replay_arrival_ages": [ctx.step - i // 32 for i in historical],
            "retained_arrivals": retained,
        }
        return mixed

    def marker(self):
        """Describe resident content and both private RNG states."""
        return {
            "capacity": self.capacity,
            "sample_seed": self.sample_seed,
            "admission_seed": self.admission_seed,
            "seen": self.seen,
            "ids": list(self._ids),
            "hashes": list(self._hashes),
            "sample_rng": self._sample.get_state().clone(),
            "admission_rng": self._admit.get_state().clone(),
        }

    def state_dict(self):
        """Persist actual resident images alongside identities and generators."""
        return {**self.marker(), "images": torch.stack(self._images) if self._images else None}

    def load_state_dict(self, state):
        """Restore a complete reservoir, rejecting inconsistent or changed pixels."""
        if (state["capacity"], state["sample_seed"], state["admission_seed"]) != (
            self.capacity,
            self.sample_seed,
            self.admission_seed,
        ):
            raise ValueError("Reservoir configuration differs")
        ids, hashes, images, seen = state["ids"], state["hashes"], state["images"], state["seen"]
        if seen < 0 or seen % 32 or len(ids) != min(seen, self.capacity):
            raise ValueError("Invalid reservoir population")
        if len(set(ids)) != len(ids) or any(i < 0 or i >= seen for i in ids) or len(hashes) != len(ids):
            raise ValueError("Invalid reservoir identities")
        if (images is None) != (not ids) or (images is not None and len(images) != len(ids)):
            raise ValueError("Image population differs")
        pixels = []
        for image, checksum in zip([] if images is None else images, hashes, strict=True):
            item = image.detach().cpu().clone()
            if hashlib.sha256(item.contiguous().numpy().tobytes()).hexdigest() != checksum:
                raise ValueError("Reservoir pixels differ")
            pixels.append(item)
        self._images, self._ids, self._hashes, self.seen = pixels, list(ids), list(hashes), seen
        self._sample.set_state(state["sample_rng"].cpu())
        self._admit.set_state(state["admission_rng"].cpu())
