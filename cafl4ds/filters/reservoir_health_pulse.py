"""Fixed-budget historical replay with causal geometry-triggered extra replay pulses."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from cafl4ds.filters.episode_health_replay import EpisodeHealthReplay
from cafl4ds.filters.historical_reservoir import HistoricalReservoir


class ReservoirHealthPulse(EpisodeHealthReplay):
    """Keep 16/16 normally, using 8/24 for the 32 arrivals after an episode alarm."""

    def split(self, step):
        """Act only after observation; a fixed pulse ends even if the episode stays latched."""
        if step < 1 or (self.last_alarm is not None and step <= self.last_alarm):
            raise ValueError("Request allocations after the latest alarm")
        past = 24 if self.last_alarm is not None and step <= self.last_alarm + 32 else 16
        return 32 - past, past


class PulseReservoir(HistoricalReservoir):
    """Preserve Algorithm R and both private RNG streams while selecting 16 or 24 old frames."""

    def observe(self, images, ctx, replay_batch=16):
        """Draw nested current/history subsets, then admit every incoming frame independently."""
        if replay_batch not in (16, 24):
            raise ValueError("Replay must be 16 or 24")
        if images.ndim != 4 or len(images) != 32 or images.device.type != "cpu":
            raise ValueError("Require exactly 32 host-resident frames")
        if 32 * ctx.step != self.seen:
            raise ValueError("Require consecutive arrivals from zero")
        count = min(replay_batch, len(self._ids))
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


def shifted_pulses(current_counts):
    """Preserve the exact dose while circularly shifting the post-reference allocations."""
    values = np.asarray(current_counts, dtype=int)
    if values.shape != (4096,) or values[0] != 32 or not np.all(values[1:132] == 16):
        raise ValueError("Expected a 4096-arrival reservoir trajectory with normal startup")
    if not np.isin(values[132:], (8, 16)).all():
        raise ValueError("Expected only normal or pulse allocations")
    return {
        f"shift_{offset}": np.r_[values[:132], np.roll(values[132:], offset)].tolist() for offset in (256, 512, 1024)
    }
