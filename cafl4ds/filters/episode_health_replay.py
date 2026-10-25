"""Emit one geometry alarm per unhealthy episode while retaining timed recovery."""

from __future__ import annotations

import copy

import numpy as np
import torch

from cafl4ds.dnd_health_stopping import reading
from cafl4ds.filters.health_replay_mix import HealthReplayMix


class EpisodeHealthReplay(HealthReplayMix):
    """Fixed 0.10 entry, 0.05 rearming, two-check persistence, and unchanged ramp."""

    def __init__(self) -> None:
        """Preserve the inherited reference/cadence/allocation and add episode state."""
        super().__init__()
        self.latched = False
        self.low_streak = 0
        self.last_observation = None

    def observe(self, step, features):
        """Observe after training; any alarm affects only subsequent arrivals."""
        if not self.scheduled(step):
            return False
        if self.last_observation is not None and step <= self.last_observation:
            raise ValueError("Health observations must advance strictly")
        features = np.asarray(features)
        if features.ndim != 2 or len(features) <= features.shape[1] or not np.isfinite(features).all():
            raise ValueError("Finite monitor bank larger than feature width required")
        if step == 132:
            if self.reference is not None:
                raise ValueError("The fixed reference cannot be replaced")
            self.reference = np.array(features, copy=True)
            self.last_observation = step
            return False
        if self.reference is None:
            raise ValueError("Missing arrival-132 reference")
        values = reading(self.reference, features)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite health reading")
        score = float(np.max(values))
        high, low = score >= 0.1 - 1e-12, score <= 0.05 + 1e-12
        fired = rearmed = False
        if self.latched:
            self.streak = 0
            self.low_streak = self.low_streak + 1 if low else 0
            if self.low_streak >= 2:
                self.latched, self.low_streak, rearmed = False, 0, True
        else:
            self.low_streak = 0
            self.streak = self.streak + 1 if high else 0
            if self.streak >= 2:
                self.latched, self.streak, fired = True, 0, True
                self.last_alarm, self.credit = step, 0
        self.last_observation = step
        self.events.append(
            {
                "step": step,
                "values": values.tolist(),
                "score": score,
                "unhealthy": high,
                "low": low,
                "streak": self.streak,
                "low_streak": self.low_streak,
                "latched": self.latched,
                "rearmed": rearmed,
                "alarm": fired,
            }
        )
        return fired

    def state_dict(self):
        """Save all inherited and added controller state using safe tensor storage."""
        state = copy.deepcopy(vars(self))
        if state["reference"] is not None:
            state["reference"] = torch.from_numpy(state["reference"])
        return state

    def load_state_dict(self, state):
        """Restore a complete independently owned controller snapshot."""
        if set(state) != set(vars(self)):
            raise ValueError("Incomplete episode controller state")
        restored = copy.deepcopy(state)
        if restored["reference"] is not None:
            restored["reference"] = restored["reference"].cpu().numpy().copy()
        self.__dict__.update(restored)


def timing_schedules(current_counts):
    """Build the four predeclared retrospective controls without changing replay dose."""
    values = np.asarray(current_counts, dtype=int)
    if values.shape != (4096,) or not np.isin(values, (8, 16, 24, 32)).all():
        raise ValueError("Expected 4096 valid allocations")
    if not (values[:132] == 32).all():
        raise ValueError("Pre-reference allocations must be all-current")
    result = {}
    for seed in (87001, 87007, 87013):
        generator = torch.Generator().manual_seed(seed)
        blocks = values[132:4092].reshape(495, 8)
        order = torch.randperm(495, generator=generator).numpy()
        result[f"perm_{seed}"] = np.r_[values[:132], blocks[order].ravel(), values[4092:]].tolist()
    result["shift_256"] = np.r_[values[:132], np.roll(values[132:], 256)].tolist()
    return result
