"""Quarter-rate recovery on an unbounded arrival clock and ordered video caches."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from cafl4ds.dnd_health_stopping import reading


class NaturalHealthRate:
    """Keep the prior alarm/ramp settings while continuing monitoring past 444."""

    def __init__(self, threshold=0.1, patience=2, ramp_arrivals=32, reference_step=132, interval=8) -> None:
        """Keep the preceding experiment's settings and an initially empty reference."""
        if (threshold, patience, ramp_arrivals, reference_step, interval) != (0.1, 2, 32, 132, 8):
            raise ValueError("The prior alarm and recovery settings remain fixed")
        self.reference = None
        self.streak = self.credit = 0
        self.last_alarm = None
        self.events = []

    @staticmethod
    def scheduled(step) -> bool:
        """Continue the same eight-arrival monitor schedule without an end cap."""
        return step >= 132 and (step - 132) % 8 == 0

    def observe(self, step, features):
        """Read scheduled geometry; confirmed alarms affect subsequent arrivals."""
        if not self.scheduled(step):
            return False
        if step == 132:
            if self.reference is not None:
                raise ValueError("The fixed reference cannot be replaced")
            self.reference = np.array(features, copy=True)
            return False
        if self.reference is None:
            raise ValueError("Missing arrival-132 reference")
        values = reading(self.reference, features)
        unhealthy = bool((values >= 0.1 - 1e-12).any())
        self.streak = self.streak + 1 if unhealthy else 0
        fired = self.streak >= 2
        self.events.append(
            {"step": step, "values": values.tolist(), "unhealthy": unhealthy, "streak": self.streak, "alarm": fired}
        )
        if fired:
            self.last_alarm, self.streak, self.credit = step, 0, 0
        return fired

    def admit(self, step):
        """Advance integer admission credit using elapsed arrivals since the alarm."""
        numerator = 4 if self.last_alarm is None else min(4, 1 + (step - self.last_alarm - 1) // 32)
        self.credit += numerator
        accepted = self.credit >= 4
        if accepted:
            self.credit -= 4
        return accepted, numerator / 4


def paced_mask(horizon, fork, count):
    """Accept exactly count suffix batches, without rearranging any arrivals."""
    if fork is None:
        return np.ones(horizon, dtype=bool)
    remaining = horizon - fork
    if not 0 <= count <= remaining:
        raise ValueError("Invalid suffix count")
    return np.array(
        [True] * fork + [((j + 1) * count // remaining) > (j * count // remaining) for j in range(remaining)]
    )


class OrderedFrames:
    """Read consecutive pixels across clip boundaries, without exposing attributes."""

    def __init__(self, clips) -> None:
        """Open immutable uint8 caches in their declared source order."""
        self.paths = [Path(c["frames_path"]) for c in clips]
        self.arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in self.paths]
        self.edges = [0, *np.cumsum([len(a) for a in self.arrays]).tolist()]

    def batch(self, index):
        """Read the next consecutive 32 frames, including any video boundary."""
        start, stop = index * 32, (index + 1) * 32
        if not 0 <= start < stop <= self.edges[-1]:
            raise IndexError(index)
        parts = []
        while start < stop:
            clip = bisect.bisect_right(self.edges, start) - 1
            end = min(stop, self.edges[clip + 1])
            parts.append(self.arrays[clip][start - self.edges[clip] : end - self.edges[clip]])
            start = end
        return np.array(np.concatenate(parts), copy=True)


def path_metrics(curve, points):
    """Use observed maxima and arriving-batch exposure, without invented drive time."""
    curve, points = np.asarray(curve), np.asarray(points)
    deficits = np.maximum.accumulate(curve, axis=0) - curve
    widths = np.where(points[:-1] >= 132, np.diff(points), 0)
    drop = deficits.max(0)
    return np.r_[
        curve[-1],
        drop,
        (deficits[:-1] * widths[:, None]).sum(0) / widths.sum(),
        ((deficits[:-1] > 1) * widths[:, None]).sum(0),
        max(drop[1:]),
    ]
