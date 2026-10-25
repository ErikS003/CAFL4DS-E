"""A per-capability research guard that keeps monitoring after learned function is lost."""

from __future__ import annotations

import numpy as np


class LatchedHealthGuard:
    """Offline-calibrated budgets; no labels, future health or task gradients online."""

    def __init__(self, mode="pause", budget=0.05, minimum_gap=0.25):
        if mode not in ("pause", "rollback"):
            raise ValueError("Unknown health-guard mode")
        self.mode, self.budget, self.minimum_gap = mode, budget, minimum_gap
        self.active = np.zeros(2, bool)
        self.best = np.full(2, np.inf)
        self.previous_step = 0
        self.stopped = False
        self.cutoff = None

    def observe(self, step, health, gap):
        h, g = np.asarray(health, float), np.asarray(gap, float)
        if self.stopped:
            raise ValueError("Stopped episode may not inspect later observations")
        if h.shape != (2,) or g.shape != (2,) or not np.isfinite(h).all() or not np.isfinite(g).all():
            raise ValueError("Finite day/night health and readability required")
        if step < self.previous_step:
            raise ValueError("Observations must be chronological")
        violated = self.active & (h - self.best > self.budget)
        if violated.any():
            self.stopped = True
            self.cutoff = self.previous_step if self.mode == "rollback" else step
            return {
                "action": self.mode,
                "cutoff": self.cutoff,
                "violated_domains": np.flatnonzero(violated).tolist(),
                "active": self.active.tolist(),
                "best_before_alarm": [float(v) if np.isfinite(v) else None for v in self.best],
            }
        newly = ~self.active & (g >= self.minimum_gap)
        self.active |= newly
        self.best[self.active] = np.minimum(self.best[self.active], h[self.active])
        self.previous_step = step
        return {
            "action": "continue",
            "cutoff": step,
            "active": self.active.tolist(),
            "best": [float(v) if np.isfinite(v) else None for v in self.best],
        }
