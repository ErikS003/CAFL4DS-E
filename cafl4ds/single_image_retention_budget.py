"""A cumulative anchor ceiling and a bounded window for singleton adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetentionBudget:
    """Carry window state across accepted updates and interrupted execution."""

    anchor: float
    window_anchor: float
    global_fraction: float
    window_fraction: float
    window_updates: int
    accepted_in_window: int = 0

    def __post_init__(self) -> None:
        """Reject budgets that cannot express a finite, cumulative constraint."""
        if not np.isfinite([self.anchor, self.window_anchor, self.global_fraction, self.window_fraction]).all():
            raise ValueError("Finite budget measurements required")
        if min(self.anchor, self.window_anchor) <= 0 or min(self.global_fraction, self.window_fraction) < 0:
            raise ValueError("Positive objectives and nonnegative allowances required")
        if self.window_updates < 1 or not 0 <= self.accepted_in_window < self.window_updates:
            raise ValueError("Invalid accepted-update window")

    @property
    def ceiling(self) -> float:
        """The original anchor always bounds cumulative deterioration."""
        return min(self.anchor * (1 + self.global_fraction), self.window_anchor * (1 + self.window_fraction))

    def passes(self, objective: float, covariance: float, cap: float) -> bool:
        """Apply the same measured rule regardless of the predictor's sign."""
        if not np.isfinite([objective, covariance, cap]).all() or cap <= 0:
            raise ValueError("Finite measurements and a positive covariance cap required")
        return bool(objective <= self.ceiling + 1e-12 and covariance <= cap + 1e-12)

    def commit(self, objective: float) -> None:
        """Only retained updates consume a window; rejected trials cannot reset it."""
        if not np.isfinite(objective) or objective <= 0 or objective > self.ceiling + 1e-12:
            raise ValueError("Cannot commit an objective outside the retention budget")
        self.accepted_in_window += 1
        if self.accepted_in_window == self.window_updates:
            self.window_anchor = float(objective)
            self.accepted_in_window = 0
