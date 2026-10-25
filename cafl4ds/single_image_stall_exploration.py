"""Sparse low-rate exploration after inactivity, with bounded covariance retries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StallController:
    """Keep scheduling state separate from the transactional learner state."""

    stall_arrivals: int = 256
    exploration_factor: float = 0.0625
    covariance_cooldown: int = 64
    last_committed: int = -1
    last_exploration_attempt: int = -1
    blocked_until: int = 0

    def __post_init__(self) -> None:
        """Validate counters and a strictly reduced exploratory step."""
        counters = (
            self.stall_arrivals,
            self.covariance_cooldown,
            self.last_committed,
            self.last_exploration_attempt,
            self.blocked_until,
        )
        if any(type(v) is not int for v in counters):
            raise ValueError("Integer controller counters required")
        if self.stall_arrivals < 1 or self.covariance_cooldown < 0 or self.blocked_until < 0:
            raise ValueError("Invalid controller intervals")
        if min(self.last_committed, self.last_exploration_attempt) < -1:
            raise ValueError("Invalid controller history")
        if not np.isfinite(self.exploration_factor) or not 0 < self.exploration_factor < 1:
            raise ValueError("A finite exploratory factor strictly below one is required")

    def factor(self, arrival: int, j_admitted: bool) -> float:
        """Use zero-based continuation arrivals; the 256th idle image may explore."""
        if arrival < 0:
            raise ValueError("A nonnegative arrival index is required")
        if arrival < self.blocked_until:
            return 0.0
        if j_admitted:
            return 1.0
        if min(arrival - self.last_committed, arrival - self.last_exploration_attempt) >= self.stall_arrivals:
            return self.exploration_factor
        return 0.0

    def finish(self, arrival: int, exploratory: bool, accepted: bool) -> None:
        """A failed covariance trial cools down without pretending learning happened."""
        if exploratory:
            self.last_exploration_attempt = arrival
        if accepted:
            self.last_committed = arrival
        else:
            self.blocked_until = arrival + 1 + self.covariance_cooldown


def covariance_only(embeddings: np.ndarray) -> float:
    """Match the existing normalized off-diagonal covariance without other assays."""
    z = np.asarray(embeddings, np.float64)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("A finite matrix with at least two bank examples is required")
    q = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    covariance = np.atleast_2d(np.cov(q, rowvar=False))
    offdiag = covariance - np.diag(np.diag(covariance))
    return float((offdiag**2).sum() / z.shape[1])
