"""Give a due low-rate pulse priority over a positive corrected-J admission."""

from __future__ import annotations

from cafl4ds.single_image_stall_exploration import StallController


class PriorityStallController(StallController):
    """Treat inactivity from J rejection and covariance rejection alike."""

    def factor(self, arrival: int, j_admitted: bool) -> float:
        """Respect covariance cooldown, then apply a due pulse before normal admission."""
        if arrival < 0:
            raise ValueError("A nonnegative arrival index is required")
        if arrival < self.blocked_until:
            return 0.0
        if min(arrival - self.last_committed, arrival - self.last_exploration_attempt) >= self.stall_arrivals:
            return self.exploration_factor
        return 1.0 if j_admitted else 0.0
