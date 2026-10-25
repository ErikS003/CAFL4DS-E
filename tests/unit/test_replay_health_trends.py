"""Health trends must retain causal time and trajectory identity."""
from __future__ import annotations

import pytest

from cafl4ds.replay_health_trends import health_trends
from cafl4ds.replay_local import HEALTH


def test_trend_uses_elapsed_updates_and_rejects_future_or_other_trajectory() -> None:
    """A compatible earlier observation defines change per update only."""
    previous = {"parent_state": "s", "schedule": "day_then_night", "pool_group": "p", "prefix": 8,
                "features": dict.fromkeys(HEALTH, 1.0)}
    current = {**previous, "prefix": 10, "features": dict.fromkeys(HEALTH, 5.0)}
    assert health_trends(current, previous) == dict.fromkeys((f"trend_{k}" for k in HEALTH), 2.0)
    with pytest.raises(ValueError, match="future"):
        health_trends(previous, current)
    with pytest.raises(ValueError, match="different trajectories"):
        health_trends(current, {**previous, "pool_group": "other"})
