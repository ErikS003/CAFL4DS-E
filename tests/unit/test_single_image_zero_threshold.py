"""The looser J cutoff must wait for successful predecessor completion."""

import numpy as np
import pytest

from scripts.run_single_image_zero_threshold import predecessor_ready


def test_zero_cutoff_admits_a_superset_and_rejects_zero() -> None:
    """Changing a negative upper cutoff to zero makes admission easier."""
    predictions = np.array([-0.01, -0.001152126437698479, -0.0001, 0.0, 0.001])
    original, easier = predictions < -0.001152126437698479, predictions < 0.0
    if not np.all(~original | easier) or easier.sum() <= original.sum() or easier[3]:
        raise AssertionError("Wrong threshold direction or equality handling")


def test_running_or_incompletely_verified_predecessor_does_not_release() -> None:
    """Neither a running process nor an incomplete final publication releases the job."""
    if predecessor_ready({"stage": "complete", "running": True}, {"offered": 57600}, {"passed": True}, 57600):
        raise AssertionError("A running predecessor released GPU work")
    if predecessor_ready({"stage": "complete", "running": False}, {"offered": 57600}, None, 57600):
        raise AssertionError("Missing verification released GPU work")
    if not predecessor_ready({"stage": "complete", "running": False}, {"offered": 57600}, {"passed": True}, 57600):
        raise AssertionError("A verified completed predecessor did not release")


@pytest.mark.parametrize("stage", ["failed", "stopped", "stopped_for_other_run", "cancelled"])
def test_unsuccessful_predecessor_never_starts_new_run(stage: str) -> None:
    """A failed or user-stopped predecessor fails closed."""
    with pytest.raises(RuntimeError):
        predecessor_ready({"stage": stage, "running": False}, None, None, 57600)


def test_wrong_horizon_is_rejected() -> None:
    """A mismatched completed run cannot satisfy the dependency."""
    with pytest.raises(RuntimeError):
        predecessor_ready({"stage": "complete", "running": False}, {"offered": 384}, {"passed": True}, 57600)
