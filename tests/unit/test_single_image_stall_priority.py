"""J-positive covariance stalls must receive the same sparse low-rate pulses."""

from dataclasses import asdict

import pytest

from cafl4ds.single_image_stall_priority import PriorityStallController


@pytest.mark.parametrize("j_admitted", [False, True])
def test_due_pulse_overrides_either_J_outcome(j_admitted: bool) -> None:
    """The correction covers exactly the covariance-blocked J-admitted case."""
    controller = PriorityStallController()
    assert controller.factor(254, j_admitted) == (1.0 if j_admitted else 0.0)
    assert controller.factor(255, j_admitted) == 1 / 16
    controller.finish(255, exploratory=True, accepted=True)
    assert controller.factor(510, j_admitted) == (1.0 if j_admitted else 0.0)
    assert controller.factor(511, j_admitted) == 1 / 16


def test_covariance_failures_cannot_starve_J_positive_pulses() -> None:
    """Repeated normal-rate failures and cooldown eventually yield one small pulse."""
    controller = PriorityStallController()
    trials = []
    for arrival in range(600):
        factor = controller.factor(arrival, True)
        if factor:
            trials.append((arrival, factor))
            controller.finish(arrival, exploratory=factor < 1, accepted=False)
    assert any(factor == 1 / 16 for _, factor in trials)
    assert trials[:4] == [(0, 1.0), (65, 1.0), (130, 1.0), (195, 1.0)]
    assert trials[4] == (260, 1 / 16)
    assert all(b[0] - a[0] >= 65 for a, b in zip(trials, trials[1:], strict=False))


def test_failed_pulse_spacing_and_cooldown_survive_resume() -> None:
    """Cooldown wins over pulse eligibility without losing its saved counters."""
    controller = PriorityStallController()
    controller.finish(255, exploratory=True, accepted=False)
    resumed = PriorityStallController(**asdict(controller))
    assert all(resumed.factor(i, True) == 0 for i in range(256, 320))
    assert resumed.factor(320, True) == 1
    assert resumed.factor(511, True) == 1 / 16


def test_ordinary_commit_restarts_inactivity() -> None:
    """A genuine ordinary update postpones exploration without changing its rate."""
    controller = PriorityStallController()
    controller.finish(50, exploratory=False, accepted=True)
    assert controller.factor(305, True) == 1
    assert controller.factor(306, True) == 1 / 16
