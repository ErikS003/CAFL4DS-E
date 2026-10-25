"""Sparse exploration must obey inactivity, rate, cooldown and covariance rules."""

from dataclasses import asdict

import numpy as np
import pytest

from cafl4ds.prediction_benchmark import health_values
from cafl4ds.single_image_stall_exploration import StallController, covariance_only


def test_first_exploration_and_failed_attempt_spacing() -> None:
    """No commitment is invented after a failed pulse; retries remain sparse."""
    controller = StallController()
    assert all(controller.factor(i, False) == 0 for i in range(255))
    assert controller.factor(255, False) == 1 / 16
    controller.finish(255, exploratory=True, accepted=False)
    assert controller.last_committed == -1
    assert all(controller.factor(i, False) == 0 for i in range(256, 511))
    assert controller.factor(511, False) == 1 / 16


def test_normal_and_exploratory_commit_restart_inactivity() -> None:
    """Both types of genuine update postpone the next exploration pulse."""
    controller = StallController()
    assert controller.factor(8, True) == 1
    controller.finish(8, exploratory=False, accepted=True)
    assert controller.factor(263, False) == 0
    assert controller.factor(264, False) == 1 / 16
    controller.finish(264, exploratory=True, accepted=True)
    assert controller.factor(519, False) == 0
    assert controller.factor(520, False) == 1 / 16


def test_covariance_failure_blocks_exactly_next_64_arrivals() -> None:
    """Even normal J admissions cannot trigger per-frame covariance retries."""
    controller = StallController()
    controller.finish(10, exploratory=False, accepted=False)
    assert all(controller.factor(i, True) == 0 for i in range(11, 75))
    assert controller.factor(75, True) == 1
    resumed = StallController(**asdict(controller))
    assert asdict(resumed) == asdict(controller)


def test_persistent_stall_has_at_most_225_exploration_attempts() -> None:
    """An arbitrary mixture of successful and failed pulses respects its work bound."""
    controller = StallController()
    attempts = 0
    for i in range(57600):
        if controller.factor(i, False):
            attempts += 1
            controller.finish(i, exploratory=True, accepted=bool(attempts % 2))
    assert attempts == 225


def test_covariance_guard_matches_existing_full_assay_statistic() -> None:
    """The fast guard must retain the exact earlier definition and normalization."""
    z = np.random.default_rng(13).normal(size=(256, 96))
    assert covariance_only(z) == health_values(z, z)["covariance"]
    with pytest.raises(ValueError):
        covariance_only(np.full((256, 96), np.nan))


@pytest.mark.parametrize("factor", [0, 1, -0.5, float("nan")])
def test_invalid_exploratory_rates_are_rejected(factor: float) -> None:
    """The exploratory branch cannot silently become a full-rate step."""
    with pytest.raises(ValueError):
        StallController(exploration_factor=factor)
