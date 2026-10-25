"""Positive nominal forecasts can be rescued only by measured, covariance-safe gains."""

import pytest

from cafl4ds.single_image_j_rescue import retry_factor, trial_passes


def test_smaller_step_can_fix_overshoot_but_not_a_harmful_direction() -> None:
    """An actual convex loss exposes the distinction between overshoot and bad direction."""
    before = (1.0 - 0.0) ** 2
    nominal = (1.0 - 3.0) ** 2 - before
    smaller = (1.0 - 0.25 * 3.0) ** 2 - before
    if nominal <= 0 or smaller >= 0 or not trial_passes(True, smaller, 1.2, 1.3, 1e-5):
        raise AssertionError("Failed a real smaller-step objective rescue")
    harmful = (1.0 + 0.25 * 3.0) ** 2 - before
    if trial_passes(True, harmful, 1.2, 1.3, 1e-5):
        raise AssertionError("A smaller harmful step was accepted")


@pytest.mark.parametrize(
    "delta,covariance,expected",
    [
        (-0.01, 1.2, True),
        (-0.01, 1.4, False),
        (0.01, 1.2, False),
        (0.0, 1.2, False),
        (-1e-7, 1.2, False),
        (-1e-5, 1.2, False),
    ],
)
def test_rescue_requires_both_guards(delta: float, covariance: float, expected: bool) -> None:
    """Covariance safety alone and negligible J changes do not qualify a rescue."""
    if trial_passes(True, delta, covariance, 1.3, 1e-5) != expected:
        raise AssertionError("Wrong rescue guard decision")


def test_ordinary_path_preserves_n78_decision_and_covariance_retry() -> None:
    """J-admitted images retain the N78 rule; only the rejected path changes."""
    if not trial_passes(False, 0.1, 1.2, 1.3, 1e-5):
        raise AssertionError("Ordinary N78 path changed")
    if trial_passes(False, -0.1, 1.4, 1.3, 1e-5):
        raise AssertionError("Hard covariance cap bypassed")
    if retry_factor(0.25, True, 1.1, 1.2, 1.3, 0.8, 0.25) != 0.0625:
        raise AssertionError("J rejection did not shrink LR")
    if retry_factor(0.25, True, 1.299, 1.4, 1.3, 0.8, 0.25) >= 0.0625:
        raise AssertionError("Covariance headroom did not constrain rescue retry")


def test_nonfinite_trial_is_not_accepted() -> None:
    """A failed numerical assay cannot admit a candidate."""
    with pytest.raises(ValueError):
        trial_passes(True, float("nan"), 1.2, 1.3, 1e-5)
