"""Budgeted adaptation permits uphill steps without a fresh allowance per image."""

from dataclasses import asdict

import pytest

from cafl4ds.single_image_retention_budget import RetentionBudget


def test_uphill_steps_share_one_allowance() -> None:
    """Individually small increases cannot accumulate beyond the shared ceiling."""
    budget = RetentionBudget(1.0, 1.0, 0.01, 0.01, 32)
    assert budget.passes(1.006, 1.2, 1.3)
    budget.commit(1.006)
    assert budget.passes(1.009, 1.2, 1.3)
    budget.commit(1.009)
    assert not budget.passes(1.015, 1.2, 1.3)
    assert budget.ceiling == 1.01


def test_window_refresh_never_bypasses_original_anchor() -> None:
    """A rising sequence of window anchors still respects the permanent anchor."""
    budget = RetentionBudget(1.0, 1.0, 0.01, 0.01, 2)
    budget.commit(1.005)
    budget.commit(1.009)
    assert budget.window_anchor == 1.009
    assert budget.ceiling == 1.01
    assert not budget.passes(1.015, 1.0, 1.3)


def test_improved_window_tightens_allowance_and_resume_preserves_it() -> None:
    """Improvement tightens the local budget and survives serialization."""
    budget = RetentionBudget(1.0, 1.0, 0.01, 0.01, 2)
    budget.commit(0.8)
    budget.commit(0.75)
    budget.commit(0.754)
    resumed = RetentionBudget(**asdict(budget))
    assert resumed.ceiling == pytest.approx(0.7575)
    assert resumed.accepted_in_window == 1
    assert not resumed.passes(0.76, 1.0, 1.3)
    assert not resumed.passes(0.75, 1.31, 1.3)


def test_rejected_trials_cannot_consume_or_reset_budget() -> None:
    """Only committed updates can advance a window."""
    budget = RetentionBudget(1.0, 1.0, 0.01, 0.01, 2)
    budget.commit(0.9)
    before = asdict(budget)
    for _ in range(10):
        assert not budget.passes(1.02, 1.0, 1.3)
    assert asdict(budget) == before
    with pytest.raises(ValueError):
        budget.commit(1.02)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_failed_measurements_never_admit(value: float) -> None:
    """Numerical failures cannot bypass the measured guard."""
    budget = RetentionBudget(1.0, 1.0, 0.01, 0.01, 2)
    with pytest.raises(ValueError):
        budget.passes(value, 1.0, 1.3)
