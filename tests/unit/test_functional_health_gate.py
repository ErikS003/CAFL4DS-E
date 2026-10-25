"""Health-budget and unreadable-monitor decisions for the research filter prototype."""

import numpy as np
import pytest

from cafl4ds.functional_health_gate import decisions


def test_any_domain_can_exhaust_budget_and_ties_are_accepted():
    np.testing.assert_array_equal(
        decisions([[0.05, -0.2], [0.051, -0.3], [-0.1, 0.051], [-0.1, -0.2]]), [True, False, False, True]
    )


def test_unreadable_monitor_falls_back_to_training_without_claiming_safety():
    np.testing.assert_array_equal(
        decisions([[0.2, 0.2], [0.2, 0.2], [-0.1, -0.1]], [0.249, 0.25, 0.1]), [True, False, True]
    )


def test_invalid_health_or_readability_shape_is_rejected():
    with pytest.raises(ValueError):
        decisions([[np.nan, 0.1]])
    with pytest.raises(ValueError):
        decisions([[0.1, 0.2]], [1, 2])
    with pytest.raises(ValueError):
        decisions([[0.1, 0.2]], [np.nan])
