"""Protect the focused readout's evaluation weights and rejected-arrival semantics."""

import numpy as np

from scripts.natural_health_view import accuracy_panel, trailing_updates


def test_accuracy_panel_weights_classes_then_domains():
    """Population imbalance must not replace the original class/domain estimand."""
    arrays = {
        "labels": np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        "domains": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "query": np.arange(8),
    }
    prediction = np.array([[0, 0, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(accuracy_panel(prediction, arrays), [[75, 50, 100]])


def test_smoothing_retains_rejection_gaps_and_counts_only_accepted_updates():
    """Skipping is neither zero loss nor another smoothing observation."""
    trace = [
        {"accepted": True, "response": [2.0, 4.0]},
        {"accepted": False, "response": None},
        {"accepted": True, "response": [6.0, 8.0]},
        {"accepted": True, "response": [10.0, 12.0]},
    ]
    result = trailing_updates(trace, 0, window=2)
    assert np.isnan(result[1])
    np.testing.assert_array_equal(result[[0, 2, 3]], [2, 4, 8])
