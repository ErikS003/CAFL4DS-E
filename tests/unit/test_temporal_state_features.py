"""Test the causal boundary and distinguish redundancy from temporal order."""

import numpy as np
import pytest

from scripts.temporal_state_features import causal_features


def test_future_arrivals_cannot_change_state_features() -> None:
    """A future corruption or content change must not reach a current feature."""
    frames = np.random.default_rng(17).integers(0, 256, (256, 64, 64, 3), dtype=np.uint8)
    expected = causal_features(frames, 2)
    frames[128:] = 0
    assert causal_features(frames, 2) == expected
    assert causal_features(frames[:128], 2) == expected


def test_spread_and_change_measure_different_content_properties() -> None:
    """Chronological and alternating frames share diversity but differ in change."""
    frames = np.zeros((64, 64, 64, 3), dtype=np.uint8)
    frames[32:] = 255
    alternating = frames[np.ravel(np.column_stack((np.arange(32), np.arange(32, 64))))]
    ordered = causal_features(frames, 1)
    shuffled = causal_features(alternating, 1)
    assert ordered["within_window_rgb_spread"] == shuffled["within_window_rgb_spread"]
    assert shuffled["mean_adjacent_rgb_change"] > 50 * ordered["mean_adjacent_rgb_change"]


def test_partial_or_nonexistent_past_is_rejected() -> None:
    """Do not invent a feature before a FIFO window closes."""
    with pytest.raises(ValueError, match="complete observed window"):
        causal_features(np.zeros((63, 64, 64, 3), dtype=np.uint8), 1)
    with pytest.raises(ValueError, match="complete observed window"):
        causal_features(np.zeros((128, 64, 64, 3), dtype=np.uint8), 0)
