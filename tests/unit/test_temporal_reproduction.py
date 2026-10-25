"""Prevent live-tuples versus JSON-lists from rejecting exact optimizer observations."""

import pytest

from cafl4ds.temporal_reproduction import same_losses


def test_tuple_list_representation_is_ignored() -> None:
    """A persisted optimizer observation has identical numeric meaning."""
    assert same_losses([[(0.6824762225151062, 0.45064207911491394)]], [[[0.6824762225151062, 0.45064207911491394]]])


def test_small_numeric_difference_is_not_ignored() -> None:
    """Canonicalization does not relax exact numerical reproduction."""
    assert not same_losses([(0.2, 0.3)], [[0.2, 0.300000000001]])


def test_missing_loss_pair_is_rejected() -> None:
    """Equal prefixes cannot conceal missing optimizer work."""
    assert not same_losses([(0.2, 0.3)], [])


def test_nonfinite_loss_is_rejected() -> None:
    """Invalid optimizer observations cannot survive JSON normalization."""
    with pytest.raises(ValueError):
        same_losses([(float("nan"), 0.3)], [[float("nan"), 0.3]])
