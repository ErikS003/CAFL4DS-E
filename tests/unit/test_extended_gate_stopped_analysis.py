"""Stopped-run inference must handle unequal seed coverage without changing its estimand."""

import numpy as np
import pytest

from scripts.finalize_single_image_extended_gate_stop import cluster_interval


def test_unequal_history_sizes_keep_equal_pair_weighting():
    result = cluster_interval(
        np.array([1.0, 3.0, 9.0]), np.array(["a", "a", "b"]), np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    )
    assert result["mean"] == pytest.approx(13 / 3)
    assert result["ci95"] == pytest.approx([2.175, 8.65])


def test_constant_vector_curves_have_constant_bounds():
    values = np.tile([2.0, 5.0], (5, 1))
    draws = np.random.default_rng(6).integers(0, 3, (100, 3))
    result = cluster_interval(values, np.array(["a", "a", "b", "b", "c"]), draws)
    np.testing.assert_array_equal(result["mean"], [2, 5])
    np.testing.assert_array_equal(result["ci95"], [[2, 5], [2, 5]])
