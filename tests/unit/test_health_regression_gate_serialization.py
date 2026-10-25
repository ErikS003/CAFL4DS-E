"""Only numeric representation changes are allowed in cached-fit finalization."""

import json

import numpy as np
import pytest

from scripts.finalize_health_regression_gate import plain


def test_numpy_counts_booleans_and_predictions_preserve_values():
    original = {"count": np.int64(2), "passed": np.bool_(False), "predictions": np.array([0.125, -0.25])}
    result = json.loads(json.dumps(plain(original), allow_nan=False))
    assert result == {"count": 2, "passed": False, "predictions": [0.125, -0.25]}


def test_nonfinite_numbers_are_not_silently_replaced():
    with pytest.raises(ValueError):
        json.dumps(plain({"value": np.float64(np.nan)}), allow_nan=False)
