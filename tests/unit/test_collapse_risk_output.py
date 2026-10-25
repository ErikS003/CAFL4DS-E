"""Serialization compatibility without changing frozen scientific values."""

import json

import numpy as np

from scripts.verify_collapse_risk_regression_output import plain


def test_numpy_gate_booleans_and_counts_are_json_safe():
    original = {
        "gates": {"yes": np.bool_(True), "no": np.bool_(False)},
        "count": np.int64(12),
        "values": (np.float64(0.125), [np.float32(0.5)]),
    }
    normalized = plain(original)
    assert json.loads(json.dumps(normalized, allow_nan=False)) == {
        "gates": {"yes": True, "no": False},
        "count": 12,
        "values": [0.125, [0.5]],
    }
    assert isinstance(original["gates"]["yes"], np.bool_)


def test_native_nested_values_are_preserved():
    value = {"passed": False, "low": -1.25, "high": 0.125, "missing": None, "ids": ["cohort", 17]}
    assert plain(value) == value
