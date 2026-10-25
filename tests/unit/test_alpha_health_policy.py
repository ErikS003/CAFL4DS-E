"""Verify reference anchoring, training normalization and nested stream exclusion."""

import numpy as np
import pytest

from cafl4ds.alpha_health_policy import ACTIONS, RECIPES, choose_action, fit, tune


def test_reference_and_train_only_normalization() -> None:
    """Extreme held-out features cannot modify normalization; reference stays zero."""
    rows = [
        {
            "alpha": a,
            "features": {"step": float(i), "history_alpha": 0.5},
            "target": float(i) * (1 - a),
            "stream": f"s{i % 2}",
        }
        for i in range(4)
        for a in (0.0, 0.5)
    ]
    model = fit(rows, "action_history", 1.0)
    expected = np.array([1.5, 0.5])
    np.testing.assert_array_equal(model.mean, expected)
    weight = model.weight.copy()
    heldout = [{"alpha": a, "features": {"step": 1000.0, "history_alpha": 0.1}} for a in ACTIONS]
    assert model.predict(heldout)[0] == 0.0
    np.testing.assert_array_equal(model.mean, expected)
    np.testing.assert_array_equal(model.weight, weight)
    assert choose_action(dict.fromkeys(ACTIONS, 0.0)) == 1.0
    with pytest.raises(ValueError):
        fit([{"alpha": 1.0}], "action_history", 1.0)


def test_nested_tuning_and_target_blind_prediction() -> None:
    """All inner validation units are streams; prediction ignores injected labels."""
    rows = [
        {
            "alpha": a,
            "features": {k: float(i + 1) * (1 - a / 2) for k in RECIPES["combined"]},
            "target": float(i) * (1 - a) * 0.01,
            "stream": f"s{i % 2}",
        }
        for i in range(6)
        for a in (0.0, 0.5)
    ]
    model, cv = tune(rows, "combined")
    assert len(cv) == 10
    assert {r["validation_stream"] for r in cv} == {"s0", "s1"}
    expected = model.predict(rows)
    poisoned = [{**r, "target": 1e8, "stream": "new"} for r in rows]
    np.testing.assert_array_equal(model.predict(poisoned), expected)
