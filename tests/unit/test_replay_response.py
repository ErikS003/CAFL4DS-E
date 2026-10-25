"""Reference anchoring and nighttime guards for replay-allocation proposals."""

import numpy as np

from cafl4ds.replay_response import choose, fit, tune


def test_reference_zero_and_label_blind_predictions() -> None:
    """Changing a prediction row's label cannot alter its feature-only forecast."""
    rows = [
        {
            "stream": stream,
            "action": action,
            "features": {"step": float(step), "history_replay_fraction": 0.5},
            "target": (action - 16) * 0.0001 * step,
        }
        for stream in ("s1", "s2")
        for step in (4, 12)
        for action in (8, 24)
    ]
    model, cv = tune(rows, "action_history")
    assert {r["validation_stream"] for r in cv} == {"s1", "s2"}
    reference = {"action": 16, "features": {"step": 10000.0, "history_replay_fraction": -300.0}}
    assert model.predict([reference]).item() == 0.0
    probe = {**rows[0], "target": 1e9}
    np.testing.assert_array_equal(model.predict([probe]), model.predict([rows[0]]))
    np.testing.assert_allclose(model.mean, [8.0, 0.5])
    separate = fit([r for r in rows if r["stream"] == "s1"], "action_history", 1.0)
    np.testing.assert_allclose(separate.mean, [8.0, 0.5])


def test_night_guard_keeps_reference_if_only_positive_action_harms_night() -> None:
    """A positive accuracy proposal is insufficient when its nighttime forecast is negative."""
    predictions = {16: 0.0, 8: 0.002, 24: -0.001}
    assert choose(predictions) == 8
    assert choose(predictions, {16: 0.0, 8: -0.001, 24: 0.002}) == 16
    assert choose({16: 0.0, 8: 0.0, 24: 0.0}) == 16
