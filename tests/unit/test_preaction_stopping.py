"""Causal boundaries, complete-history exclusion and cumulative decision semantics."""

import numpy as np
import pytest

from cafl4ds.preaction_stopping import PreactionStopping, past_losses, select_duration


def example():
    rng = np.random.default_rng(66)
    return rng.normal(size=(6, 2)), rng.normal(size=(6, 15)), rng.normal(size=(6, 4, 2))


def test_held_labels_cannot_change_predictions_or_training_envelopes():
    loss, health, y = example()
    model = PreactionStopping(sham_count=2)
    a = model.crossfit(loss, health, y)
    changed = y.copy()
    changed[0] += 10000
    b = model.crossfit(loss, health, changed)
    for name in a["predictions"]:
        np.testing.assert_array_equal(a["predictions"][name][:4], b["predictions"][name][:4])
        assert not a["operators"][name][:4, :4].any()
    for name in a["envelope_half_width_pp"]:
        np.testing.assert_array_equal(a["envelope_half_width_pp"][name][:4], b["envelope_half_width_pp"][name][:4])
    for fold in a["folds"]:
        outer = fold["held_history"]
        for name, donors in fold["sham_donors"].items():
            assert name.startswith("health_sham")
            assert outer not in donors.values()
            assert all(int(source) != donor for source, donor in donors.items())
        for name in ("health", "controls", "horizon_mean"):
            for inner in fold["models"][name]["inner_folds"]:
                assert outer not in set(inner["train_rows"] // 4)
                assert inner["held_history"] not in set(inner["train_rows"] // 4)


def test_only_past_losses_are_read():
    trace = np.arange(160 * 2, dtype=float).reshape(160, 2)
    expected = past_losses(trace)
    trace[128:] = np.nan
    np.testing.assert_array_equal(expected, past_losses(trace))
    trace[127, 0] = np.nan
    with pytest.raises(ValueError, match="past losses"):
        past_losses(trace)


def test_budget_cannot_reset_or_be_repaired_at_terminal_boundary():
    # Endpoint reward cannot erase a cumulative budget violation at an earlier boundary.
    assert select_duration([[1, 0], [4, -2], [5, 0], [6, 0]]) == 1
    assert select_duration([[0.1, 0.1]] * 4) == 0
    assert select_duration([[1, 0]] * 4) == 1
    assert select_duration([[4, -2]] * 4) == 0


def test_scaling_excludes_extreme_held_features():
    loss, health, y = example()
    from cafl4ds.preaction_stopping import design

    health[0] += 100000
    fit = PreactionStopping(sham_count=1).crossfit(loss, health, y)
    expected = design(loss, health)["health"][4:].mean(0)
    np.testing.assert_allclose(fit["folds"][0]["models"]["health"]["center"], expected)


def test_invalid_profiles_rejected():
    loss, health, y = example()
    with pytest.raises(ValueError, match="Six"):
        PreactionStopping(sham_count=1).crossfit(loss[:5], health[:5], y[:5])
    y[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="targets"):
        PreactionStopping(sham_count=1).crossfit(loss, health, y)
