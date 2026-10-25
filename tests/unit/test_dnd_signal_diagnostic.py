"""Causality, censoring, grouping and cost controls for cached warning forecasts."""

import numpy as np
import pytest

from cafl4ds.dnd_signal_diagnostic import (
    crossfit,
    features_and_alarms,
    horizon_steps,
    matched_age_selection,
    outcome_targets,
    warning_statistics,
)


def test_future_embeddings_cannot_change_an_earlier_feature_or_alarm() -> None:
    """Future geometry cannot leak into an earlier decision input."""
    monitor = (132, 140, 148, 156)
    rng = np.random.default_rng(5)
    banks = {point: rng.normal(size=(30, 5)) for point in monitor}
    before, alarms = features_and_alarms(banks, monitor)
    banks[156] = np.ones((30, 5))
    after, changed_alarms = features_and_alarms(banks, monitor)
    np.testing.assert_array_equal(before[:3], after[:3])
    np.testing.assert_array_equal(alarms[:3], changed_alarms[:3])


def test_horizons_are_complete_and_the_future_running_peak_is_not_protected() -> None:
    """Censor late windows and distinguish current protection from lost future gains."""
    monitor = (132, 140, 148, 156, 164)
    points = (*monitor, 168)
    np.testing.assert_array_equal(horizon_steps(monitor, points, 16), [140, 148])
    curve = np.repeat([[50.0, 50.0, 50.0]], len(points), axis=0)
    curve[2] = 55  # A future gain followed by decline to51 is not loss of present50.
    curve[3] = 51
    curve[4:] = -100  # Changes after t+H must not enter the target.
    targets = outcome_targets(curve, points, np.array([140]), 16)
    np.testing.assert_array_equal(targets, [[0, 1, 1, 1]])


def test_heldout_outcomes_cannot_change_that_historys_forecasts() -> None:
    """An excluded history supplies features, never training targets."""
    rng = np.random.default_rng(21)
    features = rng.normal(size=(8, 12, 6))
    targets = rng.normal(size=(8, 12, 4))
    before, folds = crossfit(features, targets, 1)
    targets[3] = 1000
    after, _ = crossfit(features, targets, 1)
    np.testing.assert_array_equal(before[:, 3], after[:, 3])
    assert all(3 not in fold["train"] for fold in folds if fold["heldout"] == 3)


def test_clock_only_features_cannot_improve_over_the_age_baseline() -> None:
    """Deterministic age features cannot masquerade as incremental health value."""
    rng = np.random.default_rng(32)
    clock = np.arange(10)[:, None] * np.ones((1, 6))
    features = np.repeat(clock[None], 8, axis=0)
    targets = rng.normal(size=(8, 10, 4))
    predictions, _ = crossfit(features, targets, 1)
    for index in range(2, 6):
        np.testing.assert_array_equal(predictions[index], predictions[1])


def test_a_predictive_health_signal_can_transfer_to_an_excluded_history() -> None:
    """A real synthetic health association survives history exclusion."""
    rng = np.random.default_rng(67)
    features = rng.normal(size=(8, 20, 6))
    targets = np.stack([3 + features[..., 0], features[..., 0], -features[..., 0], features[..., 0]], axis=-1)
    predictions, _ = crossfit(features, targets, 1)
    errors = ((predictions - targets[None]) ** 2).mean((1, 2))
    assert errors[2, 0] < errors[1, 0] / 2


def test_matched_controls_preserve_per_history_warning_counts_and_earlier_ties() -> None:
    """Budget matching and deterministic ties must not consult held-out outcomes."""
    warnings = np.array([[True, False, True], [False, False, False], [True, True, True]])
    risks = np.array([[0.5, 0.5, 0.2], [1, 2, 3], [2, 3, 1]])
    selected = matched_age_selection(risks, warnings, np.array([140, 148, 156]))
    np.testing.assert_array_equal(selected.sum(1), warnings.sum(1))
    np.testing.assert_array_equal(selected[0], [True, True, False])


def test_warning_costs_do_not_cancel_positive_learning_with_negative_learning() -> None:
    """Report beneficial learning forgone even when signed gains average to zero."""
    targets = np.array([[[2, 2, 3, 1], [2, -2, -3, -1]]], dtype=float)
    stats = warning_statistics(targets, np.ones((1, 2)), 1, 0.5, 1)
    assert stats["mean_targets_when_warned_pp"][1] == 0
    assert stats["mean_positive_balanced_gain_when_warned_pp"] == 1
    assert stats["material_learning_fraction"] == stats["useful_hold_precision"] == 0.5


def test_combined_persistence_can_switch_instruments_between_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original OR alarm needs consecutive alarms, not the same instrument twice."""

    def fake_reading(_reference: np.ndarray, current: np.ndarray) -> np.ndarray:
        return current[0]

    monkeypatch.setattr("cafl4ds.dnd_signal_diagnostic.reading", fake_reading)
    banks = {
        132: np.array([[0, 0, 0]]),
        140: np.array([[0.2, 0, 0]]),
        148: np.array([[0, 0.2, 0]]),
        156: np.array([[0, 0.3, 0]]),
    }
    _, alarms = features_and_alarms(banks, tuple(banks))
    assert not alarms[1].any()
    assert alarms[2, 3] and not alarms[2, :3].any()
    assert alarms[3, 1] and alarms[3, 3]
