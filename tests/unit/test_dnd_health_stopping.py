"""Causal stopping, patience and separation of outcomes from control."""

import numpy as np
import pytest

from cafl4ds.dnd_health_stopping import HealthStop, POINTS, curve


def features(seed=8):
    return np.random.default_rng(seed).normal(size=(40, 8))


def test_two_alarms_required_and_quiet_resets():
    bank = features()
    guard = HealthStop()
    guard.observe(132, bank)
    assert not guard.observe(140, features(10), kind="cka")
    assert not guard.observe(148, bank, kind="cka")
    assert not guard.observe(156, features(11), kind="cka")
    assert guard.observe(164, features(12), kind="cka")
    before = list(guard.events)
    assert not guard.observe(172, np.full_like(bank, np.nan), kind="cka")
    assert guard.events == before and guard.stop == 164


def test_future_changes_cannot_change_prefix_choice():
    bank = features()
    stops = []
    for tail in (bank, features(22)):
        guard = HealthStop()
        for step, z in ((132, bank), (140, features(1)), (148, features(2)), (156, tail)):
            guard.observe(step, z, kind="cka")
        stops.append(guard.stop)
    assert stops == [148, 148]


def test_missing_reference_and_fixed_config_rejected():
    with pytest.raises(ValueError, match="reference"):
        HealthStop().observe(140, features())
    with pytest.raises(ValueError, match="Fixed"):
        HealthStop(threshold=0.2)


def test_stop_curve_holds_exact_prefix_at_all_later_clocks():
    full = np.arange(len(POINTS) * 3).reshape(len(POINTS), 3)
    path = curve(full, 264)
    k = POINTS.index(264)
    np.testing.assert_array_equal(path[: k + 1], full[: k + 1])
    np.testing.assert_array_equal(path[k:], np.repeat(full[k : k + 1], len(POINTS) - k, axis=0))


def test_exact_threshold_triggers_without_task_outcomes(monkeypatch):
    monkeypatch.setattr("cafl4ds.dnd_health_stopping.reading", lambda a, b: np.array([0.1, 0, 0]))
    guard = HealthStop()
    bank = features()
    guard.observe(132, bank)
    assert not guard.observe(140, bank)
    assert guard.observe(148, bank)
