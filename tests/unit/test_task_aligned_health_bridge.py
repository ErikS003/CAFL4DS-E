"""Leakage, equivariance and target-law checks for the measured bridge."""

import numpy as np

from cafl4ds.task_aligned_health_bridge import HealthCapabilityBridge, controls, crossed_damage


def fixture():
    rng = np.random.default_rng(13)
    groups = np.repeat(np.arange(6), 3)
    x = rng.normal(size=(18, 6))
    health = {k: rng.normal(size=(18, 2, 2)) for k in ("dino", "mae", "dino_sham", "mae_sham")}
    y = rng.normal(size=(18, 2))
    return x, health, y, groups


def test_held_source_cannot_change_its_predictions():
    x, health, y, groups = fixture()
    study = HealthCapabilityBridge()
    original = study.crossfit(x, health, y, groups)
    changed = y.copy()
    changed[groups == 2] += 1000
    refit = study.crossfit(x, health, changed, groups)
    for name in original["predictions"]:
        np.testing.assert_allclose(original["predictions"][name][groups == 2], refit["predictions"][name][groups == 2])
        assert np.all(original["operators"][name][np.ix_(groups == 2, groups == 2)] == 0)


def test_causal_controls_ignore_future_losses():
    trace = np.arange(600).reshape(300, 2).astype(float)
    baseline = controls("day_end", 128, trace)
    trace[128:] = -10000
    np.testing.assert_array_equal(baseline, controls("day_end", 128, trace))
    assert baseline[-1] == 2


def test_ridge_is_equivariant_to_target_units():
    x, health, y, groups = fixture()
    study = HealthCapabilityBridge()
    a, b = study.crossfit(x, health, y, groups), study.crossfit(x, health, y * 7 + 3, groups)
    for name in a["predictions"]:
        np.testing.assert_allclose(b["predictions"][name], a["predictions"][name] * 7 + 3, atol=1e-12)


def test_crossed_target_sign_and_class_balance():
    labels = np.array([0, 0, 1, 0, 1, 1])
    domains = np.array([0, 0, 0, 1, 1, 1])
    before = np.eye(2)[labels][None].repeat(2, axis=0)
    after = before.copy()
    after[:, :2] = before[:, :2, ::-1]
    actual = crossed_damage(before, after, labels, domains, np.array([[1, 1], [2, 0]]), np.ones((2, 6), int))
    np.testing.assert_allclose(actual, [[50, 0], [50, 0]])


def test_source_permutation_preserves_predictions():
    x, health, y, groups = fixture()
    study = HealthCapabilityBridge()
    order = np.random.default_rng(34).permutation(18)
    a = study.crossfit(x, health, y, groups)
    b = study.crossfit(x[order], {k: v[order] for k, v in health.items()}, y[order], groups[order])
    for name in a["predictions"]:
        np.testing.assert_allclose(a["predictions"][name][order], b["predictions"][name], atol=1e-12)
