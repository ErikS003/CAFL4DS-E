"""Availability, task-retention sensitivity, coordinate invariance and grouped leakage checks."""

import json

import numpy as np
import pytest

from cafl4ds.functional_retention import (
    FunctionalRetention,
    accept,
    bank_folds,
    crossfit,
    invariant_ridge,
    plain,
    prospective_features,
)


def test_rotation_and_isotropic_scale_preserve_functional_health():
    rng = np.random.default_rng(33)
    features = rng.normal(size=(128, 12))
    targets = features[:, :2] / np.linalg.norm(features, axis=1, keepdims=True)
    monitor = FunctionalRetention(targets, np.arange(128) % 2, np.arange(128) % 4)
    original = monitor.score(features)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    for changed in (features @ q, features * 7):
        observed = monitor.score(changed)
        np.testing.assert_allclose(observed["predictions"], original["predictions"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(observed["health"], original["health"], atol=1e-12, rtol=0)


def test_constant_features_and_identity_permutation_lose_reference_function():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(128, 8))
    y = x[:, :2] / np.linalg.norm(x, axis=1, keepdims=True)
    monitor = FunctionalRetention(y, np.arange(128) % 2, np.arange(128) % 4)
    baseline = monitor.score(x)["health"].mean()
    assert monitor.score(np.ones_like(x))["health"].mean() > baseline + 0.5
    assert monitor.score(x[rng.permutation(128)])["health"].mean() > baseline + 0.5


def test_id_fold_assignment_survives_reordering_and_rejects_duplicates():
    names = [f"image_{i}" for i in range(128)]
    original = dict(zip(names, bank_folds(names), strict=True))
    reordered = names[::-1]
    assert original == dict(zip(reordered, bank_folds(reordered), strict=True))
    with pytest.raises(ValueError):
        bank_folds(["same"] * 128)


def test_scales_come_only_from_fitting_rows_even_for_outlying_query():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(20, 7))
    y = rng.normal(size=(20, 2))
    _, baseline = invariant_ridge(x, y, x[:3])
    _, changed = invariant_ridge(x, y, np.full((3, 7), 1e20))
    np.testing.assert_array_equal(changed["center"], baseline["center"])
    np.testing.assert_array_equal(changed["beta"], baseline["beta"])
    assert changed["scale"] == baseline["scale"]


def test_prospective_features_ignore_future_health_and_losses():
    rng = np.random.default_rng(95)
    geometry = rng.uniform(0.1, 1, size=(5, 5))
    functional = rng.uniform(0.1, 1, size=(5, 2))
    losses = list(rng.uniform(0.1, 1, size=32))
    x = prospective_features(geometry, functional, 17, losses)
    geometry[3:], functional[3:], losses[8:] = 999, -999, [-999] * 24
    np.testing.assert_array_equal(prospective_features(geometry, functional, 17, losses), x)
    assert x.shape == (23,)


def test_all_held_source_outcomes_are_excluded_from_fits():
    rng = np.random.default_rng(101)
    data = {
        "x": rng.normal(size=(30, 23)),
        "response": rng.normal(size=(30, 2)),
        "sham_response": rng.normal(size=(30, 2)),
        "damage": rng.normal(size=30),
        "groups": np.repeat(np.arange(6), 5),
    }
    a = crossfit(data)
    data["response"][:5] = 1000
    data["sham_response"][:5] = -1000
    data["damage"][:5] = 5000
    b = crossfit(data)
    for name in ("mean", "geometry", "direct", "chain"):
        np.testing.assert_array_equal(a["forecasts"][name][:5], b["forecasts"][name][:5])
    assert a["folds"]["0"]["utility_models"] == b["folds"]["0"]["utility_models"]
    assert a["folds"]["0"]["risk_model"] == b["folds"]["0"]["risk_model"]
    assert not np.allclose(a["forecasts"]["measured"][:5], b["forecasts"]["measured"][:5])
    np.testing.assert_array_equal(a["predicted_health"][:5], b["predicted_health"][:5])
    assert not set(a["folds"]["0"]["fitting_rows"]).intersection(range(5))


def test_threshold_tie_and_json_boundary_are_explicit():
    np.testing.assert_array_equal(accept([0.99, 1, 1.01]), [True, True, False])
    assert json.loads(
        json.dumps(plain({"gate": np.bool_(False), "count": np.int64(3), "matrix": np.ones((1, 2))}))
    ) == {"gate": False, "count": 3, "matrix": [[1.0, 1.0]]}


def test_constant_reference_targets_fail_instead_of_fabricating_health():
    with pytest.raises(ValueError, match="variance"):
        FunctionalRetention(np.ones((64, 2)), np.arange(64) % 2, np.arange(64) % 4)
