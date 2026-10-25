import numpy as np
import pytest

from cafl4ds.preaction_benefit import (
    FIELDS,
    action_values,
    causal_norm,
    design,
    excluded_predictions,
    preaction_features,
    ridge,
    screen,
)


def population():
    g = np.array(
        [
            (seed, source, context)
            for seed in (1, 2, 3)
            for source in (1, 2, 3)
            for context in (1, 2, 3, 4)
            for _ in range(2)
        ]
    )
    d = np.tile([0, 1], 36)
    x = np.random.default_rng(3).normal(size=(72, 16))
    return x, d, g


def test_features_and_norm_use_only_observed_prefix():
    readings = {b: {"change": {f: float(i) for i, f in enumerate(FIELDS)}} for b in ("monitor", "audit")}
    a = np.tile([1.0, 2.0, 3.0], (128, 1))
    a[-16:, 2] = 4
    x = preaction_features(readings, a)
    assert x.shape == (16,) and causal_norm(a) == 4
    np.testing.assert_array_equal(x[-3:], [1, 2, 4])
    with pytest.raises(ValueError):
        causal_norm(np.tile([1.0, 2.0, 3.0], (512, 1)))


def test_both_diets_and_all_supports_are_held_together_with_triple_exclusion():
    x, d, g = population()
    y = x[:, 3]
    pred, folds = excluded_predictions(x, d, g, y)
    assert len(folds) == 36 and all(len(f["train"]) == 24 and len(f["test"]) == 2 for f in folds)
    for f in folds:
        for k in range(3):
            assert not set(g[f["train"], k]) & set(g[f["test"], k])
    # Changing test outcomes cannot change predictions for that test fold.
    y2 = y.copy()
    y2[folds[0]["test"]] += 10000
    pred2, _ = excluded_predictions(x, d, g, y2)
    for m in pred:
        np.testing.assert_array_equal(pred[m][folds[0]["test"]], pred2[m][folds[0]["test"]])


def test_train_only_scaling_and_ridge_matches_augmented_least_squares():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 9))
    y = rng.normal(size=24)
    z = rng.normal(size=(2, 9))
    a = (x - x.mean(0)) / x.std(0)
    b = (z - x.mean(0)) / x.std(0)
    coef = np.linalg.lstsq(np.r_[a, np.sqrt(10) * np.eye(9)], np.r_[y - y.mean(), np.zeros(9)], rcond=None)[0]
    np.testing.assert_allclose(ridge(x, y, z), y.mean() + b @ coef, atol=1e-12)


def test_fixed_sensor_and_policy_values_use_paired_benefits():
    x, d, _ = population()
    z = design(x, d, "health_added")
    np.testing.assert_array_equal(z[:, -2], x[:, [3, 8]].mean(1))
    np.testing.assert_array_equal(z[:, -1], x[:, [3, 8]].mean(1) * d)
    np.testing.assert_array_equal(action_values(np.array([-2, 3, 4]), np.array([1, -1, 0])), [-2, 0, 0])


def test_forecast_screen_requires_replication_and_does_not_qualify_controller():
    _, _, g = population()
    y = np.arange(72.0)
    p = {m: y.copy() for m in ("mean", "diet", "cheap", "health_only", "health_added")}
    p["cheap"] = y + 2
    result = screen(y, p, g)
    assert result["forecast_screen_passed"] and result["no_live_controller_qualified"]
    p["health_added"][g[:, 0] == 3] = y[g[:, 0] == 3] + 3
    assert not screen(y, p, g)["forecast_screen_passed"]
