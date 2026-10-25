"""Source exclusion and predecision availability for the regression experiment."""

import numpy as np
import pytest

from cafl4ds.health_regression_gate import accept, crossfit, fit, observations, predict, response


def synthetic():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(24, 17))
    r = rng.normal(size=(24, 5))
    y = rng.normal(size=(24, 3))
    groups = np.repeat(np.arange(6), 4)
    return x, r, y, groups


def test_held_future_health_and_utility_cannot_change_prospective_prediction():
    x, r, y, groups = synthetic()
    first = crossfit(x, r, y, groups)
    r[groups == 0] += 1e5
    y[groups == 0] -= 1e5
    second = crossfit(x, r, y, groups)
    for name in ("history", "predicted", "plugin", "mean", "cheap"):
        np.testing.assert_array_equal(
            np.array(first["predictions"][name])[groups == 0], np.array(second["predictions"][name])[groups == 0]
        )
    np.testing.assert_array_equal(
        np.array(first["predicted_health"])[groups == 0], np.array(second["predicted_health"])[groups == 0]
    )


def test_inner_generator_excludes_its_own_history_as_well_as_outer_history():
    x, r, y, groups = synthetic()
    first = crossfit(x, r, y, groups)
    r[groups == 1] += 100
    second = crossfit(x, r, y, groups)
    training_groups = groups[groups != 0]
    a = np.array(first["folds"]["0"]["generated_training_health"])
    b = np.array(second["folds"]["0"]["generated_training_health"])
    np.testing.assert_array_equal(a[training_groups == 1], b[training_groups == 1])
    assert first["folds"]["0"]["inner_health"]["1"]["excluded_groups"] == ["0", "1"]


def test_held_feature_outlier_does_not_enter_fit_scalers():
    x, r, y, groups = synthetic()
    first = crossfit(x, r, y, groups)
    x[groups == 0] += 1e7
    second = crossfit(x, r, y, groups)
    assert first["folds"]["0"]["outer_health"] == second["folds"]["0"]["outer_health"]
    assert first["folds"]["0"]["utility_models"] == second["folds"]["0"]["utility_models"]


def test_observations_reject_future_rows_and_use_only_past_loss():
    h = np.ones((3, 5))
    h[1] *= 2
    h[2] *= 4
    x = observations(h, 8, list(range(8)))
    assert x.shape == (17,)
    assert x[0] == 16 and x[1] == 5.5
    np.testing.assert_array_equal(x[7:10], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(x[10:12], [2, 2])
    with pytest.raises(ValueError, match="Three past"):
        observations(np.ones((4, 5)), 8, list(range(8)))


def test_forward_health_contrast_is_relative_only_for_spread_and_rank():
    np.testing.assert_allclose(response([2, 4, 8, 0.2, 0.4], [3, 2, 8, 0.3, 0.1]), [0.5, -0.5, 0, 0.1, -0.3])


def test_point_policy_boundaries_and_domain_veto():
    choices = accept([[0.25, -1, -1], [0.249, 0, 0], [1, -1.001, 4], [1, 3, -1.001], [0.5, 0, 0]])
    np.testing.assert_array_equal(choices, [True, False, False, False, True])


def test_constant_features_are_finite_and_predict_target_mean():
    model = fit(np.ones((8, 4)), np.arange(24).reshape(8, 3))
    np.testing.assert_allclose(predict(model, np.ones((2, 4))), np.tile(np.arange(24).reshape(8, 3).mean(0), (2, 1)))
    assert np.isfinite(model["scale"]).all()
