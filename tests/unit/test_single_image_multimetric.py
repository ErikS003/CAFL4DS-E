"""Three-way scoring, causal fitting and view dependence for metric forecasts."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from cafl4ds.single_image_multimetric import class_score, counts, forecast, intervals, regression_score


def settings():
    return OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parents[2] / "cafl4ds/configs/single_image_multimetric.yaml")
    )


def test_three_class_score_matches_explicit_view_losses():
    delta = np.array([[-1, 0, 1], [0, 0, 1]], float)
    p = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    s = class_score(counts(delta, 1e-10), p)
    labels = np.where(delta < 0, 0, np.where(delta == 0, 1, 2)).ravel()
    prediction = np.repeat(p, 3, axis=0)
    expected = np.eye(3)[labels]
    np.testing.assert_allclose(s["brier"], ((prediction - expected) ** 2).sum(1).mean() / 2)
    np.testing.assert_allclose(s["log_loss"], -np.log(prediction[np.arange(6), labels]).mean())
    assert s["tie_fraction"] == 0.5 and s["changed_views"] == 3


def test_tie_accuracy_does_not_masquerade_as_direction_prediction():
    delta = np.r_[np.zeros(98), -1, 1][:, None]
    p = np.tile([0.0, 1.0, 0.0], (100, 1))
    s = class_score(counts(delta, 1e-10), p)
    assert s["accuracy"] == 0.98
    assert s["direction_balanced_accuracy"] == 0.5
    assert s["changed_fraction"] == 0.02


def test_singleton_and_repeated_view_magnitude_errors_differ():
    s = regression_score(np.array([[-1, 1], [2, 2]]), np.array([0, 2]))
    assert s["mean_rmse"] == 0
    np.testing.assert_allclose(s["rmse"], np.sqrt(0.5))
    assert s["mae"] == 0.5


def test_query_and_late_excluded_labels_cannot_change_frozen_forecast():
    rng = np.random.default_rng(42)
    origins = np.tile(np.arange(128, 1089, 32), 2)
    x = rng.normal(size=(len(origins), 3))
    y = rng.normal(size=(len(origins), 1))
    pool = np.repeat([True, False], 31)
    test = ~pool & (origins >= 544)
    lag = rng.normal(size=len(origins))
    groups = np.zeros(len(origins), int)
    config = settings()
    for name in ("classifiers", "regressors"):
        config[name]["nonlinear"]["max_iter"] = 3
    original, meta = forecast({"state": x}, y, lag, origins, groups, pool, test, config)
    altered = y.copy()
    altered[~(pool & (origins + 1 <= 512))] += 100
    changed, new_meta = forecast({"state": x}, altered, lag, origins, groups, pool, test, config)
    assert meta == new_meta
    for kind in original:
        for name in original[kind]:
            np.testing.assert_array_equal(original[kind][name], changed[kind][name])


def test_interval_keeps_all_views_with_their_source_history():
    config = settings()
    config["bootstrap_draws"] = 100
    delta = np.array([[1, 2], [-1, -2], [3, 4], [-3, -4]])
    groups = np.array(["a", "a", "b", "b"])
    a, b = delta.mean(1), np.zeros(4)
    result = intervals(delta, a, b, 0.5, 0.5, groups, config, "regression")
    assert result["mse"][1] < 0 and result["mae"][1] < 0
