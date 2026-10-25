"""Dependence recovery, boundary handling and temporal/source leakage checks."""

import copy

import numpy as np
import pytest

from cafl4ds.health_timeseries import acf, aggregate, ccf, feature, forecast


def test_acf_recovers_stationary_ar1_and_difference_removes_random_walk_dependence() -> None:
    """Known synthetic processes distinguish persistent levels from predictable changes."""
    rng = np.random.default_rng(301)
    noise = rng.normal(size=20000)
    x = np.zeros(len(noise))
    for t in range(1, len(x)):
        x[t] = 0.7 * x[t - 1] + noise[t]
    assert acf(x)[1] == pytest.approx(0.7, abs=0.02)
    walk = np.cumsum(noise)
    assert acf(walk)[1] > 0.99
    assert abs(acf(np.diff(walk))[1]) < 0.03


def test_trajectory_boundaries_and_offsets_are_not_concatenated() -> None:
    """Offsets between independent trajectories cannot manufacture autocorrelation."""
    rng = np.random.default_rng(307)
    a, b = rng.normal(size=(2, 33))
    first = aggregate([acf(a), acf(b)], ["one", "two"])
    second = aggregate([acf(a + 100), acf(b - 100)], ["one", "two"])
    np.testing.assert_allclose(first["mean"], second["mean"], atol=1e-13)


def test_cross_correlation_positive_lag_means_input_leads() -> None:
    """A known delayed response must peak at its positive lag."""
    rng = np.random.default_rng(311)
    x = rng.normal(size=20000)
    y = np.r_[0.0, x[:-1]]
    values = [ccf(x, y, lag) for lag in range(-4, 5)]
    assert np.argmax(values) - 4 == 1
    assert ccf(x, y, 1) == pytest.approx(1)


def data() -> list[dict]:
    """Create a fixed grouped small trajectory panel for leakage tests."""
    rng = np.random.default_rng(313)
    return [
        {
            "stream": str(g),
            "checkpoint": "p",
            "health": np.cumsum(rng.normal(size=(33, 5)), axis=0).tolist(),
            "loss": rng.normal(size=32).tolist(),
            "gradient_norm": rng.uniform(size=32).tolist(),
            "parent_updates": 10,
        }
        for g in range(6)
    ]


def test_features_do_not_read_future_health_loss_or_gradient() -> None:
    """Changing observations strictly after an origin cannot alter any model input."""
    s = data()[0]
    changed = copy.deepcopy(s)
    changed["health"][13:] = (np.array(changed["health"])[13:] + 100).tolist()
    changed["loss"][12:] = [999.0] * 20
    changed["gradient_norm"][12:] = [999.0] * 20
    for name in ("ar1", "ar2", "delta_ar2", "varx"):
        np.testing.assert_array_equal(feature(s, 0, 12, name), feature(changed, 0, 12, name))


def test_whole_history_and_time_cutoff_keep_future_labels_out_of_fit() -> None:
    """Held targets and any series' late observations cannot alter held-fold coefficients."""
    original = data()
    first = forecast(original, 0, 4)
    altered = copy.deepcopy(original)
    for s in altered:
        s["health"][31:] = (np.array(s["health"])[31:] + 1000).tolist()
    second = forecast(altered, 0, 4)
    for model in ("ar1", "ar2", "delta_ar2", "varx"):
        assert first["folds"][model] == second["folds"][model]
        np.testing.assert_array_equal(first["models"][model]["predictions"], second["models"][model]["predictions"])
    assert max(r["target_update"] for r in first["training_rows"]) == 20
    assert min(r["origin"] for r in first["test_rows"]) == 20
    altered = copy.deepcopy(original)
    altered[0]["health"] = (np.array(altered[0]["health"]) * 30).tolist()
    third = forecast(altered, 0, 4)
    for model in ("ar1", "ar2", "delta_ar2", "varx"):
        assert first["folds"][model]["0"] == third["folds"][model]["0"]
