"""Leakage, geometry, and decision-evidence checks for prediction pilots."""

import numpy as np
import pytest

from cafl4ds.prediction_benchmark import health_effect, health_values, held_group_ridge, improvement, scores


def test_held_history_cannot_influence_its_forecast():
    groups = np.repeat(np.arange(6), 4)
    x = np.arange(48).reshape(24, 2).astype(float)
    y = np.sin(x[:, 0])
    first, _ = held_group_ridge(x, y, groups)
    x[:4] += 1000
    y[:4] += 1000
    second, _ = held_group_ridge(x, y, groups)
    # A held-source label change cannot change its predictions; its changed x can.
    x[:4] -= 1000
    third, folds = held_group_ridge(x, y, groups)
    np.testing.assert_array_equal(first[:4], third[:4])
    np.testing.assert_array_equal(folds["0"]["center"], x[4:].mean(0))
    assert not np.array_equal(first[:4], second[:4])


def test_macro_scores_do_not_count_repeats_as_new_sources():
    y = np.array([0.0, 2.0, 2.0])
    groups = np.array([0, 1, 1])
    result = scores(y, {"zero": np.zeros(3)}, groups)
    assert result["zero"]["mse"] == 2


def test_correlation_does_not_imply_calibrated_skill():
    y = np.arange(1, 13) / 10
    result = scores(y, {"zero": np.zeros(12), "bad": 100 * y}, np.repeat(np.arange(6), 2))
    assert result["bad"]["correlation"] == pytest.approx(1)
    assert result["bad"]["skill_vs_zero_percent"] < 0
    assert not improvement(result["bad"], result["zero"], 4)["passes"]


def test_geometry_rotation_and_exact_hold():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(64, 8))
    rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    before = health_values(z, z)
    after = health_values(z @ rotation, z)
    assert after["cka"] == pytest.approx(0, abs=1e-12)
    assert after["cosine"] > 0.1
    assert after["rank"] == pytest.approx(before["rank"])
    assert all(v == 0 for v in health_effect(before, before, before).values())


def test_zero_energy_cannot_qualify():
    result = scores(np.zeros(12), {"zero": np.zeros(12), "candidate": np.zeros(12)}, np.repeat(np.arange(6), 2))
    assert result["candidate"]["skill_vs_zero_percent"] is None
    assert not improvement(result["candidate"], result["zero"], 4)["passes"]
