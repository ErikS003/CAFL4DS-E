"""Check isolation, feature exclusions and causal direct-effect forecasting."""

import numpy as np
import pytest

from cafl4ds.single_image_forecast import HEALTH
from cafl4ds.single_image_impact import designs, forecasts, isolated_effect
from scripts.verify_single_image_impact_analysis import independent_fit


def fixture():
    rng = np.random.default_rng(63)
    origins = np.tile(np.arange(8, 49, 8), 4)
    held = np.repeat([False, False, True, True], 6)
    histories = np.repeat(["a", "b", "c", "d"], 6)
    x = {"image": rng.normal(size=(24, 5))}
    y = rng.normal(size=(24, 3))
    settings = {"prefix_images": 24, "test_origin_start": 32, "ridge_penalty": 10.0}
    return x, y, origins, held, histories, settings


def test_isolation_cancels_common_optimizer_motion_and_preserves_units():
    common = np.array([[12.0, 0.7], [-8.0, 0.3]])
    increment = np.array([[0.25, -0.002], [-0.5, 0.004]])
    np.testing.assert_allclose(isolated_effect(common + increment, common), increment, atol=1e-15)
    np.testing.assert_array_equal(isolated_effect(common, common), np.zeros_like(common))
    with pytest.raises(ValueError):
        isolated_effect(common, np.array([np.nan]))


def test_label_free_inputs_cannot_observe_any_downstream_readout():
    metrics = list(HEALTH) + ["finetune100_accuracy_balanced", "linear_ce_day"]
    rng = np.random.default_rng(2)
    before, lag = rng.normal(size=(2, 4, len(metrics)))
    args = [rng.normal(size=(4, n)) for n in (6, 19, 6)]
    initial = designs(before, lag, *args, np.arange(4), metrics)
    before[:, 7:] += 999
    lag[:, 7:] -= 123
    changed = designs(before, lag, *args, np.arange(4), metrics)
    for key in initial:
        if key.startswith("label_free"):
            np.testing.assert_array_equal(initial[key], changed[key])
        else:
            assert not np.array_equal(initial[key], changed[key])


def test_unmatured_effects_do_not_change_current_predictions_or_controls():
    x, y, origins, held, histories, settings = fixture()
    initial, _, test = forecasts(x, y, origins, held, histories, settings)
    y[origins + 1 > 40] += 100
    changed, _, _ = forecasts(x, y, origins, held, histories, settings)
    for key in initial:
        np.testing.assert_array_equal(initial[key][origins[test] <= 40], changed[key][origins[test] <= 40])


def test_transfer_excludes_held_effects_and_previous_control_is_stream_local():
    x, y, origins, held, histories, settings = fixture()
    initial, _, test = forecasts(x, y, origins, held, histories, settings)
    y[held] += 50
    changed, _, _ = forecasts(x, y, origins, held, histories, settings)
    np.testing.assert_array_equal(initial["transfer_image"], changed["transfer_image"])
    for j, i in enumerate(np.flatnonzero(test)):
        previous = np.flatnonzero((histories == histories[i]) & (origins + 1 <= origins[i]))
        np.testing.assert_array_equal(changed["previous_effect"][j], y[previous[-1]])


def test_direct_predictions_match_independent_mature_label_least_squares():
    x, y, origins, held, histories, settings = fixture()
    predictions, _, test = forecasts(x, y, origins, held, histories, settings)
    calibration = origins + 1 <= 24
    for mode, fit in (("transfer", calibration & ~held), ("frozen", calibration)):
        expected = independent_fit(x["image"], y, fit, fit, test, penalty=10.0)
        np.testing.assert_allclose(predictions[mode + "_image"], expected, rtol=1e-7, atol=1e-10)
    expected = np.zeros_like(y)
    for t in sorted(set(origins[test])):
        fit = calibration | (held & (origins + 1 <= t))
        query = test & (origins == t)
        expected[query] = independent_fit(x["image"], y, calibration, fit, query, penalty=10.0)
    np.testing.assert_allclose(predictions["online_image"], expected[test], rtol=1e-7, atol=1e-10)
