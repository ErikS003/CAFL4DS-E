"""Validate the independent batch-one regression audit on a singular design."""

import numpy as np

from cafl4ds.single_image_signal import residual_forecasts
from scripts.verify_single_image_signal_analysis import independent_fit


def test_recursive_predictions_match_independent_augmented_least_squares():
    rng = np.random.default_rng(74)
    origins = np.tile(np.arange(8, 65, 8), 4)
    ends = origins + 8
    held = np.repeat([False, False, True, True], 8)
    x = rng.normal(size=(32, 4))
    x[:, 2] = x[:, 0] + 2 * x[:, 1]
    x[:, 3] = 1
    y = rng.normal(size=(32, 3))
    offset = rng.normal(size=y.shape)
    predicted, _ = residual_forecasts(x, y, offset, origins, ends, held, prefix=24, test_start=32)
    calibration = ends <= 24
    test = held & (origins >= 32)
    residual = y - offset
    for mode, initial in (("transfer", calibration & ~held), ("frozen", calibration)):
        expected = offset[test] + independent_fit(x, residual, initial, initial, test)
        np.testing.assert_allclose(predicted[mode], expected, rtol=1e-7, atol=1e-9)
    expected = np.zeros_like(y)
    for t in sorted(set(origins[test])):
        fit = calibration | (held & (ends <= t))
        query = test & (origins == t)
        expected[query] = offset[query] + independent_fit(x, residual, calibration, fit, query)
    np.testing.assert_allclose(predicted["online"], expected[test], rtol=1e-7, atol=1e-9)
