"""Direct forecasts of an image update relative to a matched zero-gradient step."""

from __future__ import annotations

import numpy as np

from cafl4ds.single_image_forecast import HEALTH
from cafl4ds.single_image_signal import residual_forecasts


def isolated_effect(actual: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Signed actual-minus-zero intervention contrast, in the metric's units."""
    if actual.shape != zero.shape or not np.isfinite(actual).all() or not np.isfinite(zero).all():
        raise ValueError("Require aligned finite actual and zero-step measurements")
    return actual - zero


def designs(before, ordinary_lag, optimizer, image, gradient, origins, metrics):
    """Predictor inputs contain no current zero-step or actual-update outcomes."""
    bank = [metrics.index(k) for k in HEALTH]
    state = np.column_stack((before, ordinary_lag, optimizer, origins))
    health = np.column_stack((before[:, bank], ordinary_lag[:, bank], optimizer, origins))
    return {
        "state": state,
        "image": np.column_stack((state, image)),
        "gradient": np.column_stack((state, image, gradient)),
        "label_free_state": health,
        "label_free_image": np.column_stack((health, image)),
        "label_free_gradient": np.column_stack((health, image, gradient)),
    }


def forecasts(x, target, origins, held, histories, settings):
    """One-step targets enter shared online fitting only after their arrival ends."""
    ends = origins + 1
    calibration = ends <= settings["prefix_images"]
    transfer = calibration & ~held
    test = held & (origins >= settings["test_origin_start"])
    zero = np.zeros_like(target)
    predictions = {
        "zero_effect": zero[test],
        "transfer_mean": np.tile(target[transfer].mean(0), (test.sum(), 1)),
        "prefix_mean": np.tile(target[calibration].mean(0), (test.sum(), 1)),
        "online_mean": np.array([target[calibration | (held & (ends <= t))].mean(0) for t in origins[test]]),
    }
    previous = np.zeros_like(target)
    for i in np.flatnonzero(test):
        available = np.flatnonzero((histories == histories[i]) & (ends <= origins[i]))
        if len(available):
            previous[i] = target[available[np.argmax(ends[available])]]
    predictions["previous_effect"] = previous[test]
    metadata = {}
    for recipe, design in x.items():
        fitted, meta = residual_forecasts(
            design,
            target,
            zero,
            origins,
            ends,
            held,
            settings["prefix_images"],
            settings["test_origin_start"],
            settings["ridge_penalty"],
        )
        predictions.update({mode + "_" + recipe: p for mode, p in fitted.items()})
        metadata[recipe] = meta
    return predictions, metadata, test
