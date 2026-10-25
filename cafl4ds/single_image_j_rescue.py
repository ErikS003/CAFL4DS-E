"""Actual-objective safeguards for smaller-rate updates rejected by the J forecast."""

from __future__ import annotations

import numpy as np

from cafl4ds.single_image_covariance_regularization import next_factor


def trial_passes(rescue: bool, delta_j: float, covariance: float, cap: float, minimum_gain: float) -> bool:
    """Preserve ordinary covariance admission; require measurable J gain for rescues."""
    if not np.isfinite([delta_j, covariance, cap, minimum_gain]).all() or minimum_gain <= 0:
        raise ValueError("Finite candidate measurements and a positive improvement margin required")
    return bool(covariance <= cap + 1e-12 and (not rescue or delta_j < -minimum_gain))


def retry_factor(
    factor: float, rescue: bool, current: float, candidate: float, cap: float, safety: float, shrink: float
) -> float:
    """Shrink J failures; apply stricter measured headroom scaling on covariance failures."""
    if not 0 < shrink < 1:
        raise ValueError("A strict rate shrink is required")
    if not rescue:
        return next_factor(factor, current, candidate, cap, safety)
    result = factor * shrink
    if candidate > cap + 1e-12:
        result = min(result, next_factor(factor, current, candidate, cap, safety))
    return float(result)
