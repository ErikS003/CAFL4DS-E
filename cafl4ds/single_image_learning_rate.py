"""A small ordered learning-rate action set with independent trial semantics."""

import numpy as np


def validate_factors(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty learning-rate factors required")
    if values[0] != 1 or np.any(values <= 0) or np.any(np.diff(values) >= 0):
        raise ValueError("Factors must start at one and strictly decrease above zero")
    return values


def set_rates(optimizer, nominal, factor):
    if len(nominal) != len(optimizer.param_groups) or not 0 < factor <= 1:
        raise ValueError("Aligned nominal rates and a positive factor at most one required")
    for group, value in zip(optimizer.param_groups, nominal, strict=True):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Finite positive nominal learning rates required")
        group["lr"] = float(value * factor)


def choose_first_passing(attempted, covariance, factors, cap):
    """Reconstruct the ordered actual covariance decisions from saved traces."""
    factors = validate_factors(factors)
    attempted, covariance = np.asarray(attempted, bool), np.asarray(covariance, float)
    if attempted.shape != factors.shape or covariance.shape != factors.shape:
        raise ValueError("Aligned attempt, covariance and factor arrays required")
    count = int(attempted.sum())
    if not np.array_equal(attempted, np.arange(len(factors)) < count):
        raise ValueError("Trials must form an ordered prefix")
    passing = np.flatnonzero(attempted & (covariance <= cap + 1e-12))
    if len(passing):
        first = int(passing[0])
        if count != first + 1:
            raise ValueError("No trials may follow the first passing candidate")
        return float(factors[first])
    if count not in (0, len(factors)):
        raise ValueError("An unsuccessful search must exhaust the factor ladder")
    return 0.0
