"""Research-only reactive and paid health gates with a monitor-readability fallback."""

import numpy as np


def decisions(domain_increase, readability=None, budget=0.05, minimum_gap=0.25):
    """Unreadable monitors defer to normal training; this is not a safety certificate."""
    change = np.asarray(domain_increase, float)
    if change.ndim != 2 or change.shape[1] != 2 or not np.isfinite(change).all():
        raise ValueError("Finite day/night functional-health changes required")
    readable = np.ones(len(change), bool) if readability is None else np.asarray(readability) >= minimum_gap
    if readability is not None and not np.isfinite(np.asarray(readability)).all():
        raise ValueError("Finite readability values required")
    if readable.shape != (len(change),):
        raise ValueError("One current-state readability value per decision required")
    return ~readable | (np.max(change, axis=1) <= budget)
