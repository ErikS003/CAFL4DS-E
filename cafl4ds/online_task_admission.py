"""Bounded task-informed adjustment of ordinary heuristic admission."""
import numpy as np


def adjusted_probability(base, prediction, ready, reason, mode):
    if not ready or reason not in ('good', 'borderline', 'unfavorable', 'fallback') or mode not in ('normal', 'caution'):
        return float(base)
    if not np.isfinite(prediction) or prediction == 0:
        return float(base)
    return float(np.clip(base * (1.2 if prediction < 0 else .8), .05, .95))
