"""Causal permanent-stop guards and paired harm-prevention summaries."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

POINTS = (0, 132, *range(140, 261, 8), 264, *range(272, 393, 8), 396)
GUARDS = {
    "health_combined": (132, "combined"),
    "health_cka": (132, "cka"),
    "health_spread": (132, "spread"),
    "return_health": (264, "combined"),
}


def reading(reference, current):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if (
        reference.ndim != 2
        or reference.shape != current.shape
        or len(reference) <= reference.shape[1]
        or not np.isfinite(np.r_[reference.ravel(), current.ravel()]).all()
    ):
        raise ValueError("Matched finite monitor banks larger than their feature width required")

    def spread(z):
        singular = np.linalg.svd(z, compute_uv=False)
        p = singular / max(singular.sum(), 1e-12) + 1e-7
        unit = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        return np.exp(-sum(p * np.log(p))), unit.var(0, ddof=1).mean()

    a, b = spread(reference), spread(current)
    x, y = reference - reference.mean(0), current - current.mean(0)
    cka = np.sum((x.T @ y) ** 2) / max(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y), 1e-30)
    return np.array([1 - b[0] / a[0], 1 - b[1] / a[1], max(0, 1 - cka)])


class HealthStop:
    def __init__(self, threshold=0.1, patience=2):
        if threshold != 0.1 or patience != 2:
            raise ValueError("Fixed exploratory thresholds and patience required")
        self.threshold, self.patience = threshold, patience
        self.reference, self.streak, self.stop = None, 0, None
        self.events = []

    def observe(self, step, features, activation=132, kind="combined"):
        if kind not in ("combined", "cka", "spread"):
            raise ValueError("Known guard required")
        if self.stop is not None or step < activation:
            return False
        if step == activation:
            self.reference = np.array(features, copy=True)
            return False
        if self.reference is None:
            raise ValueError("Missing fixed acquired reference")
        values = reading(self.reference, features)
        indices = [2] if kind == "cka" else [0, 1] if kind == "spread" else [0, 1, 2]
        # Tiny roundoff at an exact decimal threshold must not change a logical alarm.
        alarm = bool(np.any(values[indices] >= self.threshold - 1e-12))
        self.streak = self.streak + 1 if alarm else 0
        if self.streak >= self.patience:
            self.stop = step
        self.events.append(
            {
                "step": step,
                "values": values.tolist(),
                "alarm": alarm,
                "streak": self.streak,
                "stopped": self.stop is not None,
            }
        )
        return self.stop is not None


def interval(values):
    values = np.asarray(values, float)
    mean = values.mean(0)
    half = t.ppf(0.975, 2) * values.std(0, ddof=1) / np.sqrt(3)
    return {
        "mean_pp": mean.tolist(),
        "ci95_pp": np.stack([mean - half, mean + half], axis=-1).tolist(),
        "seed_values_pp": values.tolist(),
    }


def curve(full, stop):
    points = np.array(POINTS)
    return np.asarray(full)[np.minimum(np.arange(len(points)), np.flatnonzero(points == stop).item())]


def summarize(full, stops):
    """All outcomes are evaluated after causal guard choices have been fixed."""
    full = np.asarray(full, float)
    if full.shape != (3, len(POINTS), 3):
        raise ValueError("Three seeds, declared clocks and balanced/day/night scores required")
    rules = {"frozen": [0] * 3, "full": [396] * 3, "fixed_day1": [132] * 3, "fixed_night": [264] * 3, **stops}
    result = {}
    d1, night = POINTS.index(132), POINTS.index(264)
    import itertools

    for name, selected in rules.items():
        paths = np.stack([curve(v, stop) for v, stop in zip(full, selected, strict=True)])
        peak = np.maximum.accumulate(paths, axis=1)
        terminal_full = paths[:, -1] - full[:, -1]
        contrast = interval(terminal_full)
        random = [
            np.mean([curve(v, k)[-1] for v, k in zip(full, permutation, strict=True)], axis=0)
            for permutation in itertools.permutations(selected)
        ]
        result[name] = {
            "stops": selected,
            "updates_total": sum(selected),
            "trajectory_pp": paths.tolist(),
            "terminal_accuracy": interval(paths[:, -1]),
            "terminal_minus_full": contrast,
            "terminal_minus_warm": interval(paths[:, -1] - full[:, 0]),
            "night_minus_day1": interval(paths[:, night] - paths[:, d1]),
            "return_minus_night": interval(paths[:, -1] - paths[:, night]),
            "maximum_drawdown": interval((peak - paths).max(1)),
            "mean_checkpoint_deficit": interval((peak - paths).mean(1)),
            "matched_work_random_terminal_pp": np.mean(random, axis=0).tolist(),
            "material_damping_descriptive": bool(
                (terminal_full[:, 1:] > 0).all()
                and (terminal_full[:, 1:].mean(0) >= 0.5).all()
                and (np.array(contrast["ci95_pp"])[1:, 0] > 0).all()
            ),
        }
    return {
        "rules": result,
        "points": list(POINTS),
        "seeds": [5003, 5009, 5011],
        "metric_order": ["balanced", "day", "night"],
        "controller_qualified": False,
        "catastrophic_forgetting_established": False,
        "fresh_source_confirmation": False,
    }
