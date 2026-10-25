"""Prospective accumulated-movement calibration at fixed update landmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import t

LANDMARKS = (0, 4, 8, 16, 32)
MODELS = {"mean": (), "count_loss": ("updates", "recent_loss"), "cka": ("updates", "recent_loss", "cka_drift")}


def remaining_effect(stop: dict[str, float], endpoint: dict[str, float]) -> dict[str, float]:
    """Define only the capability change still ahead of a landmark."""
    delta = {key: endpoint[key] - stop[key] for key in ("balanced", "A", "B")}
    return {**delta, "damage": -min(delta["A"], delta["B"]), "balanced_damage": -delta["balanced"]}


def fit(points: list[dict[str, Any]], fields: tuple[str, ...], target: str = "damage") -> dict[str, Any]:
    """Predict held histories with train-fold scaling and fixed ridge one."""
    y = np.array([p[target] for p in points])
    x = np.array([[p[f] for f in fields] for p in points]).reshape(len(points), len(fields))
    groups = sorted({p["stream"] for p in points})
    if len(groups) != 6 or len(points) != 18 or any(sum(p["stream"] == g for p in points) != 3 for g in groups):
        raise ValueError("Six histories with three checkpoint means are required")
    prediction = np.zeros(len(points))
    slopes = {}
    for group in groups:
        test = np.array([p["stream"] == group for p in points])
        center, scale = x[~test].mean(0), x[~test].std(0)
        scale[scale == 0] = 1
        z = (x[~test] - center) / scale
        beta = np.linalg.solve(z.T @ z + np.eye(len(fields)), z.T @ (y[~test] - y[~test].mean()))
        prediction[test] = y[~test].mean() + ((x[test] - center) / scale) @ beta
        slopes[group] = dict(zip(fields, beta.tolist(), strict=True))
    errors = (prediction - y) ** 2
    return {
        "predictions": prediction.tolist(),
        "coefficients": slopes,
        "mse": float(errors.mean()),
        "source_mse": {g: float(errors[[p["stream"] == g for p in points]].mean()) for g in groups},
    }


def report(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a frozen incremental-prediction screen, preserving all failures."""
    models = {name: fit(points, fields) for name, fields in MODELS.items()}
    contrasts = {}
    for baseline in ("mean", "count_loss"):
        values = np.array(
            [models[baseline]["source_mse"][g] - models["cka"]["source_mse"][g] for g in models[baseline]["source_mse"]]
        )
        width = t.ppf(0.9875, 5) * values.std(ddof=1) / np.sqrt(6)
        contrasts[baseline] = {
            "mse_improvement": float(values.mean()),
            "adjusted_ci95": [float(values.mean() - width), float(values.mean() + width)],
            "reduction_fraction": 1 - models["cka"]["mse"] / max(models[baseline]["mse"], 1e-12),
        }
    population = {
        "large_future_damage": sum(p["damage"] >= 1 for p in points),
        "no_future_domain_damage": sum(p["damage"] <= 0 for p in points),
        "beneficial_continuations": sum(p["balanced"] >= 0.5 for p in points),
    }
    gates = {
        "incremental_prediction": all(
            c["reduction_fraction"] >= 0.1 and c["adjusted_ci95"][0] > 0 for c in contrasts.values()
        ),
        "positive_cka_folds": sum(c["cka_drift"] > 0 for c in models["cka"]["coefficients"].values()) >= 5,
        "two_sided_population": all(v >= 2 for v in population.values()),
    }
    return {
        "points": points,
        "models": models,
        "contrasts": contrasts,
        "population": population,
        "gates": gates,
        "qualified": all(gates.values()),
    }
