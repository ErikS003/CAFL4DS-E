"""Categorical replay-allocation response models, anchored to16 historical images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

ACTIONS = (16, 8, 24)
STATE = ("step", "history_replay_fraction")
HEALTH = ("health_rankme", "health_variance", "health_cka_drift", "health_reconstruction", "health_occupancy_tv")
RECONSTRUCTION = ("selected_reconstruction_mean", "current_reconstruction_mean", "replay_reconstruction_mean")
RECIPES = {
    "action_history": STATE,
    "reconstruction": STATE + RECONSTRUCTION,
    "health_only": STATE + HEALTH,
    "reconstruction_health": STATE + RECONSTRUCTION + HEALTH,
}
REGULARIZATION = (0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass
class ResponseModel:
    """Train-only normalization and action-specific response coefficients."""

    fields: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    regularization: float

    def matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Reference design is identically zero regardless of observed features."""
        raw = np.array([[r["features"][k] for k in self.fields] for r in rows])
        z = (raw - self.mean) / self.scale
        return np.column_stack(
            [
                v
                for action in (8, 24)
                for v in (
                    [float(r["action"] == action) for r in rows],
                    *(z[:, i] * np.array([r["action"] == action for r in rows]) for i in range(len(self.fields))),
                )
            ]
        )

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Query labels or targets cannot enter prediction."""
        return np.asarray(self.matrix(rows) @ self.weight)

    def as_dict(self) -> dict[str, Any]:
        """Save everything needed to replay predictions."""
        return {
            "fields": self.fields,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weight": self.weight.tolist(),
            "regularization": self.regularization,
        }


def fit(rows: list[dict[str, Any]], recipe: str, regularization: float) -> ResponseModel:
    """Fit nonreference development contrasts from supplied trajectories only."""
    if not rows or any(r["action"] not in (8, 24) for r in rows):
        raise ValueError("Require nonreference replay contrasts")
    fields = RECIPES[recipe]
    raw = np.array([[r["features"][k] for k in fields] for r in rows])
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite features")
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-12] = 1.0
    model = ResponseModel(fields, mean, scale, np.zeros(2 * (1 + len(fields))), regularization)
    ridge = Ridge(alpha=regularization, fit_intercept=False)
    ridge.fit(model.matrix(rows), [r["target"] for r in rows])
    model.weight = np.asarray(ridge.coef_)
    return model


def tune(rows: list[dict[str, Any]], recipe: str) -> tuple[ResponseModel, list[dict[str, Any]]]:
    """Inner whole-stream holdout, including fitting normalization inside each fold."""
    groups = sorted({r["stream"] for r in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two training streams")
    records = []
    for regularization in REGULARIZATION:
        for heldout in groups:
            training = [r for r in rows if r["stream"] != heldout]
            validation = [r for r in rows if r["stream"] == heldout]
            model = fit(training, recipe, regularization)
            mse = float(np.mean((model.predict(validation) - np.array([r["target"] for r in validation])) ** 2))
            records.append({"regularization": regularization, "validation_stream": heldout, "mse": mse})
    means = {a: float(np.mean([r["mse"] for r in records if r["regularization"] == a])) for a in REGULARIZATION}
    best = min(means.values())
    chosen = max(a for a in REGULARIZATION if means[a] <= best + 1e-14)
    return fit(rows, recipe, chosen), records


def choose(predictions: dict[int, float], night: dict[int, float] | None = None) -> int:
    """Keep16 for ties; optionally reject nonreference nighttime-harming proposals."""
    eligible = [a for a in ACTIONS if a == 16 or night is None or night[a] >= 0]
    best = max(predictions[a] for a in eligible)
    return next(a for a in eligible if predictions[a] >= best - 1e-12)
