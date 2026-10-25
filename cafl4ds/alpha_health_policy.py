"""Train-only normalized, reference-anchored alpha-response predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

ACTIONS = (1.0, 0.5, 0.0)
REGULARIZATION = (0.1, 1.0, 10.0, 100.0, 1000.0)
STATE = ("step", "history_alpha")
RECON = (
    "arrival_reconstruction_mean",
    "arrival_reconstruction_std",
    "selected_reconstruction_mean",
    "selected_reconstruction_std",
    "selected_coverage_deficit",
    "selected_overlap_reference",
)
HEALTH = ("health_rankme", "health_variance", "health_cka_drift", "health_occupancy_tv", "health_reconstruction")
GRADIENT = (
    "health_gradient_norm",
    "selected_batch_gradient_norm",
    "selected_batch_replay_cosine",
    "selected_gradient_monitor_cosine",
    "selected_gradient_norm",
)
RECIPES = {
    "action_history": STATE,
    "reconstruction_coverage": STATE + RECON,
    "health_only": STATE + HEALTH,
    "reconstruction_health": STATE + RECON + HEALTH,
    "reconstruction_gradients": STATE + RECON + GRADIENT,
    "combined": STATE + RECON + HEALTH + GRADIENT,
}


def choose_action(predictions: dict[float, float]) -> float:
    """Reference-first deterministic ties; no query outcome can enter this choice."""
    best = max(predictions.values())
    return next(a for a in ACTIONS if predictions[a] >= best - 1e-12)


@dataclass
class ActionRegressor:
    """Standardize using training actions only, then interact state with action contrasts."""

    fields: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    regularization: float

    def matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Reference action has exact zero design, including its intercept."""
        raw = np.array([[row["features"][k] for k in self.fields] for row in rows], dtype=float)
        z = (raw - self.mean) / self.scale
        return np.column_stack(
            [
                v
                for a in (0.0, 0.5)
                for v in (
                    (np.array([row["alpha"] == a for row in rows], dtype=float)),
                    *(z[:, i] * np.array([row["alpha"] == a for row in rows], dtype=float) for i in range(z.shape[1])),
                )
            ]
        )

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Predict paired utility, with no access to target or stream identifiers."""
        return np.asarray(self.matrix(rows) @ self.weight)

    def as_dict(self) -> dict[str, Any]:
        """Persist coefficients and training normalization for exact replay."""
        return {
            "fields": self.fields,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weight": self.weight.tolist(),
            "regularization": self.regularization,
        }


def fit(rows: list[dict[str, Any]], recipe: str, regularization: float) -> ActionRegressor:
    """Fit only the supplied nonreference rows and their training target."""
    fields = RECIPES[recipe]
    if not rows or any(row["alpha"] == 1.0 for row in rows):
        raise ValueError("Fit on nonreference contrasts")
    raw = np.array([[row["features"][k] for k in fields] for row in rows], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite predecision features")
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-12] = 1.0
    model = ActionRegressor(fields, mean, scale, np.zeros(2 * (1 + len(fields))), regularization)
    ridge = Ridge(alpha=regularization, fit_intercept=False)
    ridge.fit(model.matrix(rows), np.array([r["target"] for r in rows]))
    model.weight = np.asarray(ridge.coef_)
    return model


def tune(rows: list[dict[str, Any]], recipe: str) -> tuple[ActionRegressor, list[dict[str, Any]]]:
    """Nested stream holdout; target normalization and feature scaling never cross folds."""
    groups = sorted({r["stream"] for r in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two training streams")
    records = []
    for regularization in REGULARIZATION:
        for heldout in groups:
            fitting = [r for r in rows if r["stream"] != heldout]
            validation = [r for r in rows if r["stream"] == heldout]
            model = fit(fitting, recipe, regularization)
            errors = model.predict(validation) - np.array([r["target"] for r in validation])
            records.append(
                {"regularization": regularization, "validation_stream": heldout, "mse": float(np.mean(errors**2))}
            )
    means = {a: float(np.mean([r["mse"] for r in records if r["regularization"] == a])) for a in REGULARIZATION}
    best = min(means.values())
    selected = max(a for a, value in means.items() if value <= best + 1e-14)
    return fit(rows, recipe, selected), records
