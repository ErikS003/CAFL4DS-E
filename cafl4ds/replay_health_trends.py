"""Causal health changes and train-only categorical replay response fits."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from cafl4ds.replay_local import HEALTH, RECIPES
from cafl4ds.replay_response import ResponseModel

FIELDS = {k: RECIPES[k] for k in ("composition", "reconstruction", "health", "gradients", "health_gradients")}
FIELDS["health_trend"] = FIELDS["health"] + tuple(f"trend_{k}" for k in HEALTH)
FIELDS["health_gradients_trend"] = FIELDS["health_gradients"] + tuple(f"trend_{k}" for k in HEALTH)


def health_trends(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    """Use an earlier measurement on exactly the same trajectory and pool."""
    if any(current[k] != previous[k] for k in ("parent_state", "schedule", "pool_group")):
        raise ValueError("Trend observations belong to different trajectories")
    elapsed = current["prefix"] - previous["prefix"]
    if elapsed <= 0:
        raise ValueError("A trend cannot use a present or future observation")
    return {f"trend_{k}": (current["features"][k] - previous["features"][k]) / elapsed for k in HEALTH}


def fit_response(rows: list[dict[str, Any]], recipe: str) -> ResponseModel:
    """Fit a fixed ridge100 response; normalize only the supplied training states."""
    fields = FIELDS[recipe]
    raw = np.array([[r["features"][k] for k in fields] for r in rows])
    if len(rows) == 0 or not np.isfinite(raw).all():
        raise ValueError("Missing/nonfinite training features")
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-12] = 1
    model = ResponseModel(fields, mean, scale, np.zeros(2*(1+len(fields))), 100.0)
    ridge = Ridge(alpha=100.0, fit_intercept=False)
    ridge.fit(model.matrix(rows), [r["target"] for r in rows])
    model.weight = np.asarray(ridge.coef_)
    return model
