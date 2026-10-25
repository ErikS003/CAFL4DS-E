"""Causal health changes between two saved points on the same trajectory."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from cafl4ds.domain_replay_health import CONTRAST, DOMAIN
from cafl4ds.domain_replay_health import RECIPES as LEVEL_RECIPES
from cafl4ds.domain_replay_health import fit as level_fit
from cafl4ds.replay_response import REGULARIZATION, ResponseModel

CHANGE = tuple(f"change_{name}" for name in CONTRAST)
DOMAIN_CHANGE = tuple(f"change_{name}" for name in DOMAIN)
RECIPES = {
    "action_history": LEVEL_RECIPES["action_history"],
    "reconstruction": LEVEL_RECIPES["reconstruction"],
    "reconstruction_health": LEVEL_RECIPES["reconstruction_health"],
    "reconstruction_domain_contrast": LEVEL_RECIPES["reconstruction_domain_contrast"],
    "contrast_change": LEVEL_RECIPES["reconstruction_health"] + CHANGE,
    "contrast_level_change": LEVEL_RECIPES["reconstruction_domain_contrast"] + CHANGE,
    "domain_level_change": LEVEL_RECIPES["reconstruction_domain_health"] + DOMAIN_CHANGE,
}


def changes(early: dict[str, Any], late: dict[str, Any]) -> dict[str, float]:
    """Only pair the same stream and seed; normalize by the elapsed eight updates."""
    left, right = early["state"].rsplit("_", 1), late["state"].rsplit("_", 1)
    if left[0] != right[0] or left[1] != "4" or right[1] != "12":
        raise ValueError("Require the same trajectory at prefixes 4 and 12")
    return {f"change_{k}": (float(late[k]) - float(early[k])) / 8 for k in (*CONTRAST, *DOMAIN)}


def fit(rows: list[dict[str, Any]], recipe: str, regularization: float) -> ResponseModel:
    """Keep level controls exact and fit new recipes with the same ridge design."""
    if recipe in LEVEL_RECIPES:
        return level_fit(rows, recipe, regularization)
    fields = RECIPES[recipe]
    if not rows or any(r["action"] not in (8, 24) for r in rows):
        raise ValueError("Require nonreference contrasts")
    raw = np.array([[r["features"][k] for k in fields] for r in rows])
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite features")
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-12] = 1
    model = ResponseModel(fields, mean, scale, np.zeros(2 * (1 + len(fields))), regularization)
    ridge = Ridge(alpha=regularization, fit_intercept=False)
    ridge.fit(model.matrix(rows), [r["target"] for r in rows])
    model.weight = np.asarray(ridge.coef_)
    return model


def tune(rows: list[dict[str, Any]], recipe: str) -> tuple[ResponseModel, list[dict[str, Any]]]:
    """Leave whole streams out, fitting normalization using training states only."""
    streams = sorted({r["stream"] for r in rows})
    if len(streams) < 2:
        raise ValueError("Require two training streams")
    records = []
    for regularization in REGULARIZATION:
        for heldout in streams:
            training = [r for r in rows if r["stream"] != heldout]
            validation = [r for r in rows if r["stream"] == heldout]
            model = fit(training, recipe, regularization)
            mse = float(np.mean((model.predict(validation) - np.array([r["target"] for r in validation])) ** 2))
            records.append({"regularization": regularization, "validation_stream": heldout, "mse": mse})
    means = {a: float(np.mean([r["mse"] for r in records if r["regularization"] == a])) for a in REGULARIZATION}
    best = min(means.values())
    chosen = max(a for a in REGULARIZATION if means[a] <= best + 1e-14)
    return fit(rows, recipe, chosen), records
