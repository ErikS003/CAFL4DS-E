"""Day/night health features and response recipes with unchanged replay actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from cafl4ds import measurements
from cafl4ds.replay_response import RECIPES as POOLED_RECIPES
from cafl4ds.replay_response import REGULARIZATION, ResponseModel
from cafl4ds.replay_response import fit as pooled_fit

SIGNALS = ("rankme", "variance", "cka_drift", "reconstruction")
DOMAIN = tuple(f"{group}_health_{signal}" for group in ("day", "night") for signal in SIGNALS)
CONTRAST = tuple(f"night_minus_day_{signal}" for signal in SIGNALS)
RECIPES = {
    **POOLED_RECIPES,
    "reconstruction_domain_contrast": POOLED_RECIPES["reconstruction_health"] + CONTRAST,
    "reconstruction_domain_health": POOLED_RECIPES["reconstruction_health"] + DOMAIN,
    "domain_health": POOLED_RECIPES["health_only"] + DOMAIN,
}


def sensor(z: torch.Tensor, warm: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    """Read four health quantities without task labels or learning side effects."""
    return {
        "rankme": measurements.rankme(z),
        "variance": float(F.normalize(z, dim=1).var(dim=0).mean()),
        "cka_drift": measurements.cka_drift(warm, z),
        "reconstruction": float(reconstruction.mean()),
    }


def domain_sensors(
    z: torch.Tensor, warm: torch.Tensor, reconstruction: torch.Tensor, groups: torch.Tensor
) -> dict[str, float]:
    """Partition fixed monitor metadata; no scene labels enter the computation."""
    if len(z) != 96 or warm.shape != z.shape or reconstruction.shape != (96,) or groups.shape != (96,):
        raise ValueError("Require a fixed96-image monitor")
    values = {}
    for domain, group in enumerate(("day", "night")):
        mask = groups == domain
        if int(mask.sum()) != 48:
            raise ValueError("Require48 monitor images per domain")
        values.update({f"{group}_health_{k}": v for k, v in sensor(z[mask], warm[mask], reconstruction[mask]).items()})
    values.update({f"night_minus_day_{k}": values[f"night_health_{k}"] - values[f"day_health_{k}"] for k in SIGNALS})
    return values


def fit(rows: list[dict[str, Any]], recipe: str, regularization: float) -> ResponseModel:
    """Preserve exact original fits for controls; add only prespecified sensor fields."""
    if recipe in POOLED_RECIPES:
        return pooled_fit(rows, recipe, regularization)
    if not rows or any(r["action"] not in (8, 24) for r in rows):
        raise ValueError("Require nonreference contrasts")
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
    """The original nested validation logic with an explicit extended recipe registry."""
    streams = sorted({r["stream"] for r in rows})
    if len(streams) < 2:
        raise ValueError("Require two training streams")
    records = []
    for regularization in REGULARIZATION:
        for heldout in streams:
            train = [r for r in rows if r["stream"] != heldout]
            validation = [r for r in rows if r["stream"] == heldout]
            model = fit(train, recipe, regularization)
            mse = float(np.mean((model.predict(validation) - np.array([r["target"] for r in validation])) ** 2))
            records.append({"regularization": regularization, "validation_stream": heldout, "mse": mse})
    means = {a: float(np.mean([r["mse"] for r in records if r["regularization"] == a])) for a in REGULARIZATION}
    best = min(means.values())
    chosen = max(a for a in REGULARIZATION if means[a] <= best + 1e-14)
    return fit(rows, recipe, chosen), records
