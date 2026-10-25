"""Read-only encoder gradient contrasts and prespecified local response recipes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge

from cafl4ds.image_utility import observation
from cafl4ds.replay_response import REGULARIZATION, ResponseModel
from cafl4ds.ssl.base import SSLMethod

HISTORY = ("step", "history_replay_fraction")
COMPOSITION = HISTORY + ("current_night_fraction", "memory_night_fraction", "domain_changed")
RECON = COMPOSITION + ("current_reconstruction_mean", "replay_reconstruction_mean", "selected_reconstruction_mean")
HEALTH = tuple(f"health_{k}" for k in ("rankme", "variance", "cka_drift", "reconstruction")) + tuple(
    f"night_minus_day_{k}" for k in ("rankme", "variance", "cka_drift", "reconstruction")
)
GRADIENT = ("log_current_gradient_norm", "log_replay_gradient_norm", "gradient_cosine", "gradient_cosine_std")
RECIPES = {
    "action_history": HISTORY,
    "composition": COMPOSITION,
    "reconstruction": RECON,
    "health": RECON + HEALTH,
    "gradients": RECON + GRADIENT,
    "health_gradients": RECON + HEALTH + GRADIENT,
}


def encoder_gradient(method: SSLMethod, images: torch.Tensor, seed: int) -> torch.Tensor:
    """Return the full encoder SSL gradient without assigning parameter gradients."""
    parameters = [p for p in method.encoder.parameters() if p.requires_grad]
    with observation(method, seed), torch.enable_grad():
        gradients = torch.autograd.grad(method.training_step(images), parameters, allow_unused=True)
    vector = torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).detach().flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )
    if not torch.isfinite(vector).all():
        raise ValueError("Nonfinite encoder gradient")
    return vector


def gradient_signals(method: SSLMethod, current: torch.Tensor, replay: torch.Tensor) -> dict[str, float]:
    """Average two matched stochastic SSL mask draws on equal32-image inputs."""
    current_norms, replay_norms, cosines = [], [], []
    for seed in (91001, 92010):
        a, b = encoder_gradient(method, current, seed), encoder_gradient(method, replay, seed)
        na, nb = float(a.norm()), float(b.norm())
        if na <= 0 or nb <= 0:
            raise ValueError("Zero gradient cannot define alignment")
        current_norms.append(na)
        replay_norms.append(nb)
        cosines.append(float(torch.dot(a, b) / (na * nb)))
    return {
        "log_current_gradient_norm": float(np.log(np.mean(current_norms))),
        "log_replay_gradient_norm": float(np.log(np.mean(replay_norms))),
        "gradient_cosine": float(np.mean(cosines)),
        "gradient_cosine_std": float(np.std(cosines)),
    }


def fit(rows: list[dict[str, Any]], recipe: str, regularization: float) -> ResponseModel:
    """Normalize using only supplied training rows, with categorical action effects."""
    if not rows or any(r["action"] not in (8, 24) for r in rows):
        raise ValueError("Require nonreference training contrasts")
    fields = RECIPES[recipe]
    raw = np.array([[r["features"][k] for k in fields] for r in rows])
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite observation")
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-12] = 1
    model = ResponseModel(fields, mean, scale, np.zeros(2 * (1 + len(fields))), regularization)
    ridge = Ridge(alpha=regularization, fit_intercept=False)
    ridge.fit(model.matrix(rows), [r["target"] for r in rows])
    model.weight = np.asarray(ridge.coef_)
    return model


def tune(rows: list[dict[str, Any]], recipe: str) -> tuple[ResponseModel, list[dict[str, Any]]]:
    """Tune on whole original image pools, retaining all related trajectories together."""
    groups = sorted({r["pool_group"] for r in rows})
    if len(groups) < 2:
        raise ValueError("Need two training pools")
    records = []
    for alpha in REGULARIZATION:
        for group in groups:
            train = [r for r in rows if r["pool_group"] != group]
            validation = [r for r in rows if r["pool_group"] == group]
            model = fit(train, recipe, alpha)
            mse = float(np.mean((model.predict(validation) - np.array([r["target"] for r in validation])) ** 2))
            records.append({"regularization": alpha, "validation_pool": group, "mse": mse})
    means = {a: float(np.mean([r["mse"] for r in records if r["regularization"] == a])) for a in REGULARIZATION}
    chosen = max(a for a in REGULARIZATION if means[a] <= min(means.values()) + 1e-14)
    return fit(rows, recipe, chosen), records
