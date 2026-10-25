"""Fixed-reference geometry measurements with separate motion and collapse guards."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cafl4ds.measurements import cosine_drift, linear_cka, offdiag_covariance, rankme

FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")


def geometry(features: np.ndarray) -> dict[str, float]:
    """Read the existing geometry instruments on a fixed image population."""
    z = torch.as_tensor(features).float()
    if z.ndim != 2 or len(z) <= z.shape[1] or not torch.isfinite(z).all():
        raise ValueError("Finite reference bank larger than embedding width required")
    normalized = F.normalize(z, dim=1)
    return {
        "rank": rankme(z),
        "variance": float(normalized.var(0, unbiased=True).mean()),
        "covariance": offdiag_covariance(normalized),
    }


def changes(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Measure paired changes without equating representation motion with damage."""
    a, b = geometry(before), geometry(after)
    return {
        "rank_drop": (a["rank"] - b["rank"]) / max(a["rank"], 1e-12),
        "variance_drop": (a["variance"] - b["variance"]) / max(a["variance"], 1e-12),
        "covariance_rise": (b["covariance"] - a["covariance"]) / max(a["covariance"], 1e-12),
        "cka_drift": max(0.0, 1 - linear_cka(before, after)),
        "cosine_drift": cosine_drift(before, after),
    }


def embeddings(model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    """Read fixed, unaugmented embeddings and restore every module mode and RNG."""
    device = next(model.parameters()).device
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            with torch.no_grad():
                values = torch.cat([model.encoder.embed(batch.to(device)).cpu() for batch in images.split(32)])
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    return values.numpy()


def forecasts(points: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    """Calibrate terminal damage with complete source-group holdout and fixed ridge1."""
    y = np.array([p["terminal_damage"] for p in points])
    x = np.array([[p[f] for f in features] for p in points]).reshape(18, len(features))
    groups = sorted({p["stream"] for p in points})
    if len(points) != 18 or len(groups) != 6:
        raise ValueError("Six groups with three checkpoint means required")
    predictions = np.zeros(18)
    coefficients = {}
    for group in groups:
        test = np.array([p["stream"] == group for p in points])
        center, scale = x[~test].mean(0), x[~test].std(0)
        scale[scale == 0] = 1
        z = (x[~test] - center) / scale
        beta = np.linalg.solve(z.T @ z + np.eye(len(features)), z.T @ (y[~test] - y[~test].mean()))
        predictions[test] = y[~test].mean() + ((x[test] - center) / scale) @ beta
        coefficients[group] = dict(zip(features, beta.tolist(), strict=True))
    error = (predictions - y) ** 2
    return {
        "predictions": predictions.tolist(),
        "coefficients": coefficients,
        "mse": float(error.mean()),
        "source_mse": {g: float(error[[p["stream"] == g for p in points]].mean()) for g in groups},
    }
