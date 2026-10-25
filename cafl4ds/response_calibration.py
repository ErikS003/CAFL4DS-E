"""Reference-anchored predictors and support-only measurement sensitivity."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.image_utility import FixedProbe, balanced_losses, embed
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from scripts.evaluate_checkpoint_robustness import probe_predictions

CALIBRATION_SCHEDULES = ("constant_half", "constant_1", "half_to_one", "one_to_zero")
NONREFERENCE = tuple(s for s in CALIBRATION_SCHEDULES if s != "constant_1")
RECIPE_FIELDS = {
    "action": (),
    "history": ("previous_alpha",),
    "arrival": ("previous_alpha", "arrival_reconstruction", "arrival_cluster_tv"),
    "health": (
        "previous_alpha",
        "arrival_reconstruction",
        "arrival_cluster_tv",
        "health_cka_drift",
        "health_occupancy_tv",
    ),
}
SUPPORT_SEEDS = (69011, 69017, 69023, 69029)


def anchored_features(
    schedule: str, state: dict[str, float], recipe: str, centers: dict[str, float], scales: dict[str, float]
) -> dict[str, float]:
    """Multiply fit-normalized state by action contrasts; reference input is identically zero."""
    if schedule not in CALIBRATION_SCHEDULES:
        raise ValueError("Undeclared calibration schedule")
    result = {}
    for action in NONREFERENCE:
        contrast = float(schedule == action)
        result[f"action_{action}"] = contrast
        for field in RECIPE_FIELDS[recipe]:
            result[f"{action}_{field}"] = contrast * (state[field] - centers[field]) / scales[field]
    return result


def support_sensitivity(method: SSLMethod, evaluation: Evaluation, roles: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Refit four equal-size support draws; query labels enter scoring only, never head fitting."""
    z = embed(method, evaluation.images)
    pool = torch.ones(len(z), dtype=torch.bool)
    pool[evaluation.query] = False
    result = {}
    for seed in SUPPORT_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        parts = []
        for domain in (0, 1):
            for label in torch.unique(evaluation.labels, sorted=True):
                cell = (pool & (evaluation.domains == domain) & (evaluation.labels == label)).nonzero().flatten()
                if len(cell) != 12:
                    raise ValueError("Sensitivity requires twelve reserved support images per cell")
                parts.append(cell[torch.randperm(len(cell), generator=generator)[:6]])
        support = torch.cat(parts)
        x = z[support].numpy().astype(np.float64)
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1
        classifier = LogisticRegression(C=1, max_iter=1000, random_state=63000)
        classifier.fit((x - mean) / scale, evaluation.labels[support].numpy())
        probe = FixedProbe(mean, scale, classifier)
        read = {}
        for role in ("development", "holdout"):
            indices = roles[role]
            labels, domains = evaluation.labels[indices], evaluation.domains[indices]
            probabilities = classifier.predict_proba((z[indices].numpy() - mean) / scale)
            linear = torch.from_numpy(classifier.classes_[probabilities.argmax(1)])
            knn = probe_predictions("knn", z[support], evaluation.labels[support], z[indices])
            values = {
                "refit_ce": probe.losses(z[indices], labels),
                "linear_error": (linear != labels).double(),
                "knn_error": (knn != labels).double(),
            }
            read[role] = {
                f"{metric}_{key}": value
                for metric, losses in values.items()
                for key, value in balanced_losses(losses, labels, domains).items()
            }
        result[str(seed)] = read
    return result
