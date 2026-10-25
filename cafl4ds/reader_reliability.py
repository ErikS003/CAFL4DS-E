"""Support-only reader comparisons on cached, identical encoder features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.image_utility import FixedProbe, balanced_losses
from scripts.evaluate_checkpoint_robustness import probe_predictions

DRAW_SEEDS = (69011, 69017, 69023, 69029, *range(70011, 70027))
TRAINING_REPEATS = (0, 2, 3, 4)


def support_draws(
    labels: torch.Tensor, domains: torch.Tensor, query: torch.Tensor, original: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Use six/twelve images per domain/class, with private RNG and no query overlap."""
    pool = torch.ones(len(labels), dtype=torch.bool)
    pool[query] = False
    if len(original) != 36 or not pool[original].all() or len(torch.unique(original)) != 36:
        raise ValueError("Original support must contain 36 distinct reserved images")
    result = {"original36": original.clone(), "full72": pool.nonzero().flatten()}
    for seed in DRAW_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        parts = []
        for domain in (0, 1):
            for label in torch.unique(labels, sorted=True):
                cell = (pool & (domains == domain) & (labels == label)).nonzero().flatten()
                if len(cell) != 12 or ((domains[original] == domain) & (labels[original] == label)).sum() != 6:
                    raise ValueError("Require twelve reserved and six original support images per cell")
                parts.append(cell[torch.randperm(12, generator=generator)[:6]])
        result[f"draw36_{seed}"] = torch.cat(parts)
    if len(result["full72"]) != 72:
        raise ValueError("Require six balanced support cells")
    return result


def fit_reader(z: torch.Tensor, labels: torch.Tensor, support: torch.Tensor) -> FixedProbe:
    """Keep normalization and logistic-regression fitting confined to declared support."""
    x = z[support].numpy().astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    classifier = LogisticRegression(C=1, max_iter=1000, random_state=63000)
    classifier.fit((x - mean) / scale, labels[support].numpy())
    return FixedProbe(mean, scale, classifier)


def read_cached(
    z: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, roles: dict[str, torch.Tensor], support: torch.Tensor
) -> dict[str, Any]:
    """Refit CE/linear and cosine distance-weighted kNN readers without SSL/model access."""
    queries = torch.cat([roles[r] for r in ("development", "holdout")])
    if torch.isin(support, queries).any():
        raise ValueError("Support/query overlap")
    probe = fit_reader(z, labels, support)
    result = {}
    for role in ("development", "holdout"):
        indices = roles[role]
        probabilities = probe.classifier.predict_proba((z[indices].numpy() - probe.mean) / probe.scale)
        prediction = torch.from_numpy(probe.classifier.classes_[probabilities.argmax(1)])
        knn = probe_predictions("knn", z[support], labels[support], z[indices])
        losses = {
            "refit_ce": probe.losses(z[indices], labels[indices]),
            "linear_error": (prediction != labels[indices]).double(),
            "knn_error": (knn != labels[indices]).double(),
        }
        result[role] = {
            f"{metric}_{zone}": value
            for metric, values in losses.items()
            for zone, value in balanced_losses(values, labels[indices], domains[indices]).items()
        }
    return result
