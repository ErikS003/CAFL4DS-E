"""Probe regularization selected exclusively by balanced support-label cross-validation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.image_utility import FixedProbe, balanced_losses

C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_SEEDS = (71011, 71017, 71023)


def balanced_folds(
    labels: torch.Tensor, domains: torch.Tensor, support: torch.Tensor, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reserve four support images per domain/class; fold fitting has eight per cell."""
    if len(support) != 72 or len(torch.unique(support)) != 72:
        raise ValueError("Require 72 distinct support images")
    generator = torch.Generator().manual_seed(seed)
    cells = []
    for domain in (0, 1):
        for label in torch.unique(labels[support], sorted=True):
            cell = support[(domains[support] == domain) & (labels[support] == label)]
            if len(cell) != 12:
                raise ValueError("Require twelve support images per domain/class")
            cells.append(cell[torch.randperm(12, generator=generator)])
    result = []
    for fold in range(3):
        validation = torch.cat([cell[fold * 4 : (fold + 1) * 4] for cell in cells])
        fitting = support[~torch.isin(support, validation)]
        result.append((fitting, validation))
    return result


def fit_probe(z: torch.Tensor, labels: torch.Tensor, fitting: torch.Tensor, c: float) -> FixedProbe:
    """Normalization and fitting use only the fitting fold, never held-out support/query."""
    x = z[fitting].numpy().astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    classifier = LogisticRegression(C=c, max_iter=1000, random_state=63000)
    classifier.fit((x - mean) / scale, labels[fitting].numpy())
    return FixedProbe(mean, scale, classifier)


def choose_c(means: dict[float, float]) -> float:
    """Resolve numerical CV ties toward stronger regularization."""
    minimum = min(means.values())
    return min(c for c, value in means.items() if value <= minimum + 1e-10)


def support_cv(
    z: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, support: torch.Tensor
) -> list[dict[str, float | int]]:
    """Query observations cannot enter fold construction, fitting, normalization or scoring."""
    records = []
    for seed in CV_SEEDS:
        for fold, (fitting, validation) in enumerate(balanced_folds(labels, domains, support, seed)):
            for c in C_GRID:
                probe = fit_probe(z, labels, fitting, c)
                losses = probe.losses(z[validation], labels[validation])
                metrics = balanced_losses(losses, labels[validation], domains[validation])
                records.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "C": c,
                        "ce": metrics["balanced"],
                        "ce_A": metrics["A"],
                        "ce_B": metrics["B"],
                    }
                )
    return records


def probe_outcomes(
    z: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    support: torch.Tensor,
    roles: dict[str, torch.Tensor],
    c: float,
) -> dict[str, dict[str, float]]:
    """Evaluate a frozen regularization recipe with support-only fitted heads."""
    if torch.isin(support, torch.cat(list(roles.values()))).any():
        raise ValueError("Support/query overlap")
    probe = fit_probe(z, labels, support, c)
    result = {}
    for role in ("development", "holdout"):
        indices = roles[role]
        probabilities = probe.classifier.predict_proba((z[indices].numpy() - probe.mean) / probe.scale)
        prediction = torch.from_numpy(probe.classifier.classes_[probabilities.argmax(1)])
        losses = {
            "refit_ce": probe.losses(z[indices], labels[indices]),
            "linear_error": (prediction != labels[indices]).double(),
        }
        result[role] = {
            f"{metric}_{zone}": value
            for metric, values in losses.items()
            for zone, value in balanced_losses(values, labels[indices], domains[indices]).items()
        }
    return result
