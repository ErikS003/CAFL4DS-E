"""Balanced support-capacity diagnostics with fixed disjoint outcome queries."""

from __future__ import annotations

import torch

from cafl4ds.image_utility import balanced_losses
from cafl4ds.reader_calibration import C_GRID, CV_SEEDS, fit_probe, probe_outcomes
from scripts.evaluate_checkpoint_robustness import probe_predictions


def expanded_support(original: torch.Tensor, monitor: torch.Tensor, roles: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reclassify only monitor images as support; reserve both outcome-query sets."""
    support = torch.cat([original, monitor])
    if len(original) != 72 or len(monitor) != 192 or len(torch.unique(support)) != 264:
        raise ValueError("Require 72 original and 192 distinct monitor images")
    queries = torch.cat([roles[r] for r in ("development", "holdout")])
    if torch.isin(support, queries).any():
        raise ValueError("Support/query overlap")
    return support


def capacity_folds(
    labels: torch.Tensor, domains: torch.Tensor, support: torch.Tensor, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Four balanced support folds: 33 fitting and 11 held-out images per cell."""
    if len(support) != 264 or len(torch.unique(support)) != 264:
        raise ValueError("Require 264 distinct support images")
    generator = torch.Generator().manual_seed(seed)
    cells = []
    for domain in (0, 1):
        for label in torch.unique(labels[support], sorted=True):
            cell = support[(domains[support] == domain) & (labels[support] == label)]
            if len(cell) != 44:
                raise ValueError("Require 44 support images per domain/class")
            cells.append(cell[torch.randperm(44, generator=generator)])
    result = []
    for fold in range(4):
        validation = torch.cat([cell[fold * 11 : (fold + 1) * 11] for cell in cells])
        fitting = support[~torch.isin(support, validation)]
        result.append((fitting, validation))
    return result


def capacity_cv(
    z: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, support: torch.Tensor
) -> list[dict[str, float | int]]:
    """Choose regularization from fitting-parent support folds only."""
    records = []
    for seed in CV_SEEDS:
        for fold, (fitting, validation) in enumerate(capacity_folds(labels, domains, support, seed)):
            for c in C_GRID:
                probe = fit_probe(z, labels, fitting, c)
                metrics = balanced_losses(
                    probe.losses(z[validation], labels[validation]), labels[validation], domains[validation]
                )
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


def capacity_outcomes(
    z: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    support: torch.Tensor,
    roles: dict[str, torch.Tensor],
    c: float,
) -> dict[str, dict[str, float]]:
    """Read CE/linear/kNN with the expanded neighbor pool and unchanged outcome queries."""
    values = probe_outcomes(z, labels, domains, support, roles, c)
    for role in ("development", "holdout"):
        indices = roles[role]
        prediction = probe_predictions("knn", z[support], labels[support], z[indices])
        errors = (prediction != labels[indices]).double()
        values[role].update(
            {
                f"knn_error_{zone}": value
                for zone, value in balanced_losses(errors, labels[indices], domains[indices]).items()
            }
        )
    return values
