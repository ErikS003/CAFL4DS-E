"""Support-only shared and known-domain conditional classification readers."""

from __future__ import annotations

import torch

from cafl4ds.image_utility import balanced_losses
from cafl4ds.reader_calibration import fit_probe
from scripts.evaluate_checkpoint_robustness import probe_predictions


def reader_losses(
    z: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    support: torch.Tensor,
    query: torch.Tensor,
    conditional: bool,
    regularization: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Fit using support labels only; conditional readers require known query domains."""
    if torch.isin(support, query).any():
        raise ValueError("Support/query overlap")
    ce = torch.empty(len(query), dtype=torch.float64)
    error = torch.empty_like(ce)
    for zone in (0, 1) if conditional else (None,):
        fitting = support if zone is None else support[domains[support] == zone]
        positions = torch.arange(len(query)) if zone is None else torch.where(domains[query] == zone)[0]
        selected = query[positions]
        key = "shared" if zone is None else str(zone)
        probe = fit_probe(z, labels, fitting, regularization[key])
        ce[positions] = probe.losses(z[selected], labels[selected])
        prediction = probe_predictions("knn", z[fitting], labels[fitting], z[selected])
        error[positions] = (prediction != labels[selected]).double()
    return {"ce": ce, "knn_error": error}


def reader_metrics(losses: dict[str, torch.Tensor], labels: torch.Tensor, domains: torch.Tensor) -> dict[str, float]:
    """Average scenes within each domain, then give both domains equal weight."""
    return {
        f"{metric}_{zone}": value
        for metric, values in losses.items()
        for zone, value in balanced_losses(values, labels, domains).items()
    }
