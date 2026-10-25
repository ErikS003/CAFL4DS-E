"""Nested label budgets and shared/known-domain linear probe diagnostics."""

from __future__ import annotations

import torch

from cafl4ds.image_utility import balanced_losses
from cafl4ds.reader_calibration import fit_probe


def nested_supports(labels: torch.Tensor, domains: torch.Tensor, pool: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create reproducible nested balanced samples without changing caller RNG."""
    if len(pool) != 1152 or len(torch.unique(pool)) != 1152:
        raise ValueError("Require 1152 distinct support images")
    result = {}
    for seed in (80011, 80017, 80023):
        generator = torch.Generator().manual_seed(seed)
        cells = []
        for domain in (0, 1):
            for label in (0, 1, 2):
                cell = pool[(domains[pool] == domain) & (labels[pool] == label)]
                if len(cell) != 192:
                    raise ValueError("Require 192 images per domain/scene")
                cells.append(cell[torch.randperm(192, generator=generator)])
        for size in (16, 32, 64, 128):
            result[f"{size}_{seed}"] = torch.cat([cell[:size] for cell in cells])
    result["192_all"] = pool.clone()
    return result


def linear_outcomes(
    z: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    support: torch.Tensor,
    queries: dict[str, torch.Tensor],
    mode: str,
    cs: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Fit each support-only head once; classify queries using known domains if requested."""
    if mode not in ("shared", "conditional"):
        raise ValueError("Unknown reader mode")
    if torch.isin(support, torch.cat(list(queries.values()))).any():
        raise ValueError("Support/query overlap")
    losses = {
        role: {
            "linear_error": torch.empty(len(q), dtype=torch.float64),
            "refit_ce": torch.empty(len(q), dtype=torch.float64),
        }
        for role, q in queries.items()
    }
    for zone in (None,) if mode == "shared" else (0, 1):
        fitting = support if zone is None else support[domains[support] == zone]
        key = "shared" if zone is None else str(zone)
        probe = fit_probe(z, labels, fitting, cs[key])
        for role, query in queries.items():
            positions = torch.arange(len(query)) if zone is None else torch.where(domains[query] == zone)[0]
            selected = query[positions]
            prediction = probe.classifier.predict((z[selected].numpy() - probe.mean) / probe.scale)
            losses[role]["linear_error"][positions] = (torch.from_numpy(prediction) != labels[selected]).double()
            losses[role]["refit_ce"][positions] = probe.losses(z[selected], labels[selected])
    return {
        role: {
            f"{m}_{zone}": value
            for m, values in metrics.items()
            for zone, value in balanced_losses(values, labels[queries[role]], domains[queries[role]]).items()
        }
        for role, metrics in losses.items()
    }
