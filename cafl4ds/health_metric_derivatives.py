"""Differentiable replicas of the fixed-bank image-health instruments."""

from __future__ import annotations

import torch
import torch.nn.functional as F

METRICS = ("variance", "covariance", "rank", "cka", "cosine")


def differentiable_health(z: torch.Tensor, warm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Match original float64 metric definitions without changing their geometry."""
    z, warm = z.double(), warm.to(z.device).double()
    if z.ndim != 2 or z.shape != warm.shape or len(z) <= z.shape[1]:
        raise ValueError("Matched banks larger than embedding dimension required")
    q, w = F.normalize(z, dim=1), F.normalize(warm, dim=1)
    centered = q - q.mean(0)
    covariance = centered.T @ centered / (len(q) - 1)
    offdiag = covariance - torch.diag(torch.diagonal(covariance))
    singular = torch.linalg.svdvals(z)
    p = singular / singular.sum().clamp_min(1e-12) + 1e-7
    a, b = z - z.mean(0), warm - warm.mean(0)
    cka = (a.T @ b).square().sum() / (torch.linalg.norm(a.T @ a) * torch.linalg.norm(b.T @ b)).clamp_min(1e-30)
    return {
        "variance": q.var(0, unbiased=True).mean(),
        "covariance": offdiag.square().sum() / z.shape[1],
        "rank": torch.exp(-(p * p.log()).sum()),
        "cka": 1 - cka,
        "cosine": 1 - (q * w).sum(1).mean(),
    }


def health_gradients(
    model: torch.nn.Module, bank: torch.Tensor, warm: torch.Tensor
) -> tuple[dict[str, list[torch.Tensor]], dict[str, float]]:
    """Read five parameter gradients without assigning .grad or changing learner state."""
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    modes = {name: module.training for name, module in model.named_modules()}
    device = next(model.parameters()).device
    gradients: dict[str, list[torch.Tensor]] = {}
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            z = torch.cat([model.encoder.embed(batch.to(device)) for batch in bank.split(32)])
            metrics = differentiable_health(z, warm)
            for i, name in enumerate(METRICS):
                raw = torch.autograd.grad(
                    metrics[name], parameters, retain_graph=i < len(METRICS) - 1, allow_unused=True
                )
                gradients[name] = [
                    torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, raw, strict=True)
                ]
                if not all(torch.isfinite(g).all() for g in gradients[name]):
                    raise ValueError("Nonfinite health derivative: " + name)
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    return gradients, {name: float(value.detach()) for name, value in metrics.items()}


def project_displacement(
    gradients: dict[str, list[torch.Tensor]],
    names: list[str],
    after: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    parent: dict[str, float],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Project known optimizer displacements; save layer terms for arithmetic audits."""
    forecasts, terms = {}, {}
    for metric, vector in gradients.items():
        terms[metric] = [
            float(
                (
                    g.double()
                    * (
                        after["encoder." + name].to(g.device).double()
                        - baseline["encoder." + name].to(g.device).double()
                    )
                ).sum()
            )
            for name, g in zip(names, vector, strict=True)
        ]
        denominator = max(abs(parent[metric]), 1e-12) if metric in ("variance", "covariance", "rank") else 1.0
        forecasts[metric] = sum(terms[metric]) / denominator
    return forecasts, terms
