"""Reference-bank covariance pressure and transactional, continuous rate search."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cafl4ds.ssl.base import SSLMethod


def geometry(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the float64 health statistic on unit-normalized embeddings."""
    q = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    centered = q - q.mean(0)
    covariance = centered.T @ centered / (len(q) - 1)
    offdiag = covariance - torch.diag(torch.diag(covariance))
    return (offdiag.square().sum() / z.shape[1], torch.diag(covariance).mean())


def regularizer(
    z: torch.Tensor,
    warm_covariance: float,
    warm_variance: float,
    settings: dict[str, Any],
    strength: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Push covariance below a soft target while protecting warm-relative spread."""
    covariance, variance = geometry(z)
    ratio = covariance / warm_covariance
    pressure = float(
        (
            (ratio.detach() - settings["covariance_target_ratio"])
            / (settings["covariance_warm_multiplier"] - settings["covariance_target_ratio"])
        ).clamp(0, 1)
    )
    weight = settings["covariance_weight"] * (1 + settings["pressure_gain"] * pressure)
    weight = weight if strength is None else strength
    cov_penalty = torch.relu(ratio - settings["covariance_target_ratio"]).square()
    var_penalty = torch.relu(settings["variance_floor_ratio"] - variance / warm_variance).square()
    loss = weight * cov_penalty + settings["variance_weight"] * var_penalty
    return (
        loss,
        {
            "covariance": float(covariance.detach()),
            "variance": float(variance.detach()),
            "covariance_weight": float(weight),
            "covariance_penalty": float(cov_penalty.detach()),
            "variance_penalty": float(var_penalty.detach()),
            "regularization_loss": float(loss.detach()),
        },
    )


def next_factor(factor: float, current: float, candidate: float, cap: float, safety: float = 0.8) -> float:
    """Reserve headroom using the last measured secant, with bounded backtracking."""
    if not np.isfinite([factor, current, candidate, cap, safety]).all():
        raise ValueError("Finite rate-search inputs required")
    if not 0 < factor <= 1 or not 0 < safety < 1 or current > cap + 1e-12:
        raise ValueError("Invalid rate-search state")
    headroom, increase = (max(0.0, cap - current), candidate - current)
    scale = safety * headroom / increase if increase > 0 else 0.25
    return float(factor * min(0.5, max(0.01, scale)))


def prepare_gradients(
    model: SSLMethod,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    bank: torch.Tensor,
    features: np.ndarray,
    reference: np.ndarray,
    seed: int,
    settings: dict[str, Any],
) -> tuple[list[torch.Tensor | None], dict[str, Any], dict[str, float]]:
    """Compute one SSL gradient plus an exact full-bank VJP in bounded chunks.

    Cached features must describe the current model on the identical fixed bank.
    The caller restores pre-image state before each optimizer trial. RNG after the
    single seeded SSL view is retained only if a candidate is committed.
    """
    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ssl = model.training_step(image)
    if not torch.isfinite(ssl):
        raise ValueError("Nonfinite SSL loss")
    ssl.backward()
    device = next(model.parameters()).device
    z = torch.as_tensor(features, device=device, dtype=torch.float64).detach().requires_grad_(True)
    penalty, diagnostics = regularizer(z, float(reference[2]), float(reference[1]), settings)
    dz = torch.autograd.grad(penalty, z)[0].detach()
    model.eval()
    width = settings["regularizer_batch_size"]
    if torch.count_nonzero(dz):
        for offset in range(0, len(bank), width):
            embedding = model.encoder.embed(bank[offset : offset + width].to(device))
            embedding.backward(dz[offset : offset + width].to(embedding.dtype))
    model.train()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
    diagnostics.update(ssl_loss=float(ssl.detach()), gradient_norm=float(norm))
    return (gradients, rng, diagnostics)


def apply_gradients(
    model: SSLMethod, optimizer: torch.optim.Optimizer, gradients: list[torch.Tensor | None], rng: dict[str, Any]
) -> None:
    """Take a genuine AdamW step with the shared seeded gradient at the chosen LR."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng["cpu"].cpu())
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.clone()
    optimizer.step()
