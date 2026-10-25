"""Training-only optimization diagnostics for a small balanced memorization task."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder

ARMS = {
    "head_raw": (False, False, 0.0, 0.01),
    "head_standardized": (False, True, 0.0, 0.01),
    "joint_original_lr": (True, False, 0.0001, 0.001),
    "joint_higher_lr": (True, False, 0.001, 0.01),
}


def balanced_subset(labels: torch.Tensor, domains: torch.Tensor, seed: int = 75011) -> torch.Tensor:
    """Draw six images in each of six cells using a private generator."""
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for domain in (0, 1):
        for label in (0, 1, 2):
            cell = torch.where((domains == domain) & (labels == label))[0]
            if len(cell) < 6:
                raise ValueError("Require six fitting images per cell")
            selected.append(cell[torch.randperm(len(cell), generator=generator)[:6]])
    return torch.cat(selected)


def norm(parameters: list[torch.Tensor]) -> float:
    """Measure a finite Euclidean gradient/parameter norm."""
    return math.sqrt(sum(float(p.detach().double().square().sum()) for p in parameters))


def fitting_run(
    encoder: TinyViTEncoder, images: torch.Tensor, labels: torch.Tensor, arm: str, seed: int, steps: int = 300
) -> tuple[TinyViTEncoder, nn.Linear, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Optimize a private copy without receiving any validation/query observations."""
    joint, standardize, encoder_lr, head_lr = ARMS[arm]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trial = copy.deepcopy(encoder)
        trial.requires_grad_(joint)
        trial.eval()
        head = nn.Linear(trial.embed_dim, 3)
        initial_encoder = [p.detach().clone() for p in trial.parameters()]
        initial_head = [p.detach().clone() for p in head.parameters()]
        with torch.no_grad():
            cached = trial.embed(images).detach()
        mean = cached.mean(0) if standardize else torch.zeros(cached.shape[1])
        scale = cached.std(0, unbiased=False) if standardize else torch.ones(cached.shape[1])
        scale[scale < 1e-8] = 1
        groups = [{"params": head.parameters(), "lr": head_lr}]
        if joint:
            groups.append({"params": trial.parameters(), "lr": encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        trace: list[dict[str, Any]] = []

        def record(step: int, encoder_gradient: float, head_gradient: float) -> None:
            with torch.no_grad():
                z = trial.embed(images) if joint else cached
                logits = head((z - mean) / scale)
                value = float(nn.functional.cross_entropy(logits, labels))
                accuracy = float((logits.argmax(1) == labels).double().mean())
            delta_encoder = norm([p.detach() - q for p, q in zip(trial.parameters(), initial_encoder, strict=True)])
            delta_head = norm([p.detach() - q for p, q in zip(head.parameters(), initial_head, strict=True)])
            trace.append(
                {
                    "step": step,
                    "fit_ce": value,
                    "fit_accuracy": accuracy,
                    "encoder_gradient_norm": encoder_gradient,
                    "head_gradient_norm": head_gradient,
                    "encoder_parameter_delta": delta_encoder,
                    "head_parameter_delta": delta_head,
                }
            )

        record(0, 0, 0)
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            z = trial.embed(images) if joint else cached
            logits = head((z - mean) / scale)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fitting loss")
            loss.backward()  # type: ignore[no-untyped-call]
            encoder_gradient = norm([p.grad for p in trial.parameters() if p.grad is not None])
            head_gradient = norm([p.grad for p in head.parameters() if p.grad is not None])
            if not math.isfinite(encoder_gradient + head_gradient):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            if step % 10 == 0 or step == steps:
                record(step, encoder_gradient, head_gradient)
        return trial, head, mean, scale, trace
