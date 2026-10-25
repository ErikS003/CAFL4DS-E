"""Support-labeled task gradients and isolated optimizer-aware SSL directions."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from cafl4ds.image_utility import FixedProbe, observation
from cafl4ds.ssl.base import SSLMethod


def encoder_vector(method: SSLMethod) -> torch.Tensor:
    """Flatten trainable encoder parameters in stable parameter order."""
    return torch.cat([p.detach().flatten() for p in method.encoder.parameters() if p.requires_grad])


def head_logits(probe: FixedProbe, z: torch.Tensor) -> torch.Tensor:
    """Differentiable multiclass logistic head with frozen support normalization."""
    if len(probe.classifier.classes_) != 3:
        raise ValueError("This study requires the recorded three-class task")
    mean = torch.as_tensor(probe.mean, device=z.device)
    scale = torch.as_tensor(probe.scale, device=z.device)
    weight = torch.as_tensor(probe.classifier.coef_, device=z.device)
    bias = torch.as_tensor(probe.classifier.intercept_, device=z.device)
    return ((z.double() - mean) / scale) @ weight.T + bias


def task_gradient(
    method: SSLMethod,
    probe: FixedProbe,
    images: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Balanced CE gradient on support disjoint from fitting, without assigned .grad."""
    parameters = [p for p in method.encoder.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in parameters]
    classes = torch.as_tensor(probe.classifier.classes_)
    targets = (labels[:, None] == classes[None, :]).long().argmax(1)
    weights = torch.empty(len(labels), dtype=torch.float64)
    for domain in (0, 1):
        for label in classes:
            cell = (domains == domain) & (labels == label)
            if not cell.any():
                raise ValueError("Missing support domain/class")
            weights[cell] = 1.0 / (6 * int(cell.sum()))
    total = 0.0
    with observation(method, 113017), torch.enable_grad():
        for start in range(0, len(images), 64):
            stop = start + 64
            z = method.encoder.embed(images[start:stop])
            losses = F.cross_entropy(head_logits(probe, z), targets[start:stop], reduction="none")
            loss = (losses * weights[start:stop]).sum()
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            for acc, gradient in zip(accum, gradients, strict=True):
                if gradient is not None:
                    acc.add_(gradient.detach())
            total += float(loss.detach())
    vector = torch.cat([g.flatten() for g in accum])
    if not torch.isfinite(vector).all() or float(vector.norm()) == 0:
        raise ValueError("Nonfinite/zero task gradient")
    return vector, total


def ssl_directions(
    parent: SSLMethod, optimizer_state: dict[str, Any], selected: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One virtual AdamW step on copies; return clipped SGD and actual AdamW deltas."""
    method = copy.deepcopy(parent)
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-4, weight_decay=0.05)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    before = encoder_vector(method)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        method.train()
        optimizer.zero_grad()
        loss = method.training_step(selected)
        torch.autograd.backward(loss)
        norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0, error_if_nonfinite=True)
        gradient = torch.cat(
            [
                (torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
                for p in method.encoder.parameters()
                if p.requires_grad
            ]
        )
        optimizer.step()
    delta = encoder_vector(method) - before
    if not torch.isfinite(delta).all() or not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite proposed step")
    return -1e-4 * gradient, delta, {"ssl_loss": float(loss.detach()), "global_grad_norm": float(norm)}
