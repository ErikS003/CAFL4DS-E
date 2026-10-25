"""Cross-fitted labeled observations of replay effects after head adaptation."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from cafl4ds.image_utility import FixedProbe, balanced_losses
from cafl4ds.replay_task_alignment import encoder_vector, head_logits
from cafl4ds.ssl.base import SSLMethod


def support_folds(labels: torch.Tensor, domains: torch.Tensor, pooled: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Partition six balanced cells into six192-image folds with960 fitting labels."""
    generator = torch.Generator().manual_seed(121019)
    cells = []
    if len(pooled) != 1152 or len(torch.unique(pooled)) != 1152:
        raise ValueError("Require1152 unique support labels")
    for domain in (0, 1):
        for label in (0, 1, 2):
            cell = pooled[(domains[pooled] == domain) & (labels[pooled] == label)]
            if len(cell) != 192:
                raise ValueError("Require192 support images per domain/class")
            cells.append(cell[torch.randperm(len(cell), generator=generator)])
    result = []
    for fold in range(6):
        heldout = torch.cat([cell[fold * 32 : (fold + 1) * 32] for cell in cells]).sort().values
        fitting = pooled[~torch.isin(pooled, heldout)].sort().values
        result.append({"fitting": fitting, "heldout": heldout})
    return result


def ce(probe: FixedProbe, z: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, ix: torch.Tensor) -> float:
    """Stable balanced CE of a specified frozen or newly fitted logistic head."""
    classes = torch.as_tensor(probe.classifier.classes_)
    targets = (labels[ix, None] == classes[None, :]).long().argmax(1)
    losses = F.cross_entropy(head_logits(probe, z[ix]), targets, reduction="none")
    return balanced_losses(losses, labels[ix], domains[ix])["balanced"]


def virtual_step(
    parent: SSLMethod, optimizer_state: dict[str, Any], selected: torch.Tensor, seed: int
) -> tuple[SSLMethod, torch.Tensor, dict[str, float]]:
    """Return an isolated copied AdamW-updated model and its encoder displacement."""
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
        optimizer.step()
    delta = encoder_vector(method) - before
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite copied step")
    return method, delta, {"loss": float(loss.detach()), "grad_norm": float(norm)}
