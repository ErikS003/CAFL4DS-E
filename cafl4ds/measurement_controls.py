"""Supervised measurement controls with no access to evaluation images or labels."""

from __future__ import annotations

import copy

import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder


def shuffle_within_domain(labels: torch.Tensor, domains: torch.Tensor, seed: int) -> torch.Tensor:
    """Destroy scene correspondence while retaining domain-specific label counts."""
    result = labels.clone()
    generator = torch.Generator().manual_seed(seed)
    for domain in torch.unique(domains, sorted=True):
        indices = torch.where(domains == domain)[0]
        result[indices] = labels[indices[torch.randperm(len(indices), generator=generator)]]
    return result


def train_control(
    encoder: TinyViTEncoder, images: torch.Tensor, labels: torch.Tensor, seed: int, epochs: int = 30
) -> tuple[TinyViTEncoder, nn.Linear, list[dict[str, float | int]]]:
    """Fit a private encoder/head copy on the supplied training images only, at fixed work."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trial = copy.deepcopy(encoder)
        trial.requires_grad_(True)
        trial.train()
        head = nn.Linear(trial.embed_dim, 3)
        optimizer = torch.optim.AdamW(
            [{"params": trial.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}],
            weight_decay=0.01,
        )
        generator = torch.Generator().manual_seed(seed)
        records: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            order = torch.randperm(len(images), generator=generator)
            total = 0.0
            correct = 0
            updates = 0
            for indices in order.split(32):  # type: ignore[no-untyped-call]
                optimizer.zero_grad(set_to_none=True)
                logits = head(trial.embed(images[indices]))
                loss = nn.functional.cross_entropy(logits, labels[indices])
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                correct += int((logits.detach().argmax(1) == labels[indices]).sum())
                updates += 1
            records.append(
                {
                    "epoch": epoch + 1,
                    "online_fit_ce": total / len(images),
                    "online_fit_accuracy": correct / len(images),
                    "updates": updates,
                }
            )
        trial.eval()
        head.eval()
        return trial, head, records
