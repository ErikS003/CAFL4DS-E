"""Private supervised controls and internal-only checkpoint selection."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder


def control_snapshots(
    encoder: TinyViTEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    learning_rate: float,
    checkpoints: tuple[int, ...],
) -> tuple[dict[int, tuple[TinyViTEncoder, nn.Linear]], list[dict[str, Any]]]:
    """Train on supplied fitting data only, preserving caller model and global RNG."""
    if learning_rate <= 0 or not checkpoints or min(checkpoints) < 1:
        raise ValueError("Require positive learning rate and epochs")
    snapshots = {}
    trace = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trial = copy.deepcopy(encoder)
        trial.requires_grad_(True)
        trial.train()
        head = nn.Linear(trial.embed_dim, 3)
        optimizer = torch.optim.AdamW(
            [{"params": trial.parameters(), "lr": learning_rate}, {"params": head.parameters(), "lr": 1e-3}],
            weight_decay=0.01,
        )
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, max(checkpoints) + 1):
            total, correct, updates = 0.0, 0, 0
            for indices in torch.randperm(len(images), generator=generator).split(32):  # type: ignore[no-untyped-call]
                optimizer.zero_grad(set_to_none=True)
                logits = head(trial.embed(images[indices]))
                loss = nn.functional.cross_entropy(logits, labels[indices])
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                correct += int((logits.detach().argmax(1) == labels[indices]).sum())
                updates += 1
            trace.append(
                {
                    "epoch": epoch,
                    "online_fit_ce": total / len(images),
                    "online_fit_accuracy": correct / len(images),
                    "updates": updates,
                }
            )
            if epoch in checkpoints:
                snapshots[epoch] = (copy.deepcopy(trial).eval(), copy.deepcopy(head).eval())
    return snapshots, trace


def select_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by internal error alone, tie-breaking toward less encoder work."""
    groups: dict[tuple[float, int], list[float]] = {}
    for row in rows:
        groups.setdefault((float(row["learning_rate"]), int(row["epoch"])), []).append(
            float(row["linear_error_balanced"])
        )
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    best = min(means.values())
    lr, epoch = min((key for key, value in means.items() if value <= best + 1e-10), key=lambda key: (key[1], key[0]))
    return {"learning_rate": lr, "epoch": epoch, "internal_mean_error": means[(lr, epoch)]}
