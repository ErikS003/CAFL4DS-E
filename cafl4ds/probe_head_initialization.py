"""Supervised head initialization with fixed fitting-only standardization."""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from cafl4ds.image_utility import FixedProbe
from cafl4ds.models.vit import TinyViTEncoder

ARMS = ("raw_random", "standardized_random", "standardized_probe")


class SupervisedHead(nn.Module):
    """Use fixed fitting-warm normalization so optimizer coordinates are explicit."""

    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, probe: FixedProbe, arm: str) -> None:
        """Only normalized arms use fitting statistics; probe arm loads fitted weights."""
        super().__init__()
        if arm not in ARMS or not np.array_equal(probe.classifier.classes_, [0, 1, 2]):
            raise ValueError("Require a declared arm and three canonical scene classes")
        self.linear = nn.Linear(len(probe.mean), 3)
        self.register_buffer(
            "mean", torch.tensor(probe.mean if arm != "raw_random" else np.zeros_like(probe.mean), dtype=torch.float32)
        )
        self.register_buffer(
            "scale",
            torch.tensor(probe.scale if arm != "raw_random" else np.ones_like(probe.scale), dtype=torch.float32),
        )
        if arm == "standardized_probe":
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(probe.classifier.coef_, dtype=torch.float32))
                self.linear.bias.copy_(torch.tensor(probe.classifier.intercept_, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the frozen normalization and trainable linear head."""
        return cast(torch.Tensor, self.linear((z - self.mean) / self.scale))


def train_snapshots(
    encoder: TinyViTEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    probe: FixedProbe,
    arm: str,
    seed: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict[int, tuple[TinyViTEncoder, SupervisedHead]], list[dict[str, Any]]]:
    """Matched private trajectories; caller encoder, probe and RNG stay unchanged."""
    if not checkpoints or min(checkpoints) < 0:
        raise ValueError("Require nonnegative checkpoint epochs")
    snapshots = {}
    trace = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trial = copy.deepcopy(encoder)
        trial.requires_grad_(True)
        head = SupervisedHead(probe, arm)
        if 0 in checkpoints:
            snapshots[0] = (copy.deepcopy(trial).eval(), copy.deepcopy(head).eval())
        optimizer = torch.optim.AdamW(
            [{"params": trial.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}],
            weight_decay=0.01,
        )
        generator = torch.Generator().manual_seed(seed)
        trial.train()
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


def select_head(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select primary internal error; neither head CE nor query data enter selection."""
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row["arm"], int(row["epoch"])), []).append(float(row["linear_error_balanced"]))
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    best = min(means.values())
    arm, epoch = min(
        (key for key, value in means.items() if value <= best + 1e-10), key=lambda key: (key[1], ARMS.index(key[0]))
    )
    return {"arm": arm, "epoch": epoch, "internal_mean_error": means[(arm, epoch)]}
