"""Isolate head optimization from encoder coordinate changes."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from cafl4ds.image_utility import FixedProbe
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.probe_head_initialization import SupervisedHead

RATES = {"head_only": 0.0, "gentle_1e6": 1e-6, "gentle_1e5": 1e-5, "reference_1e4": 1e-4}
ARMS = tuple(RATES)


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
    if arm not in RATES:
        raise ValueError("Unknown encoder rate")
    snapshots = {}
    trace = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trial = copy.deepcopy(encoder)
        trial.requires_grad_(RATES[arm] > 0)
        head = SupervisedHead(probe, "standardized_probe")
        if 0 in checkpoints:
            snapshots[0] = (copy.deepcopy(trial).eval(), copy.deepcopy(head).eval())
        groups = [{"params": head.parameters(), "lr": 1e-3}]
        if RATES[arm] > 0:
            groups.insert(0, {"params": trial.parameters(), "lr": RATES[arm]})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        generator = torch.Generator().manual_seed(seed)
        trial.train(RATES[arm] > 0)
        for epoch in range(1, max(checkpoints) + 1):
            total, correct, updates = 0.0, 0, 0
            for indices in torch.randperm(len(images), generator=generator).split(32):  # type: ignore[no-untyped-call]
                optimizer.zero_grad(set_to_none=True)
                if RATES[arm] == 0:
                    with torch.no_grad():
                        z = trial.embed(images[indices])
                else:
                    z = trial.embed(images[indices])
                logits = head(z)
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
                    "encoder_updates": updates if RATES[arm] > 0 else 0,
                }
            )
            if epoch in checkpoints:
                snapshots[epoch] = (copy.deepcopy(trial).eval(), copy.deepcopy(head).eval())
    return snapshots, trace


def select_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select primary internal error; neither head CE nor query data enter selection."""
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row["arm"], int(row["epoch"])), []).append(float(row["linear_error_balanced"]))
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    best = min(means.values())
    arm, epoch = min(
        (key for key, value in means.items() if value <= best + 1e-10), key=lambda key: (key[1], RATES[key[0]])
    )
    return {"arm": arm, "epoch": epoch, "internal_mean_error": means[(arm, epoch)]}
