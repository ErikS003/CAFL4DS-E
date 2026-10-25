"""Causal unlabeled anchor sampling and isolated reconstruction-loss measurement."""

from __future__ import annotations

import torch


def reservoir_indices(arrivals: int, capacity: int, seed: int) -> list[int]:
    """Return Algorithm-R memory after observing only the preceding arrivals."""
    if arrivals < capacity or capacity < 1:
        raise ValueError("A positive reservoir must fit the observed prefix")
    generator = torch.Generator().manual_seed(seed)
    memory = list(range(capacity))
    for index in range(capacity, arrivals):
        slot = int(torch.randint(index + 1, (), generator=generator))
        if slot < capacity:
            memory[slot] = index
    return sorted(memory)


def private_loss(model: torch.nn.Module, images: torch.Tensor, seed: int | None) -> float:
    """Measure train-view loss with common private masks and restore modes/RNG."""
    if not len(images) or len(images) % 32:
        raise ValueError("Complete32-image batches required")
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
            model.train()
            with torch.no_grad():
                losses = [float(model.training_step(batch.to(device))) for batch in images.split(32)]
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if not all(torch.isfinite(torch.tensor(value)) for value in losses):
        raise ValueError("Nonfinite measurement")
    return sum(losses) / len(losses)
