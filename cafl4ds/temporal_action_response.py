"""Isolated temporal full/uniform-half actions from complete causal parent states."""

from __future__ import annotations

import copy
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cafl4ds.ssl.base import SSLMethod
from cafl4ds.temporal_anchored_filter import optimize_window


def snapshot(model: SSLMethod, optimizer: torch.optim.Optimizer, updates: int, arrivals: int) -> dict[str, Any]:
    """Capture independent model, Adam, RNG, mode and arrival/update clocks."""
    return copy.deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "training": model.training,
            "arrivals": arrivals,
            "optimizer_updates": updates,
        }
    )


def restore(
    initial: SSLMethod,
    config: DictConfig,
    parent: dict[str, Any],
) -> tuple[SSLMethod, torch.optim.Optimizer]:
    """Restore without sharing model or optimizer tensors with the parent."""
    model = copy.deepcopy(initial)
    model.load_state_dict(parent["model"], strict=True)
    optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
    optimizer.load_state_dict(copy.deepcopy(parent["optimizer"]))
    model.train(parent["training"])
    torch.set_rng_state(parent["torch_rng"].clone())
    return model, optimizer


def run_action(
    model: SSLMethod,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    offset: int,
    action: str,
    horizon: int = 8,
) -> list[dict[str, Any]]:
    """Process identical arrived windows at full or uniformly interleaved half work."""
    if action not in {"full", "half"} or horizon <= 0 or offset < 0 or (offset + horizon) * 64 > len(images):
        raise ValueError("invalid temporal action window")
    generator = torch.Generator().manual_seed(0)
    policy = "full" if action == "full" else "uniform_half"
    return [
        optimize_window(model, optimizer, policy, images, window * 64, generator)
        for window in range(offset, offset + horizon)
    ]
