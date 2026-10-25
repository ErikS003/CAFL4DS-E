"""Temporal anchors plus local pooled-RGB change at exactly half acceptance."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from cafl4ds.ssl.base import SSLMethod
from cafl4ds.temporal_content_filter import select_window as original_select
from cafl4ds.update_filtering import optimize_current

RANDOM_SEED_OFFSET = 62003


def change_indices(images: torch.Tensor) -> torch.Tensor:
    """Keep each block's latest frame and earliest maximum local-change candidate."""
    if images.shape != (64, 3, 64, 64) or not torch.isfinite(images).all():
        raise ValueError("expected finite 64 x 3 x 64 x 64 frames")
    pooled = F.avg_pool2d(images, 4)
    scores = torch.zeros(64, device=images.device, dtype=images.dtype)
    scores[1:] = (pooled[1:] - pooled[:-1]).abs().mean(dim=(1, 2, 3))
    starts = torch.arange(0, 64, 4, device=images.device)
    candidates = starts + scores.reshape(16, 4)[:, :3].argmax(dim=1)
    return torch.stack((candidates, starts + 3), dim=1).flatten()


def select_window(policy: str, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Select only after 64 arrivals; uniform phase and tie rules are fixed."""
    if len(images) != 64:
        raise ValueError("expected exactly 64 observed frames")
    if policy == "uniform_half":
        return torch.arange(1, 64, 2)
    if policy == "anchored_half":
        return change_indices(images)
    if policy not in {"full", "fixed_half", "random_half"}:
        raise ValueError(f"unknown policy: {policy}")
    return original_select(policy, images, generator)


def optimize_window(
    model: SSLMethod, optimizer: torch.optim.Optimizer, policy: str,
    images: torch.Tensor, start: int, generator: torch.Generator,
) -> dict[str, Any]:
    """Measure FIFO gathering, selection, assembly and every actual SSL update."""
    tick = time.perf_counter()
    window = images[start:start + 64]
    selection_tick = time.perf_counter()
    indices = select_window(policy, window, generator)
    selection_seconds = time.perf_counter() - selection_tick
    selected = window[indices]
    training_tick = time.perf_counter()
    losses = [optimize_current(model, optimizer, selected[i:i + 32], True) for i in range(0, len(selected), 32)]
    ssl_seconds = time.perf_counter() - training_tick
    online_seconds = time.perf_counter() - tick
    return {"start_frame": start, "end_frame_exclusive": start + 64,
            "selected_positions": indices.tolist(), "selected_global_indices": (indices + start).tolist(),
            "accepted_frames": len(selected), "optimizer_updates": len(losses), "loss_gradients": losses,
            "selection_seconds": selection_seconds, "ssl_seconds": ssl_seconds, "online_seconds": online_seconds}
