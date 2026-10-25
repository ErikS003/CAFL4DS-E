"""Bounded-window label-free frame coverage with explicit accepted-frame budgets."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias

from cafl4ds.ssl.base import SSLMethod
from cafl4ds.update_filtering import optimize_current

WINDOW = 64
RETAIN = 32
RANDOM_SEED_OFFSET = 62003


def content_descriptor(images: torch.Tensor) -> torch.Tensor:
    """Use centered pooled RGB structure rather than labels or learner features.

    Per-channel spatial centering removes mean brightness/color; L2 normalization
    removes global contrast scale. Flat images map to zero rather than NaN.
    """
    if images.ndim != 4 or images.shape[1:] != (3, 64, 64) or not torch.isfinite(images).all():
        raise ValueError("expected finite N x 3 x 64 x 64 frames")
    pooled = F.avg_pool2d(images, 4)
    pooled = pooled - pooled.mean(dim=(2, 3), keepdim=True)
    return F.normalize(pooled.flatten(1), dim=1, eps=1e-6)


def coverage_indices(images: torch.Tensor, retain: int = RETAIN) -> torch.Tensor:
    """Select diverse observed frames with deterministic greedy farthest-first coverage.

    Seed with the first observed frame. Repeatedly maximize distance to the nearest
    already selected descriptor; ties prefer the earliest remaining frame. Return
    selected positions chronologically. Nothing after this window is inspected.
    """
    if not 1 <= retain <= len(images):
        raise ValueError("retain must be between one and the observed window length")
    descriptors = content_descriptor(images)
    distances = torch.cdist(descriptors, descriptors, compute_mode="donot_use_mm_for_euclid_dist")
    chosen = [0]
    nearest = distances[0].clone()
    nearest[0] = -1
    for _ in range(retain - 1):
        candidate = int(nearest.argmax())
        chosen.append(candidate)
        nearest = torch.minimum(nearest, distances[candidate])
        nearest[chosen] = -1
    return torch.tensor(sorted(chosen), dtype=torch.long, device=images.device)


def select_window(policy: str, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Select only from the completed 64-frame FIFO window without learner RNG draws."""
    if len(images) != WINDOW:
        raise ValueError("expected exactly 64 observed frames")
    if policy == "full":
        return torch.arange(WINDOW)
    if policy == "fixed_half":
        return torch.arange(RETAIN, WINDOW)
    if policy == "random_half":
        return torch.randperm(WINDOW, generator=generator)[:RETAIN].sort().values
    if policy == "coverage_half":
        return coverage_indices(images)
    raise ValueError(f"unknown policy: {policy}")


def optimize_window(
    model: SSLMethod, optimizer: torch.optim.Optimizer, policy: str,
    images: torch.Tensor, start: int, generator: torch.Generator,
) -> dict[str, Any]:
    """Charge resident FIFO gathering, selection and every accepted-frame update."""
    tick = time.perf_counter()
    window = images[start:start + WINDOW]
    selection_tick = time.perf_counter()
    indices = select_window(policy, window, generator)
    selection_seconds = time.perf_counter() - selection_tick
    selected = window[indices]
    training_tick = time.perf_counter()
    losses = [optimize_current(model, optimizer, selected[i:i + 32], True) for i in range(0, len(selected), 32)]
    ssl_seconds = time.perf_counter() - training_tick
    online_seconds = time.perf_counter() - tick
    return {"start_frame": start, "end_frame_exclusive": start + WINDOW,
            "selected_positions": indices.tolist(), "selected_global_indices": (indices + start).tolist(),
            "accepted_frames": len(selected), "optimizer_updates": len(losses), "loss_gradients": losses,
            "selection_seconds": selection_seconds, "ssl_seconds": ssl_seconds, "online_seconds": online_seconds}
