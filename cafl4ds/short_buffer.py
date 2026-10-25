"""Causal short-buffer selection with charged CUDA work and complete learner state."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from cafl4ds.update_filtering import optimize_current

ARMS = ("frozen", "full", "uniform_half", "buffered_random_half", "buffered_diverse_half")


def synchronize(device: torch.device) -> None:
    """Include completed accelerator work in wall-clock measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Copy all parameters, optimizer buffers, module modes and CPU/CUDA RNG state."""
    return copy.deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "modes": {name: module.training for name, module in model.named_modules()},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    )


def restore(initial: torch.nn.Module, optim_config: Any, state: dict[str, Any]):
    """Restore independent tensors and every RNG after constructing the optimizer."""
    model = copy.deepcopy(initial)
    model.load_state_dict(state["model"], strict=True)
    optimizer = instantiate(optim_config, params=[p for p in model.parameters() if p.requires_grad])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    for name, module in model.named_modules():
        module.training = state["modes"][name]
    torch.set_rng_state(state["cpu_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return model, optimizer


def diverse_indices(features: torch.Tensor, retain: int = 64) -> torch.Tensor:
    """Farthest-first normalized-feature coverage, oldest seed and earliest ties."""
    if features.ndim != 2 or not 1 <= retain <= len(features) or not torch.isfinite(features).all():
        raise ValueError("Require finite features and a valid retained count")
    normalized = F.normalize(features, dim=1)
    distances = torch.cdist(normalized, normalized, compute_mode="donot_use_mm_for_euclid_dist")
    chosen = [0]
    nearest = distances[0].clone()
    nearest[0] = -1
    for _ in range(retain - 1):
        index = int(nearest.argmax())
        chosen.append(index)
        nearest = torch.minimum(nearest, distances[index])
        nearest[chosen] = -1
    return torch.tensor(sorted(chosen), dtype=torch.long)


def run_episode(model, optimizer, arrivals: torch.Tensor, arm: str, reference, seed: int) -> dict[str, Any]:
    """Train each selected frame once; no window inspects subsequent arrivals."""
    if arm not in ARMS or len(arrivals) % 128 or arrivals.device.type != "cpu":
        raise ValueError("Require a known arm and complete host-resident 128-frame blocks")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    if arm == "frozen":
        return {
            "blocks": [],
            "arrivals": len(arrivals),
            "accepted_frames": 0,
            "updates": 0,
            "online_seconds": 0.0,
            "selector_rng": generator.get_state(),
        }
    width = 32 if arm == "full" else 64 if arm == "uniform_half" else 128
    records = []
    for start in range(0, len(arrivals), width):
        synchronize(device)
        tick = time.perf_counter()
        window = arrivals[start : start + width].clone()
        gather_seconds = time.perf_counter() - tick
        feature_seconds = 0.0
        feature_count = 0
        if arm == "buffered_diverse_half":
            feature_tick = time.perf_counter()
            with torch.no_grad():
                z = torch.cat([reference.embed(batch.to(device)) for batch in window.split(32)])
            synchronize(device)
            feature_seconds = time.perf_counter() - feature_tick
            feature_count = width
        selection_tick = time.perf_counter()
        if arm == "full":
            indices = torch.arange(width)
        elif arm == "uniform_half":
            indices = torch.arange(1, width, 2)
        elif arm == "buffered_random_half":
            indices = torch.randperm(width, generator=generator)[:64].sort().values
        else:
            indices = diverse_indices(z)
        selected = window[indices]
        synchronize(device)
        selection_seconds = time.perf_counter() - selection_tick
        training_tick = time.perf_counter()
        updates = [optimize_current(model, optimizer, batch.to(device), True) for batch in selected.split(32)]
        synchronize(device)
        training_seconds = time.perf_counter() - training_tick
        records.append(
            {
                "start": start,
                "end": start + width,
                "selected_indices": (indices + start).tolist(),
                "feature_frames": feature_count,
                "accepted_frames": len(selected),
                "updates": len(updates),
                "loss_gradients": updates,
                "gather_seconds": gather_seconds,
                "feature_seconds": feature_seconds,
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "online_seconds": time.perf_counter() - tick,
            }
        )
    accepted = sum(row["accepted_frames"] for row in records)
    expected = len(arrivals) if arm == "full" else len(arrivals) // 2
    if accepted != expected:
        raise ValueError("Frame budget mismatch")
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": accepted,
        "updates": sum(row["updates"] for row in records),
        "online_seconds": sum(row["online_seconds"] for row in records),
        "selector_rng": generator.get_state(),
    }
