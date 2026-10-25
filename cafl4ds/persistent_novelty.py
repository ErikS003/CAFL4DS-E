"""Fixed-budget admission of persistent feature changes with temporal exploration."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn.functional as F

from cafl4ds.short_buffer import synchronize
from cafl4ds.update_filtering import optimize_current

POLICIES = ("full", "uniform_half", "random_half", "persistent_half")
WINDOW = 64
HISTORY = 16
CONFIRMATION = 3
ANCHORS = torch.arange(3, WINDOW, 4)


def persistent_scores(features: torch.Tensor, history: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Score changes that stay displaced in a consistent direction for three arrivals.

    The reference is the normalized mean of the last sixteen accepted embeddings
    in this clip. A new clip bootstraps from its first arrived frame. Only the
    completed 64-frame window supplies confirmation; the final two positions have
    no complete confirmation and receive zero content priority.
    """
    if features.ndim != 2 or len(features) != WINDOW or not torch.isfinite(features).all():
        raise ValueError("Expected 64 finite feature vectors")
    if history is not None and (history.shape != (HISTORY, features.shape[1]) or not torch.isfinite(history).all()):
        raise ValueError("Invalid accepted-feature history")
    z = F.normalize(features, dim=1)
    reference = z[0] if history is None else F.normalize(history.mean(0), dim=0)
    residuals = z - reference
    magnitudes = residuals.norm(dim=1)
    directions = F.normalize(residuals, dim=1)
    first, second, third = directions[:-2], directions[1:-1], directions[2:]
    agreement = (
        torch.stack(((first * second).sum(1), (first * third).sum(1), (second * third).sum(1))).amin(0).clamp_min(0)
    )
    distance = torch.stack((magnitudes[:-2], magnitudes[1:-1], magnitudes[2:])).amin(0)
    scores = torch.zeros(WINDOW, device=z.device, dtype=z.dtype)
    scores[:-2] = distance * agreement
    return scores, reference


def persistent_indices(features: torch.Tensor, history: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reserve sixteen uniform anchors and fill sixteen remaining slots by persistence."""
    scores, reference = persistent_scores(features, history)
    eligible = torch.ones(WINDOW, device=features.device, dtype=torch.bool)
    eligible[ANCHORS.to(features.device)] = False
    eligible[-2:] = False  # content slots require three observed arrivals
    candidates = eligible.nonzero().flatten()
    order = torch.argsort(scores[candidates], descending=True, stable=True)
    content = candidates[order[:16]]
    indices = torch.cat((ANCHORS.to(features.device), content)).sort().values.cpu()
    return indices, {
        "scores": scores.cpu().tolist(),
        "content_positions": content.sort().values.cpu().tolist(),
        "reference": reference.cpu().tolist(),
        "reference_from_history": history is not None,
    }


def select_window(
    policy: str, features: torch.Tensor | None, history: torch.Tensor | None, generator: torch.Generator
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Use private selection RNG and deterministic chronological ties."""
    if policy == "full":
        return torch.arange(64), {}
    if policy == "uniform_half":
        return torch.arange(1, 64, 2), {}
    if policy == "random_half":
        return torch.randperm(64, generator=generator)[:32].sort().values, {}
    if policy != "persistent_half" or features is None:
        raise ValueError("Unknown policy or missing features")
    return persistent_indices(features, history)


def run_stream(
    model, optimizer, arrivals: torch.Tensor, policy: str, reference, seed: int, frames_per_clip: int = 128
) -> dict[str, Any]:
    """Observe one complete FIFO window; train selected frames once; reset at clip boundaries."""
    if (
        policy not in POLICIES
        or arrivals.device.type != "cpu"
        or len(arrivals) % frames_per_clip
        or frames_per_clip % WINDOW
    ):
        raise ValueError("Require complete host-resident clips and a known policy")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    history, history_indices = None, []
    records = []
    for start in range(0, len(arrivals), WINDOW):
        reset = start % frames_per_clip == 0
        if reset:
            history, history_indices = None, []
        synchronize(device)
        tick = time.perf_counter()
        window = arrivals[start : start + WINDOW].clone()
        gather_seconds = time.perf_counter() - tick
        feature_seconds = 0.0
        features = None
        if policy == "persistent_half":
            feature_tick = time.perf_counter()
            with torch.no_grad():
                features = torch.cat([reference.embed(batch.to(device)) for batch in window.split(32)])
            synchronize(device)
            feature_seconds = time.perf_counter() - feature_tick
        selection_tick = time.perf_counter()
        indices, detail = select_window(policy, features, history, generator)
        selected = window[indices]
        if policy == "persistent_half":
            history = F.normalize(features[indices.to(device)][-HISTORY:].detach(), dim=1).clone()
            next_history_indices = (indices[-HISTORY:] + start).tolist()
        else:
            next_history_indices = []
        synchronize(device)
        selection_seconds = time.perf_counter() - selection_tick
        train_tick = time.perf_counter()
        losses = [optimize_current(model, optimizer, batch.to(device), True) for batch in selected.split(32)]
        synchronize(device)
        training_seconds = time.perf_counter() - train_tick
        records.append(
            {
                "start": start,
                "end": start + WINDOW,
                "clip_index": start // frames_per_clip,
                "clip_reference_reset": reset,
                "selected_indices": (indices + start).tolist(),
                "reference_source_indices": history_indices,
                "detail": detail,
                "feature_frames": 64 if features is not None else 0,
                "accepted_frames": len(selected),
                "updates": len(losses),
                "loss_gradients": losses,
                "gather_seconds": gather_seconds,
                "feature_seconds": feature_seconds,
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "online_seconds": time.perf_counter() - tick,
            }
        )
        history_indices = next_history_indices
    accepted = sum(r["accepted_frames"] for r in records)
    if accepted != (len(arrivals) if policy == "full" else len(arrivals) // 2):
        raise ValueError("Wrong admission budget")
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": accepted,
        "updates": sum(r["updates"] for r in records),
        "online_seconds": sum(r["online_seconds"] for r in records),
        "selector_state": copy.deepcopy(
            {"generator": generator.get_state(), "history": history, "history_indices": history_indices}
        ),
    }
