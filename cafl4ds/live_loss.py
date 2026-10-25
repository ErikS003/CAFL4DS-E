"""Live MAE difficulty admission with scoring isolated from training randomness."""

from __future__ import annotations

import copy
import time

import torch

from cafl4ds.short_buffer import synchronize
from cafl4ds.update_filtering import optimize_current

POLICIES = ("full", "uniform_half", "random_half", "live_loss_half")


def loss_indices(scores):
    if scores.shape != (64,) or not torch.isfinite(scores).all():
        raise ValueError("Require 64 finite losses")
    return torch.argsort(scores, descending=True, stable=True)[:32].sort().values.cpu()


def score_window(model, window, scoring_seed):
    """One fresh masked, lightly augmented MAE view per image in eval mode.

    fork_rng restores global CPU/CUDA training generators even on exceptions;
    all module modes are restored. MAE eval forwards have no mutable buffers.
    """
    device = next(model.parameters()).device
    modes = {name: module.training for name, module in model.named_modules()}
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(scoring_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(scoring_seed)
            model.eval()
            with torch.no_grad():
                scores = torch.cat([model.per_sample_loss(batch.to(device)) for batch in window.split(32)])
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    loss_indices(scores)
    return scores.detach().cpu()


def run_stream(model, optimizer, arrivals, policy, reference, seed, frames_per_clip=128):
    if (
        policy not in POLICIES
        or arrivals.device.type != "cpu"
        or len(arrivals) % frames_per_clip
        or frames_per_clip % 64
    ):
        raise ValueError("Require complete host-resident clips and a known policy")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    records = []
    for start in range(0, len(arrivals), 64):
        synchronize(device)
        tick = time.perf_counter()
        window = arrivals[start : start + 64].clone()
        gather_seconds = time.perf_counter() - tick
        feature_seconds = 0.0
        detail = {}
        scores = None
        if policy == "live_loss_half":
            score_tick = time.perf_counter()
            scoring_seed = seed + 72003 + start // 64
            scores = score_window(model, window, scoring_seed)
            synchronize(device)
            feature_seconds = time.perf_counter() - score_tick
            detail = {"scores": scores.tolist(), "scoring_seed": scoring_seed}
        select_tick = time.perf_counter()
        if policy == "full":
            indices = torch.arange(64)
        elif policy == "uniform_half":
            indices = torch.arange(1, 64, 2)
        elif policy == "random_half":
            indices = torch.randperm(64, generator=generator)[:32].sort().values
        else:
            indices = loss_indices(scores)
        selected = window[indices]
        synchronize(device)
        selection_seconds = time.perf_counter() - select_tick
        train_tick = time.perf_counter()
        losses = [optimize_current(model, optimizer, batch.to(device), True) for batch in selected.split(32)]
        synchronize(device)
        training_seconds = time.perf_counter() - train_tick
        records.append(
            {
                "start": start,
                "end": start + 64,
                "clip_index": start // frames_per_clip,
                "clip_reference_reset": start % frames_per_clip == 0,
                "selected_indices": (indices + start).tolist(),
                "reference_source_indices": [],
                "detail": detail,
                "feature_frames": 64 if scores is not None else 0,
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
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": sum(r["accepted_frames"] for r in records),
        "updates": sum(r["updates"] for r in records),
        "online_seconds": sum(r["online_seconds"] for r in records),
        "selector_state": copy.deepcopy(
            {"generator": generator.get_state(), "scoring_rule": "seed+72003+window_index"}
        ),
    }
