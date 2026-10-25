"""Separate distinct training-frame coverage from repeated current-window exposure."""

from __future__ import annotations

import copy
import time

import torch

from cafl4ds.short_buffer import synchronize
from cafl4ds.update_filtering import optimize_current

POLICIES = ("full", "uniform_half", "random_half", "uniform_twice", "random_twice")


def select_window(policy, generator):
    if policy == "full":
        return torch.arange(64)
    if policy in ("uniform_half", "uniform_twice"):
        return torch.arange(1, 64, 2)
    if policy in ("random_half", "random_twice"):
        return torch.randperm(64, generator=generator)[:32].sort().values
    raise ValueError("Unknown policy")


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
        select_tick = time.perf_counter()
        indices = select_window(policy, generator)
        selected = window[indices]
        batches = list(selected.split(32))
        if policy.endswith("twice"):
            batches = batches * 2
        synchronize(device)
        selection_seconds = time.perf_counter() - select_tick
        train_tick = time.perf_counter()
        losses = []
        stochastic_draws = []
        for batch in batches:
            before_cpu = torch.get_rng_state()
            before_device = torch.cuda.get_rng_state(device) if device.type == "cuda" else before_cpu
            losses.append(optimize_current(model, optimizer, batch.to(device), True))
            after_device = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
            stochastic_draws.append(not torch.equal(before_device, after_device))
        synchronize(device)
        training_seconds = time.perf_counter() - train_tick
        records.append(
            {
                "start": start,
                "end": start + 64,
                "clip_index": start // frames_per_clip,
                "selected_indices": (indices + start).tolist(),
                "exposure_indices": [(indices + start).tolist()] * 2
                if policy.endswith("twice")
                else [(part + start).tolist() for part in indices.split(32)],
                "accepted_frames": len(selected),
                "training_exposures": sum(len(b) for b in batches),
                "updates": len(losses),
                "loss_gradients": losses,
                "fresh_stochastic_draws": stochastic_draws,
                "gather_seconds": gather_seconds,
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "online_seconds": time.perf_counter() - tick,
            }
        )
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": sum(r["accepted_frames"] for r in records),
        "training_exposures": sum(r["training_exposures"] for r in records),
        "updates": sum(r["updates"] for r in records),
        "online_seconds": sum(r["online_seconds"] for r in records),
        "selector_state": copy.deepcopy({"generator": generator.get_state()}),
    }
