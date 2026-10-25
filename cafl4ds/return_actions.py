"""Paired continuation admission actions, including a genuine zero-update hold."""

from __future__ import annotations

import time

import torch

from cafl4ds.regime_windows import run_stream
from cafl4ds.short_buffer import capture

ACTIONS = ("hold", "uniform_half", "full")
CHECKPOINTS = (256, 4224)


def run_branch(model, optimizer, arrivals, action, seed, checkpoints=CHECKPOINTS, observe=None):
    if action not in ACTIONS or len(arrivals) % 64 or arrivals.device.type != "cpu":
        raise ValueError("Invalid action or incomplete host arrivals")
    if action != "hold":
        return run_stream(model, optimizer, arrivals, action, 32, seed, checkpoints, observe)
    generator = torch.Generator().manual_seed(seed + 62003)
    records = []
    observations = {}
    for start in range(0, len(arrivals), 32):
        tick = time.perf_counter()
        window = arrivals[start : start + 32].clone()
        gather = time.perf_counter() - tick
        clock = start + 32
        records.append(
            {
                "start": start,
                "end": clock,
                "selected_indices": [],
                "exposure_indices": [],
                "queued_indices": [],
                "scores": None,
                "scoring_seed": None,
                "updates": 0,
                "loss_gradients": [],
                "gather_seconds": gather,
                "score_seconds": 0.0,
                "selection_seconds": 0.0,
                "training_seconds": 0.0,
                "online_seconds": time.perf_counter() - tick,
            }
        )
        if clock in checkpoints:
            empty = window[:0].clone()
            state = {
                "learner": capture(model, optimizer),
                "selector_rng": generator.get_state().clone(),
                "arrivals": clock,
                "updates": 0,
                "accepted_frames": 0,
                "window_start": clock,
                "pending_indices": [],
                "pending_images": empty,
                "queued_indices": [],
                "queued_images": empty.clone(),
            }
            observations[clock] = None if observe is None else observe(clock, state)
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": 0,
        "updates": 0,
        "online_seconds": sum(r["online_seconds"] for r in records),
        "observations": observations,
    }
