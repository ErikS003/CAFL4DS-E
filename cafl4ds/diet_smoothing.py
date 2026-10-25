"""Fixed causal historical/current exposure schedules with matched training budgets."""

import time

import torch

from cafl4ds.return_actions import run_branch as baseline
from cafl4ds.short_buffer import capture, synchronize
from cafl4ds.update_filtering import optimize_current

ACTIONS = ("hold", "full", "smooth", "abrupt")
CHECKPOINTS = (256, 4224)
_history = None


def set_history(images):
    global _history
    if images.device.type != "cpu" or len(images) != 512:
        raise ValueError("History must contain512 host images")
    _history = images.clone()


def historical_count(update, action):
    if action == "smooth":
        return max(0, 16 - update // 2)
    if action == "abrupt":
        return 32 if update < 8 else 16 if update == 8 else 0
    raise ValueError("Unknown mixing schedule")


def run_branch(model, optimizer, arrivals, action, seed, checkpoints=CHECKPOINTS, observe=None):
    if action in ("hold", "full"):
        return baseline(model, optimizer, arrivals, action, seed, checkpoints, observe)
    if action not in ACTIONS or _history is None or arrivals.device.type != "cpu" or len(arrivals) % 64:
        raise ValueError("Invalid mixing input")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    records, observations = [], {}
    pointer = accepted = 0
    for update, start in enumerate(range(0, len(arrivals), 32)):
        synchronize(device)
        tick = time.perf_counter()
        count = historical_count(update, action)
        # Every historical item precedes the parent checkpoint; every new item has arrived.
        historical = [(pointer + j * 17) % 512 for j in range(count)]
        pointer += count * 17
        current = [start + ((j + 1) * 32 // (32 - count)) - 1 for j in range(32 - count)] if count < 32 else []
        batch = torch.cat((_history[historical], arrivals[current])).clone()
        selection = time.perf_counter() - tick
        train_tick = time.perf_counter()
        loss = optimize_current(model, optimizer, batch.to(device), True)
        synchronize(device)
        training = time.perf_counter() - train_tick
        accepted += len(current)
        clock = start + 32
        records.append(
            {
                "start": start,
                "end": clock,
                "selected_indices": current,
                "exposure_indices": [[-j - 1 for j in historical] + current],
                "queued_indices": [],
                "updates": 1,
                "loss_gradients": [loss],
                "history_indices": historical,
                "historical_count": count,
                "selection_seconds": selection,
                "training_seconds": training,
                "online_seconds": time.perf_counter() - tick,
            }
        )
        if clock in checkpoints:
            empty = arrivals[:0].clone()
            state = {
                "learner": capture(model, optimizer),
                "selector_rng": generator.get_state().clone(),
                "arrivals": clock,
                "updates": update + 1,
                "accepted_frames": accepted,
                "window_start": clock,
                "pending_indices": [],
                "pending_images": empty,
                "queued_indices": [],
                "queued_images": empty.clone(),
                "history_images": _history.clone(),
                "history_pointer": pointer,
            }
            observations[clock] = None if observe is None else observe(clock, state)
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": accepted,
        "updates": len(records),
        "online_seconds": sum(r["online_seconds"] for r in records),
        "observations": observations,
    }
