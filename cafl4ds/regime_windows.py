"""Causal variable-window admission across sustained regimes, without boundary resets."""

from __future__ import annotations

import time

import torch

from cafl4ds.short_buffer import capture, synchronize
from cafl4ds.update_filtering import optimize_current

POLICIES = ("full", "uniform_half", "random_half", "live_loss_half")
WIDTHS = (32, 128, 512)
CHECKPOINTS = (4224, 4480, 8448, 8704, 12672)


def live_scores(model, images, seed):
    device = next(model.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            model.eval()
            with torch.no_grad():
                scores = torch.cat([model.per_sample_loss(batch.to(device)) for batch in images.split(32)])
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if scores.shape != (len(images),) or not torch.isfinite(scores).all():
        raise ValueError("Invalid per-image loss")
    return scores.detach().cpu()


def select_indices(policy, count, generator, scores=None):
    if count <= 0 or count % 32:
        raise ValueError("Window must contain a positive multiple of32 arrivals")
    if policy == "full":
        return torch.arange(count)
    if policy == "uniform_half":
        return torch.arange(1, count, 2)
    if policy == "random_half":
        return torch.randperm(count, generator=generator)[: count // 2].sort().values
    if policy == "live_loss_half" and scores is not None:
        if scores.shape != (count,) or not torch.isfinite(scores).all():
            raise ValueError("Invalid losses")
        return torch.argsort(scores, descending=True, stable=True)[: count // 2].sort().values
    raise ValueError("Unknown policy or missing losses")


def run_stream(model, optimizer, arrivals, policy, width, seed, checkpoints=CHECKPOINTS, observe=None):
    if policy not in POLICIES or width not in WIDTHS or arrivals.device.type != "cpu" or len(arrivals) % 64:
        raise ValueError("Invalid policy/width or incomplete host arrivals")
    if any(t <= 0 or t > len(arrivals) for t in checkpoints):
        raise ValueError("Invalid checkpoint clock")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    closes = list(range(width, len(arrivals) + 1, width))
    if not closes or closes[-1] != len(arrivals):
        closes.append(len(arrivals))
    events = sorted(set(closes) | set(checkpoints))
    start, queued, updates, accepted = 0, [], 0, 0
    records = []
    observations = {}
    for clock in events:
        if clock in closes:
            synchronize(device)
            tick = time.perf_counter()
            window = arrivals[start:clock].clone()
            gather_seconds = time.perf_counter() - tick
            score_seconds = 0.0
            scores = None
            if policy == "live_loss_half":
                score_tick = time.perf_counter()
                scores = live_scores(model, window, seed + 72003 + start)
                synchronize(device)
                score_seconds = time.perf_counter() - score_tick
            select_tick = time.perf_counter()
            indices = select_indices(policy, len(window), generator, scores)
            chosen = (indices + start).tolist()
            queued.extend(chosen)
            accepted += len(chosen)
            batches = []
            while len(queued) >= 32:
                batch_indices = queued[:32]
                del queued[:32]
                batches.append((batch_indices, arrivals[batch_indices].clone()))
            synchronize(device)
            selection_seconds = time.perf_counter() - select_tick
            train_tick = time.perf_counter()
            losses = [optimize_current(model, optimizer, batch.to(device), True) for _, batch in batches]
            synchronize(device)
            training_seconds = time.perf_counter() - train_tick
            updates += len(losses)
            records.append(
                {
                    "start": start,
                    "end": clock,
                    "selected_indices": chosen,
                    "exposure_indices": [ids for ids, _ in batches],
                    "queued_indices": list(queued),
                    "scores": None if scores is None else scores.tolist(),
                    "scoring_seed": None if scores is None else seed + 72003 + start,
                    "updates": len(losses),
                    "loss_gradients": losses,
                    "gather_seconds": gather_seconds,
                    "score_seconds": score_seconds,
                    "selection_seconds": selection_seconds,
                    "training_seconds": training_seconds,
                    "online_seconds": time.perf_counter() - tick,
                }
            )
            start = clock
        if clock in checkpoints:
            state = {
                "learner": capture(model, optimizer),
                "selector_rng": generator.get_state().clone(),
                "arrivals": clock,
                "updates": updates,
                "accepted_frames": accepted,
                "window_start": start,
                "pending_indices": list(range(start, clock)),
                "pending_images": arrivals[start:clock].clone(),
                "queued_indices": list(queued),
                "queued_images": arrivals[queued].clone(),
            }
            observations[clock] = None if observe is None else observe(clock, state)
    expected = len(arrivals) if policy == "full" else len(arrivals) // 2
    if queued or accepted != expected or updates * 32 != expected:
        raise ValueError("Final admission/exposure budget or queue mismatch")
    return {
        "blocks": records,
        "arrivals": len(arrivals),
        "accepted_frames": accepted,
        "updates": updates,
        "online_seconds": sum(r["online_seconds"] for r in records),
        "observations": observations,
    }
