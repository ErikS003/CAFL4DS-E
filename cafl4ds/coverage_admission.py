"""Causal local batch coverage against a bounded accepted-content history."""

import time
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from cafl4ds.regime_windows import live_scores
from cafl4ds.short_buffer import capture, synchronize
from cafl4ds.update_filtering import optimize_current

ARMS = ("early_stop", "uniform", "random", "coverage")
CHECKPOINTS = (6144, 12288)
MEMORY = 32


def select_coverage(centroids, memory):
    """Float64 CPU geometry, deterministic earliest ties, greedy two-batch novelty."""
    z = np.asarray(centroids, dtype=np.float64)
    if z.ndim != 2 or len(z) != 4 or not np.isfinite(z).all() or np.any(np.linalg.norm(z, axis=1) == 0):
        raise ValueError("Four finite nonzero batch centroids required")
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    history = [np.asarray(v, dtype=np.float64) for v in memory]
    chosen, rounds = [], []
    for _ in range(2):
        scores = [min(float(np.clip(1 - np.sum(v * h), 0, 2)) for h in history) if history else 0.0 for v in z]
        k = min((k for k in range(4) if k not in chosen), key=lambda k: (-scores[k], k))
        rounds.append(scores)
        chosen.append(k)
        history = (history + [z[k].copy()])[-MEMORY:]
    selected = sorted(chosen)
    actual = (list(memory) + [z[k].copy() for k in selected])[-MEMORY:]
    return selected, rounds, actual


def reference_centroids(reference, images, seed):
    device = next(reference.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.no_grad():
            z = F.normalize(reference.embed(images.to(device)), dim=1).reshape(4, 32, -1).mean(dim=1)
    return z.detach().cpu().numpy()


def run_admission(
    model, optimizer, arrivals, arm, reference, seed, checkpoints=CHECKPOINTS, observe=None, diagnostics=True
):
    if arm not in ARMS or len(arrivals) % 128 or arrivals.device.type != "cpu":
        raise ValueError("Four complete host batches and a known arm required")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 62003)
    memory, recent, records, observations = [], [], [], {}
    updates = 0
    for start in range(0, len(arrivals), 128):
        synchronize(device)
        tick = time.perf_counter()
        window = arrivals[start : start + 128].clone()
        gather = time.perf_counter() - tick
        diagnostic_seconds, losses = 0.0, None
        if diagnostics:
            dt = time.perf_counter()
            losses = live_scores(model, window, seed + 93003 + start).reshape(4, 32).mean(1).tolist()
            synchronize(device)
            diagnostic_seconds = time.perf_counter() - dt
        prior_loss = statistics.fmean(recent[-4:]) if recent else None
        st = time.perf_counter()
        centroids, rounds = None, None
        if arm == "coverage":
            centroids = reference_centroids(reference, window, seed + 83003 + start)
            selected, rounds, memory = select_coverage(centroids, memory)
        elif arm == "uniform":
            selected = [0, 2]
        elif arm == "random":
            selected = sorted(torch.randperm(4, generator=generator)[:2].tolist())
        else:
            selected = list(range(4)) if start < 6144 else []
        synchronize(device)
        selection_seconds = time.perf_counter() - st
        train_tick = time.perf_counter()
        trained = [optimize_current(model, optimizer, window[k * 32 : (k + 1) * 32].to(device), True) for k in selected]
        synchronize(device)
        training_seconds = time.perf_counter() - train_tick
        recent.extend(float(v[0]) for v in trained)
        updates += len(selected)
        exposures = [list(range(start + k * 32, start + (k + 1) * 32)) for k in selected]
        records.append(
            {
                "start": start,
                "end": start + 128,
                "selected_batches": selected,
                "exposure_indices": exposures,
                "updates": len(selected),
                "loss_gradients": trained,
                "centroids": None if centroids is None else centroids.tolist(),
                "novelty_rounds": rounds,
                "incoming_losses": losses,
                "prior_training_loss": prior_loss,
                "relative_losses": None if losses is None or prior_loss is None else [v - prior_loss for v in losses],
                "gather_seconds": gather,
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "processing_seconds": gather + selection_seconds + training_seconds,
                "diagnostic_seconds": diagnostic_seconds,
            }
        )
        clock = start + 128
        if clock in checkpoints:
            state = {
                "learner": capture(model, optimizer),
                "selector_rng": generator.get_state().clone(),
                "memory": [v.tolist() for v in memory],
                "updates": updates,
                "arrivals": clock,
                "accepted_frames": updates * 32,
                "arm": arm,
            }
            observations[clock] = None if observe is None else observe(clock, state)
    expected = min(len(arrivals), 6144) // 32 if arm == "early_stop" else len(arrivals) // 64
    if updates != expected:
        raise ValueError("Matched-budget count failed")
    return {
        "blocks": records,
        "updates": updates,
        "processing_seconds": sum(v["processing_seconds"] for v in records),
        "diagnostic_seconds": sum(v["diagnostic_seconds"] for v in records),
        "observations": observations,
    }
