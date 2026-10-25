"""Isolated pre-action observations for fixed hold/full continuation diagnostics."""

import time

import torch
import torch.nn.functional as F

from cafl4ds.short_buffer import synchronize

STATES = {"day_end": 4224, "night_end": 8448, "return_early": 8704}
CHECKPOINTS = (256, 1024)
ACTIONS = ("hold", "full")


def observe_before_action(model, recent, incoming, recent_training_loss, seed):
    """Only the first arrived batch and past frames are visible; no learner mutation."""
    if len(recent) != 128 or len(incoming) != 32:
        raise ValueError("Expected128 past frames and32 arrived frames")
    device = next(model.parameters()).device
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    modes = {name: module.training for name, module in model.named_modules()}
    synchronize(device)
    tick = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed + 93003)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed + 93003)
            model.eval()
            with torch.no_grad():
                losses = model.per_sample_loss(incoming.to(device)).cpu()
                past = torch.cat([model.encoder.embed(b.to(device)).cpu() for b in recent.split(32)])
                current = model.encoder.embed(incoming.to(device)).cpu()
                past, current = F.normalize(past, dim=1), F.normalize(current, dim=1)
                distance = float((current.mean(0) - past.mean(0)).norm())
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    synchronize(device)
    loss = float(losses.mean())
    if not torch.isfinite(losses).all() or not torch.isfinite(torch.tensor(distance)):
        raise ValueError("Nonfinite observation")
    return {
        "incoming_reconstruction_loss": loss,
        "recent_training_loss_mean": recent_training_loss,
        "loss_difference": loss - recent_training_loss,
        "feature_centroid_distance_l2": distance,
        "incoming_frames_visible": 32,
        "recent_frames": 128,
        "decision_arrival_offset": 32,
        "observer_seed": seed + 93003,
        "observer_seconds": time.perf_counter() - tick,
    }
