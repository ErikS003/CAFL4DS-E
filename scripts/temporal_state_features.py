"""Cheap state descriptors restricted to already observed, closed FIFO windows."""

from __future__ import annotations

import numpy as np


def causal_features(frames: np.ndarray, offset: int) -> dict[str, float | int]:
    """Describe the last 64 arrivals; never inspect frames after the decision."""
    if frames.ndim != 4 or frames.shape[1:] != (64, 64, 3) or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB64 arrivals")
    if offset < 1 or offset * 64 > len(frames):
        raise ValueError("Decision requires a complete observed window")
    end = offset * 64
    observed = frames[max(0, end - 128):end].astype(np.float32) / 255
    pooled = observed.reshape(len(observed), 8, 8, 8, 8, 3).mean(axis=(2, 4))
    recent = pooled[-64:]
    changes = np.abs(np.diff(recent, axis=0)).mean(axis=(1, 2, 3))
    spread = np.abs(recent - recent.mean(axis=0)).mean()
    previous = pooled[:-64]
    novelty = np.abs(recent.mean(axis=0) - previous.mean(axis=0)).mean() if len(previous) else 0.0
    return {
        "observed_arrivals": end,
        "feature_window_arrivals": 64,
        "mean_adjacent_rgb_change": float(changes.mean()),
        "p90_adjacent_rgb_change": float(np.quantile(changes, 0.9)),
        "within_window_rgb_spread": float(spread),
        "previous_window_mean_shift": float(novelty),
        "previous_window_available": int(bool(len(previous))),
    }
