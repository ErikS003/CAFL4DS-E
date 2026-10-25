"""Causal context-sensitive replay timing at exactly matched historical dose."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.stats import t

ARMS = ("current", "fixed", "random", "adaptive")
SEEDS = (5003, 5009, 5011)
WARM_STEPS = 8
STEPS = 192
MEMORY = 512
RECENT = 256


def stream_seed(stream: str, seed: int) -> int:
    """Derive private sampling seeds without touching the learner RNG."""
    return seed + int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)


def memory_plan(stream: str, seed: int) -> list[list[int]]:
    """Choose identical causal old-image sequences for every mixture arm."""
    base = stream_seed(stream, seed)
    generator = np.random.default_rng(base + 74111)
    reservoir = list(range(MEMORY))
    seen, plans = MEMORY, []
    for block in range((STEPS - WARM_STEPS) // 8):
        stop = (WARM_STEPS + block * 8) * 64
        for index in range(seen, stop):
            slot = int(generator.integers(index + 1))
            if slot < MEMORY:
                reservoir[slot] = index
        seen = stop
        sampler = np.random.default_rng(base + 75203 + block)
        plans.append(sampler.choice(reservoir, 64, replace=False).tolist())
    # Include maintenance of later arrivals in the algorithmic work plan.
    for index in range(seen, STEPS * 64):
        slot = int(generator.integers(index + 1))
        if slot < MEMORY:
            reservoir[slot] = index
    return plans


def context_distance(features: np.ndarray, incoming: list[int], recent: list[int]) -> float:
    """Compare current arrivals with actual recent training exposures."""
    if not recent or min(incoming) <= max(recent) or not np.isfinite(features[incoming + recent]).all():
        raise ValueError("Finite arrived features and strictly past training history required")
    z = features[incoming + recent].astype(np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    a, b = z[: len(incoming)].mean(0), z[len(incoming) :].mean(0)
    return float(np.clip(1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), 0, 2))


class Mixture:
    """Allocate replay quota using only arrived context and private randomness."""

    def __init__(self, arm: str, plans: list[list[int]], seed: int) -> None:
        if arm not in ARMS or len(plans) != 23:
            raise ValueError("Known arm and twenty-three causal quota plans required")
        self.arm, self.plans = arm, plans
        self.generator = np.random.default_rng(seed + 76349)
        self.recent: list[int] = []
        self.calibration: list[float] = []
        self.threshold: float | None = None
        self.remaining = 0
        self.cursor = 0

    def choose(self, step: int, features: np.ndarray | None) -> dict[str, Any]:
        """Construct a batch while enforcing equal replay per eight updates."""
        start = step * 64
        score = None
        if self.arm == "adaptive" and step:
            if features is None:
                raise ValueError("Adaptive scoring requires arrived reference features")
            score = context_distance(features, list(range(start, start + 64)), self.recent)
        if step < WARM_STEPS:
            if score is not None:
                self.calibration.append(score)
            old, amount, target, forced = [], 0, 0, False
        elif self.arm == "current":
            old, amount, target, forced = [], 0, 0, False
        else:
            block, position = divmod(step - WARM_STEPS, 8)
            if position == 0:
                self.remaining, self.cursor = 64, 0
            if self.arm == "fixed":
                target = 8
            elif self.arm == "random":
                target = 16 if self.generator.random() < 0.5 else 0
            else:
                if self.threshold is None:
                    if len(self.calibration) != 7:
                        raise ValueError("Seven past-only calibration scores required")
                    self.threshold = float(np.quantile(self.calibration, 0.9, method="linear") + 1e-8)
                target = 16 if score > self.threshold else 0
            low = max(0, self.remaining - 16 * (7 - position))
            high = min(16, self.remaining)
            amount = min(max(target, low), high)
            forced = amount != target
            old = self.plans[block][self.cursor : self.cursor + amount]
            self.cursor += amount
            self.remaining -= amount
            if position == 7 and self.remaining:
                raise ValueError("Unspent replay quota")
        current = (start + np.linspace(0, 31, 32 - amount, dtype=int)).tolist()
        if any(index >= start for index in old) or len(set(current)) != len(current):
            raise ValueError("Invalid causal exposure selection")
        self.recent = (self.recent + old + current)[-RECENT:]
        return {
            "step": step,
            "offset": start,
            "old": old,
            "current": current,
            "historical_count": amount,
            "target_count": target,
            "quota_forced": forced,
            "score": score,
            "threshold": self.threshold,
            "quota_remaining": self.remaining,
        }


def interval(values: list[float], family: int) -> dict[str, Any]:
    """Compute source-group intervals without treating seeds as histories."""
    x = np.asarray(values, dtype=float)
    if len(x) != 6:
        raise ValueError("Six history means required")
    width = t.ppf(1 - 0.05 / (2 * family), 5) * x.std(ddof=1) / np.sqrt(6)
    return {
        "mean_pp": float(x.mean()),
        "ci95_pp": [float(x.mean() - width), float(x.mean() + width)],
        "source_values_pp": x.tolist(),
    }
