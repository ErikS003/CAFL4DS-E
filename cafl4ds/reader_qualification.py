"""Frozen support and paired-query resampling for the operational reader."""

from __future__ import annotations

from typing import Any

import numpy as np

C_GRID = (0.01, 0.1, 1.0, 10.0)
SUPPORT_SEEDS = tuple(range(294001, 294013))


def cells(indices: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> list[np.ndarray]:
    """Return the six declared cells, rejecting incomplete panels."""
    groups = [indices[(domains[indices] == d) & (labels[indices] == c)] for d in (0, 1) for c in (0, 1, 2)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Missing domain/class cell")
    return groups


def support_plan(pool: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict[str, Any]:
    """Freeze balanced CV folds and full-budget replacement draws."""
    groups = cells(pool, labels, domains)
    if len(pool) != 1152 or len(np.unique(pool)) != 1152 or any(len(g) != 192 for g in groups):
        raise ValueError("Expected 1,152 distinct support identities, 192 per cell")
    rng = np.random.default_rng(293003)
    splits = [np.array_split(rng.permutation(g), 5) for g in groups]
    folds = []
    for fold in range(5):
        validation = np.concatenate([s[fold] for s in splits])
        fitting = pool[~np.isin(pool, validation)]
        folds.append({"fitting": fitting.tolist(), "validation": validation.tolist()})
    draws = {"full": pool.tolist()}
    for draw, seed in enumerate(SUPPORT_SEEDS):
        rng = np.random.default_rng(seed)
        draws[f"bootstrap{draw:02d}"] = np.concatenate([rng.choice(g, 192, replace=True) for g in groups]).tolist()
    multiplicities = {}
    for name, indices in draws.items():
        identities, counts = np.unique(indices, return_counts=True)
        multiplicities[name] = {str(i): int(n) for i, n in zip(identities, counts, strict=True)}
    return {"folds": folds, "supports": draws, "multiplicities": multiplicities}


def query_weights(indices: np.ndarray, labels: np.ndarray, domains: np.ndarray, seed: int) -> np.ndarray:
    """Balanced cell weights for full query and 1,000 shared paired resamples."""
    rng = np.random.default_rng(seed)
    groups = cells(indices, labels, domains)
    weights = np.zeros((1001, 6, len(labels)))
    for cell, group in enumerate(groups):
        weights[0, cell, group] = 1 / len(group)
        for draw in range(1, 1001):
            sampled = rng.choice(group, len(group), replace=True)
            ids, counts = np.unique(sampled, return_counts=True)
            weights[draw, cell, ids] = counts / len(group)
    return weights


def summarize(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    """Central conditional sensitivity range, with practical sign/equivalence rates."""
    x = values.ravel() * scale
    lo, hi = np.quantile(x, [0.025, 0.975])
    return {
        "mean": float(x.mean()),
        "lower": float(lo),
        "upper": float(hi),
        "half_width": float((hi - lo) / 2),
        "positive_fraction": float((x > 1e-12).mean()),
        "equivalent_fraction": float((np.abs(x) <= 0.1).mean()) if scale == 100 else float("nan"),
    }
