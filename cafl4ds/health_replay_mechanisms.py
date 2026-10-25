"""Controlled replay eligibility and source-excluded health interpretation."""

from __future__ import annotations

import numpy as np

FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")
CELLS = tuple(
    f"{age}_{status}_{size}" for age in ("old", "recent") for status in ("trained", "unused") for size in (32, 256)
)
ARMS = (*CELLS, "parent_memory", "current")


def exposures(trace, frames=131072):
    """Count every actual gradient exposure, including previously replayed images."""
    count = np.zeros(frames, dtype=np.int64)
    for row in trace:
        ids = row["diet"]["current_ids"] + row["diet"]["replay_ids"]
        if min(ids) < 0 or max(ids) >= frames:
            raise ValueError("Exposure identity outside the parent history")
        np.add.at(count, ids, 1)
    return count


def pools(count, seed):
    """Uniform eligible pools; the 32-frame pool nests inside its 256-frame pool."""
    count = np.asarray(count)
    if count.ndim != 1 or len(count) % 4 or (count < 0).any():
        raise ValueError("Require four equal arrival quarters and nonnegative counts")
    rng = np.random.default_rng(seed)
    output = {}
    for age, ids in (("old", np.arange(len(count) // 4)), ("recent", np.arange(3 * len(count) // 4, len(count)))):
        for status in ("trained", "unused"):
            eligible = ids[(count[ids] > 0) if status == "trained" else (count[ids] == 0)]
            if len(eligible) < 256:
                raise ValueError("Insufficient eligible frames for the fixed factorial design")
            chosen = rng.choice(eligible, 256, replace=False)
            output[f"{age}_{status}_256"] = chosen.tolist()
            output[f"{age}_{status}_32"] = chosen[:32].tolist()
    return output


def schedule(pool, steps=128):
    """Use common current subsets and common replay slots at each pool size."""
    current_rng, replay_rng = np.random.default_rng(74131), np.random.default_rng(74137)
    rows = []
    for _ in range(steps):
        current = sorted(current_rng.choice(32, 16, replace=False).tolist())
        past = [] if pool is None else np.asarray(pool)[replay_rng.choice(len(pool), 16, replace=False)].tolist()
        rows.append({"current": list(range(32)) if pool is None else current, "past": past})
    return rows


def held_source_ridge(x, y, groups, penalty=10.0):
    """Fit scaling and coefficients only on other complete historical sources."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if x.ndim != 2 or len(x) != len(y) or len(x) != len(groups) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Require aligned finite observations")
    predicted = np.empty_like(y, dtype=float)
    folds = []
    for group in np.unique(groups):
        test = groups == group
        if test.all():
            raise ValueError("Need multiple historical sources")
        center, scale = x[~test].mean(0), x[~test].std(0)
        scale[scale == 0] = 1
        z = (x[~test] - center) / scale
        target_mean = y[~test].mean(0)
        beta = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y[~test] - target_mean))
        predicted[test] = target_mean + ((x[test] - center) / scale) @ beta
        folds.append(
            {
                "held": int(group),
                "training_rows": np.flatnonzero(~test).tolist(),
                "test_rows": np.flatnonzero(test).tolist(),
                "center": center,
                "scale": scale,
                "beta": beta,
            }
        )
    return predicted, folds


def factorial(values):
    """Return source-specific main effects; do not treat cells as replications."""
    a = np.asarray(values, float)
    if a.shape[:5] != (3, 3, 2, 2, 2):
        raise ValueError("Require source, training draw, age, exposure status, pool size")
    return {
        "old_minus_recent": (a[:, :, 0] - a[:, :, 1]).mean(axis=(1, 2, 3)),
        "unused_minus_trained": (a[:, :, :, 1] - a[:, :, :, 0]).mean(axis=(1, 2, 3)),
        "pool256_minus32": (a[:, :, :, :, 1] - a[:, :, :, :, 0]).mean(axis=(1, 2, 3)),
    }
