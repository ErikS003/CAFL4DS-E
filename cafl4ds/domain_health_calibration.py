"""Replay dose interventions and strictly excluded domain calibration."""

from __future__ import annotations

import numpy as np

ARMS = ("current", "balanced", "history")
FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")


def schedule(pool, arm, steps):
    """Nested current positions and replay positions, with a constant batch size."""
    if arm not in ARMS or len(pool) < 32 or len(set(pool)) != len(pool):
        raise ValueError("Require a declared dose and unique historical pool")
    current_rng, past_rng = np.random.default_rng(74131), np.random.default_rng(74137)
    n = {"current": 32, "balanced": 16, "history": 0}[arm]
    rows = []
    for _ in range(steps):
        now, old = current_rng.permutation(32), past_rng.permutation(len(pool))
        rows.append({"current": sorted(now[:n].tolist()), "past": np.asarray(pool)[old[: 32 - n]].tolist()})
    return rows


def excluded_ridge(x, y, sources, contexts, joint=True, penalty=10.0):
    """Exclude both test history and test future; never tune on held outcomes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sources, contexts = np.asarray(sources), np.asarray(contexts)
    if x.ndim != 2 or len(x) != len(y) or len(x) != len(sources) or len(x) != len(contexts):
        raise ValueError("Require aligned observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite calibration data")
    prediction, folds = np.empty_like(y), []
    for source in np.unique(sources):
        for context in np.unique(contexts) if joint else [None]:
            test = sources == source
            train = sources != source
            if joint:
                test &= contexts == context
                train &= contexts != context
            if not train.any() or not test.any():
                raise ValueError("Empty excluded fold")
            center, scale = x[train].mean(0), x[train].std(0)
            scale[scale == 0] = 1
            z, mean = (x[train] - center) / scale, y[train].mean(0)
            beta = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y[train] - mean))
            prediction[test] = mean + ((x[test] - center) / scale) @ beta
            folds.append(
                {
                    "source": int(source),
                    "context": None if context is None else int(context),
                    "train": np.flatnonzero(train),
                    "test": np.flatnonzero(test),
                    "center": center,
                    "scale": scale,
                    "beta": beta,
                }
            )
    return prediction, folds


def dynamics(trace, origin):
    """Only paid updates available by the prediction origin enter these features."""
    a = np.asarray([r["response"] for r in trace[:origin]], float)
    if origin < 4 or len(a) != origin:
        raise ValueError("Require at least four observed updates")
    return np.r_[a[-4:].mean(0), a[-4:].mean(0) - a[:4].mean(0)]
