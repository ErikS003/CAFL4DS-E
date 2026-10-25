"""Grouped prediction scores and float64 geometry for two separate research pilots."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, t


def held_group_ridge(x, y, groups, penalty=10.0):
    """Exclude every held-history row, including its normalization statistics."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups) or len(set(groups)) < 3:
        raise ValueError("Aligned arrays and at least three complete groups required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite observations required")
    predicted = np.zeros(len(y))
    folds = {}
    for group in sorted(set(groups)):
        test = groups == group
        center, scale = x[~test].mean(0), x[~test].std(0)
        scale[scale < 1e-12] = 1
        z = (x[~test] - center) / scale
        intercept = y[~test].mean()
        beta = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y[~test] - intercept))
        predicted[test] = intercept + ((x[test] - center) / scale) @ beta
        folds[str(group)] = {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "beta": beta.tolist(),
            "intercept": float(intercept),
            "fit_rows": int((~test).sum()),
        }
    return predicted, folds


def correlation(a, b, rank=False):
    """Represent undefined association as null, not a fabricated zero."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.ptp(a) < 1e-15 or np.ptp(b) < 1e-15:
        return None
    return float(spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def scores(y, predictions, groups):
    """Weight histories equally; rows within a history are repeated measurements."""
    y, groups = np.asarray(y), np.asarray(groups)
    result = {}
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted)
        source = {str(g): float(np.mean((predicted[groups == g] - y[groups == g]) ** 2)) for g in sorted(set(groups))}
        mse = float(np.mean(list(source.values())))
        nonzero = np.abs(y) > 1e-12
        result[name] = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "source_mse": source,
            "correlation": correlation(y, predicted),
            "predictions": predicted.tolist(),
            "sign_agreement": float(np.mean(np.sign(y[nonzero]) == np.sign(predicted[nonzero])))
            if nonzero.any()
            else None,
        }
    for value in result.values():
        value["skill_vs_zero_percent"] = (
            100 * (1 - value["mse"] / result["zero"]["mse"]) if result["zero"]["mse"] > 1e-16 else None
        )
    return result


def improvement(candidate, baseline, family):
    """Paired history-level error improvement, adjusted across prespecified tests."""
    keys = sorted(candidate["source_mse"])
    delta = np.array([baseline["source_mse"][g] - candidate["source_mse"][g] for g in keys])
    half = float(t.ppf(1 - 0.05 / (2 * family), len(keys) - 1) * delta.std(ddof=1) / np.sqrt(len(keys)))
    mean = float(delta.mean())
    fraction = 1 - candidate["mse"] / baseline["mse"] if baseline["mse"] > 1e-16 else None
    return {
        "mse_improvement": mean,
        "adjusted_95_interval": [mean - half, mean + half],
        "fractional_reduction": fraction,
        "passes": bool(fraction is not None and fraction >= 0.1 and mean - half > 0),
    }


def health_values(z, warm):
    """Raw RankMe, normalized spread/covariance and warm-relative movement, float64."""
    z, warm = np.asarray(z, np.float64), np.asarray(warm, np.float64)
    if z.shape != warm.shape or z.ndim != 2 or len(z) <= z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Finite matched banks larger than the representation dimension required")
    q = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    w = warm / np.maximum(np.linalg.norm(warm, axis=1, keepdims=True), 1e-12)
    covariance = np.cov(q, rowvar=False)
    offdiag = covariance - np.diag(np.diag(covariance))
    singular = np.linalg.svd(z, compute_uv=False)
    p = singular / max(singular.sum(), 1e-12) + 1e-7
    a, b = z - z.mean(0), warm - warm.mean(0)
    cka = np.sum((a.T @ b) ** 2) / max(np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b), 1e-30)
    return {
        "variance": float(q.var(0, ddof=1).mean()),
        "covariance": float((offdiag**2).sum() / z.shape[1]),
        "rank": float(np.exp(-np.sum(p * np.log(p)))),
        "cka": float(1 - cka),
        "cosine": float(1 - np.sum(q * w, axis=1).mean()),
    }


def health_effect(before, after, baseline):
    """Signed intervention effects, never renamed as capability or damage."""
    return {
        k: (after[k] - baseline[k]) / max(abs(before[k]), 1e-12)
        if k in ("variance", "covariance", "rank")
        else after[k] - baseline[k]
        for k in before
    }
