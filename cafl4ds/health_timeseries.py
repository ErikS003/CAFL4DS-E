"""Boundary-aware health correlations and causal grouped linear forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.prediction_benchmark import correlation, improvement

METRICS = ("variance", "covariance", "rank", "cka", "cosine")
MODELS = ("ar1", "ar2", "delta_ar2", "varx")
TRAIN_ORIGINS = tuple(range(4, 17))
TEST_ORIGINS = tuple(range(20, 29))


def acf(values: np.ndarray, maximum_lag: int = 8) -> list[float | None]:
    """Biased ACF using one trajectory's full centered zero-lag denominator."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) <= maximum_lag or not np.isfinite(x).all():
        raise ValueError("Finite one-dimensional series longer than all lags required")
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator < 1e-24:
        return [None] * (maximum_lag + 1)
    return [float(centered[: len(x) - lag] @ centered[lag:] / denominator) for lag in range(maximum_lag + 1)]


def pacf(values: np.ndarray, maximum_lag: int = 4) -> list[float]:
    """OLS partial autocorrelation with a separate intercept at each lag."""
    x = np.asarray(values, float)
    result = []
    for lag in range(1, maximum_lag + 1):
        design = np.column_stack([np.ones(len(x) - lag)] + [x[lag - j : len(x) - j] for j in range(1, lag + 1)])
        result.append(float(np.linalg.lstsq(design, x[lag:], rcond=None)[0][-1]))
    return result


def ccf(x: np.ndarray, y: np.ndarray, lag: int) -> float | None:
    """Positive lag means x precedes y; never joins separate trajectory boundaries."""
    if len(x) != len(y) or abs(lag) >= len(x) - 2:
        raise ValueError("Matched nontrivial series required")
    if lag >= 0:
        return correlation(x[: len(x) - lag], y[lag:])
    return correlation(x[-lag:], y[: len(y) + lag])


def detrended(values: np.ndarray) -> np.ndarray:
    """Remove a full-series linear trend for descriptive diagnostics only."""
    design = np.column_stack((np.ones(len(values)), np.arange(len(values))))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def feature(series: dict[str, Any], metric: int, origin: int, model: str) -> np.ndarray:
    """Every input ends at the origin, before any target-horizon update is observed."""
    h = np.asarray(series["health"], float)
    if origin < 4 or origin >= len(h):
        raise ValueError("Four observed updates required")
    if model == "ar1":
        return h[origin, [metric]]
    if model == "ar2":
        return h[[origin, origin - 1], metric]
    if model == "delta_ar2":
        return np.diff(h[origin - 2 : origin + 1, metric])[[1, 0]]
    if model == "varx":
        return np.r_[
            h[origin],
            h[origin] - h[origin - 1],
            np.mean(series["loss"][origin - 4 : origin]),
            np.mean(series["gradient_norm"][origin - 4 : origin]),
            series["parent_updates"] + origin,
        ]
    raise ValueError("Unknown frozen model")


def linear_predict(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed ridge10 and fitting-row-only normalization, with an unpenalized intercept."""
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - center) / scale
    intercept = y.mean()
    beta = np.linalg.solve(z.T @ z + 10 * np.eye(x.shape[1]), z.T @ (y - intercept))
    return intercept + ((query - center) / scale) @ beta, {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "beta": beta.tolist(),
        "intercept": float(intercept),
        "fit_rows": len(y),
    }


def summarize_errors(y: np.ndarray, predictions: dict[str, np.ndarray], groups: np.ndarray) -> dict[str, Any]:
    """Macro-history errors preserve the six source units despite overlapping windows."""
    result = {}
    for name, predicted in predictions.items():
        source = {str(g): float(np.mean((predicted[groups == g] - y[groups == g]) ** 2)) for g in sorted(set(groups))}
        mse = float(np.mean(list(source.values())))
        result[name] = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "source_mse": source,
            "correlation": correlation(y, predicted),
            "predictions": predicted.tolist(),
        }
    for model in result.values():
        model["skill_vs_persistence_percent"] = (
            100 * (1 - model["mse"] / result["persistence"]["mse"]) if result["persistence"]["mse"] > 1e-16 else None
        )
    return result


def forecast(series: list[dict[str, Any]], metric: int, horizon: int) -> dict[str, Any]:
    """Fit other histories' early prefixes, score the held history's later origins."""
    train, test = [], []
    for i, s in enumerate(series):
        h = np.asarray(s["health"])
        for origins, destination in ((TRAIN_ORIGINS, train), (TEST_ORIGINS, test)):
            for origin in origins:
                destination.append(
                    {
                        "series": i,
                        "stream": s["stream"],
                        "checkpoint": s["checkpoint"],
                        "origin": origin,
                        "target_update": origin + horizon,
                        "target": float(h[origin + horizon, metric] - h[origin, metric]),
                    }
                )
    y = np.array([r["target"] for r in test])
    groups = np.array([r["stream"] for r in test])
    predictions = {
        "persistence": np.zeros(len(test)),
        "local_trend": np.array(
            [
                horizon
                * (
                    series[r["series"]]["health"][r["origin"]][metric]
                    - series[r["series"]]["health"][r["origin"] - 4][metric]
                )
                / 4
                for r in test
            ]
        ),
    }
    folds = {}
    for name in MODELS:
        xtrain = np.array([feature(series[r["series"]], metric, r["origin"], name) for r in train])
        xtest = np.array([feature(series[r["series"]], metric, r["origin"], name) for r in test])
        outcomes = np.array([r["target"] for r in train])
        predictions[name] = np.zeros(len(test))
        folds[name] = {}
        for group in sorted(set(groups)):
            fit = np.array([r["stream"] != group for r in train])
            held = groups == group
            predictions[name][held], folds[name][str(group)] = linear_predict(xtrain[fit], outcomes[fit], xtest[held])
    scored = summarize_errors(y, predictions, groups)
    contrasts = {name: improvement(scored["ar2"], scored[name], 10) for name in ("persistence", "local_trend")}
    residual_acf = {}
    for name, prediction in predictions.items():
        values = []
        for i in range(len(series)):
            select = np.array([r["series"] == i for r in test])
            values.append(acf((y - prediction)[select], 4))
        residual_acf[name] = values
    return {
        "metric": METRICS[metric],
        "horizon": horizon,
        "training_rows": train,
        "test_rows": test,
        "models": scored,
        "folds": folds,
        "ar2_contrasts": contrasts,
        "ar2_screen_passed": all(c["passes"] for c in contrasts.values()),
        "residual_acf_per_trajectory": residual_acf,
    }


def aggregate(values: list[list[float | None]], groups: list[str]) -> dict[str, Any]:
    """Average trajectories inside history before showing history ranges."""
    a = np.array([[np.nan if v is None else v for v in row] for row in values])
    per_source = {g: np.nanmean(a[np.array(groups) == g], axis=0) for g in sorted(set(groups))}
    matrix = np.array(list(per_source.values()))

    def finite(row: np.ndarray) -> list[float | None]:
        return [float(v) if np.isfinite(v) else None for v in row]

    return {
        "mean": finite(np.nanmean(matrix, axis=0)),
        "history_min": finite(np.nanmin(matrix, axis=0)),
        "history_max": finite(np.nanmax(matrix, axis=0)),
        "source_means": {g: finite(v) for g, v in per_source.items()},
    }


def diagnostics(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe time dependence without pretending series windows are independent data."""
    result = {}
    groups = [s["stream"] for s in series]
    for k, metric in enumerate(METRICS):
        levels = [np.array(s["health"])[:, k] for s in series]
        delta = [np.diff(h) for h in levels]
        result[metric] = {
            "level_acf": aggregate([acf(h) for h in levels], groups),
            "delta_acf": aggregate([acf(h) for h in delta], groups),
            "detrended_level_acf": aggregate([acf(detrended(h)) for h in levels], groups),
            "level_pacf": aggregate([pacf(h) for h in levels], groups),
            "delta_pacf": aggregate([pacf(h) for h in delta], groups),
        }
        for signal in ("loss", "gradient_norm"):
            result[metric][signal + "_ccf"] = aggregate(
                [
                    [ccf(np.array(s[signal]), d, lag) for lag in range(-4, 5)]
                    for s, d in zip(series, delta, strict=True)
                ],
                groups,
            )
    cross = {}
    for i, first in enumerate(METRICS):
        for j, second in enumerate(METRICS):
            cross[first + ":" + second] = aggregate(
                [
                    [correlation(np.diff(np.array(s["health"])[:, i]), np.diff(np.array(s["health"])[:, j]))]
                    for s in series
                ],
                groups,
            )
    return {
        "metrics": result,
        "cross_health_delta_correlation": cross,
        "acf_pair_counts": {"levels": [33 - lag for lag in range(9)], "increments": [32 - lag for lag in range(9)]},
        "ccf_lags": list(range(-4, 5)),
        "correlations_are_descriptive": True,
    }
