"""Fixed source-group calibration and incremental-value tests for an anchor signal."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr, t
from sklearn.metrics import roc_auc_score

FEATURES = {
    "mean": (),
    "age": ("updates",),
    "loss": ("incoming_loss",),
    "age_loss": ("updates", "incoming_loss"),
    "anchor": ("relative_anchor_change",),
    "age_loss_anchor": ("updates", "incoming_loss", "relative_anchor_change"),
}


def grouped_predictions(points: list[dict[str, Any]], target: str) -> dict[str, Any]:
    """Leave out complete source groups; fit all transforms on training rows only."""
    groups = sorted({p["stream"] for p in points})
    if len(groups) != 6 or len(points) != 18:
        raise ValueError("Six groups with three checkpoint means required")
    y = np.array([p[target] for p in points], dtype=np.float64)
    result = {}
    for name, features in FEATURES.items():
        x = np.array([[p[f] for f in features] for p in points], dtype=np.float64).reshape(len(points), len(features))
        predicted = np.zeros_like(y)
        coefficients = {}
        for group in groups:
            test = np.array([p["stream"] == group for p in points])
            train = ~test
            center, scale = x[train].mean(0), x[train].std(0)
            scale[scale == 0] = 1
            z = (x[train] - center) / scale
            coefficient = np.linalg.solve(z.T @ z + np.eye(len(features)), z.T @ (y[train] - y[train].mean()))
            predicted[test] = y[train].mean() + ((x[test] - center) / scale) @ coefficient
            coefficients[group] = dict(zip(features, coefficient.tolist(), strict=True))
        error = (predicted - y) ** 2
        source_mse = {g: float(error[[p["stream"] == g for p in points]].mean()) for g in groups}
        labels = y >= 1
        result[name] = {
            "predictions": predicted.tolist(),
            "source_mse": source_mse,
            "mse": float(error.mean()),
            "coefficients": coefficients,
            "observed_harm_auc": float(roc_auc_score(labels, predicted)) if len(set(labels.tolist())) == 2 else None,
        }
    return result


def report(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply only the prespecified terminal prediction gates."""
    forecasts = {h: grouped_predictions(points, f"{h}_damage") for h in ("immediate", "terminal")}
    terminal = forecasts["terminal"]
    candidate = terminal["age_loss_anchor"]
    contrasts = {}
    for baseline in ("age_loss", "mean"):
        delta = np.array(
            [terminal[baseline]["source_mse"][g] - candidate["source_mse"][g] for g in sorted(candidate["source_mse"])]
        )
        width = float(t.ppf(1 - 0.05 / 4, 5) * delta.std(ddof=1) / np.sqrt(6))
        baseline_mse = terminal[baseline]["mse"]
        contrasts[baseline] = {
            "mean_mse_improvement": float(delta.mean()),
            "adjusted_ci95": [float(delta.mean() - width), float(delta.mean() + width)],
            "reduction_fraction": 1 - candidate["mse"] / baseline_mse if baseline_mse > 0 else None,
            "source_improvements": delta.tolist(),
        }
    positive = sum(v["relative_anchor_change"] > 0 for v in candidate["coefficients"].values())
    predictive = all(
        v["reduction_fraction"] is not None and v["reduction_fraction"] >= 0.1 and v["adjusted_ci95"][0] > 0
        for v in contrasts.values()
    )
    gates = {
        "terminal_incremental_prediction": predictive,
        "expected_sign_transfer": positive >= 5,
        "observed_action_support": any(p["terminal_damage"] >= 1 for p in points)
        and any(p["terminal_damage"] <= 0 for p in points),
    }
    ranks = {}
    for horizon in ("immediate", "terminal"):
        y = [p[f"{horizon}_damage"] for p in points]
        ranks[horizon] = {
            f: float(spearmanr([p[f] for p in points], y).statistic)
            if np.std([p[f] for p in points]) > 0 and np.std(y) > 0
            else None
            for f in ("relative_anchor_change", "updates", "incoming_loss")
        }
    return {
        "points": points,
        "forecasts": forecasts,
        "primary_contrasts": contrasts,
        "positive_anchor_folds": positive,
        "gates": gates,
        "signal_qualified": all(gates.values()),
        "rank_correlations_descriptive": ranks,
        "observed_terminal_harm_points": sum(p["terminal_damage"] >= 1 for p in points),
        "observed_terminal_nonharmful_points": sum(p["terminal_damage"] <= 0 for p in points),
        "observed_terminal_balanced_gain_points": sum(p["terminal_balanced_effect"] >= 0.5 for p in points),
        "independent_confirmation": False,
        "controller_trained": False,
    }
