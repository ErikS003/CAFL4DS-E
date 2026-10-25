"""Pre-action features, strictly excluded forecasts and paired action value."""

from __future__ import annotations

import numpy as np

FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")
SIGNALS = tuple(f"{b}_{f}" for b in ("monitor", "audit") for f in FIELDS) + (
    "cumulative_loss",
    "cumulative_gradient",
    "cumulative_step_l2",
    "trailing_loss",
    "trailing_gradient",
    "trailing_step_l2",
)
MODELS = ("mean", "diet", "cheap", "health_only", "health_added")


def preaction_features(readings, metrics):
    """All inputs end at the common action fork; no action outcomes accepted."""
    a = np.asarray(metrics, float)
    if a.shape != (128, 3) or not np.isfinite(a).all():
        raise ValueError("Require the 128 pre-action loss/gradient/actual-step rows")
    geometry = np.array([readings[b]["change"][f] for b in ("monitor", "audit") for f in FIELDS])
    return np.r_[geometry, a.mean(0), a[-16:].mean(0)]


def causal_norm(metrics):
    a = np.asarray(metrics, float)
    if a.ndim != 2 or a.shape != (128, 3):
        raise ValueError("Require the complete pre-action trace only")
    target = float(a[-16:, 2].mean())
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Require positive observed movement")
    return target


def design(signals, diets, name):
    """Fixed CKA bank average; no sensor selection or post-action measurement."""
    signals, diets = np.asarray(signals, float), np.asarray(diets, float).reshape(-1, 1)
    if signals.shape != (len(diets), 16) or not np.isfinite(signals).all():
        raise ValueError("Aligned sixteen-signal cases required")
    if name not in MODELS:
        raise ValueError(name)
    if name == "mean":
        return np.zeros((len(diets), 0))
    if name == "diet":
        return diets
    cheap = signals[:, 13:16]  # trailing16 loss, gradient, actual encoder step
    cka = signals[:, [3, 8]].mean(1, keepdims=True)
    if name == "cheap":
        return np.c_[diets, cheap, cheap * diets]
    if name == "health_only":
        return np.c_[diets, cka, cka * diets]
    return np.c_[diets, cheap, cheap * diets, cka, cka * diets]


def ridge(train_x, train_y, test_x, penalty=10.0):
    x, y, z = np.asarray(train_x, float), np.asarray(train_y, float), np.asarray(test_x, float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or len(x) != len(y):
        raise ValueError("Aligned training and test matrices required")
    if not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isfinite(y).all() or penalty <= 0:
        raise ValueError("Finite data and positive penalty required")
    center, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1)
    a, b = (x - center) / scale, (z - center) / scale
    coef = np.linalg.solve(a.T @ a + penalty * np.eye(x.shape[1]), a.T @ (y - y.mean(0)))
    return y.mean(0) + b @ coef


def excluded_predictions(signals, diets, groups, targets, penalty=10.0):
    """Exclude test seed AND source AND context, including all repeated diets."""
    groups, targets = np.asarray(groups, int), np.asarray(targets, float)
    if groups.shape != (len(signals), 3) or len(targets) != len(signals):
        raise ValueError("Aligned seed/source/context cases required")
    predicted = {m: np.empty_like(targets) for m in MODELS}
    folds = []
    for group in np.unique(groups, axis=0):
        test = np.all(groups == group, axis=1)
        train = np.all(groups != group, axis=1)
        if train.sum() < 12 or not test.any():
            raise ValueError("Insufficient strictly excluded training cases")
        assert not any(set(groups[train, k]) & set(groups[test, k]) for k in range(3))
        for m in MODELS:
            x = design(signals, diets, m)
            predicted[m][test] = ridge(x[train], targets[train], x[test], penalty)
        folds.append(
            {"group": group.tolist(), "train": np.flatnonzero(train).tolist(), "test": np.flatnonzero(test).tolist()}
        )
    return predicted, folds


def action_values(benefit, predictions):
    """Value relative to always-current; replay chosen only for positive forecast."""
    y, pred = np.asarray(benefit), np.asarray(predictions)
    if y.shape != pred.shape:
        raise ValueError("Aligned paired benefits and predictions required")
    return np.where(pred > 0, y, 0)


def screen(target, predicted, groups, improvement_fraction=0.10):
    """Small preregistered development screen, not population confirmation."""
    y, groups = np.asarray(target), np.asarray(groups)
    cheap = (predicted["cheap"] - y) ** 2
    health = (predicted["health_added"] - y) ** 2
    mse_pass = float(health.mean()) <= (1 - improvement_fraction) * float(cheap.mean())
    per_seed = [float((cheap - health)[groups[:, 0] == s].mean()) for s in np.unique(groups[:, 0])]
    per_source = [float((cheap - health)[groups[:, 1] == s].mean()) for s in np.unique(groups[:, 1])]
    value = action_values(y, predicted["health_added"])
    diet_value = action_values(y, predicted["diet"])
    cheap_value = action_values(y, predicted["cheap"])
    fixed_value = action_values(y, predicted["mean"])
    gain = value - max(float(diet_value.mean()), float(cheap_value.mean()), float(fixed_value.mean()))
    per_seed_value = [float((value - fixed_value)[groups[:, 0] == s].mean()) for s in np.unique(groups[:, 0])]
    return {
        "pooled_MSE_reduction_at_least_10_percent": bool(mse_pass),
        "MSE_improvement_per_seed": per_seed,
        "MSE_improvement_per_source": per_source,
        "forecast_screen_passed": bool(mse_pass and min(per_seed) > 0 and min(per_source) > 0),
        "policy_gain_over_best_fixed_diet_or_cheap_mean_pp": float(gain.mean()),
        "policy_gain_over_training_selected_fixed_per_seed_pp": per_seed_value,
        "action_screen_passed": bool(gain.mean() > 0 and min(per_seed_value) > 0),
        "no_live_controller_qualified": True,
    }
