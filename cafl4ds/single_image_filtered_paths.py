"""Frozen accuracy forecasts from full and randomly filtered development paths."""

from __future__ import annotations

import numpy as np

from cafl4ds.single_image_filter import FrozenRegressor
from cafl4ds.single_image_nonlinear import view_seed

TARGETS = ("accuracy", "soft_accuracy", "CE", "Brier")
RECIPES = ("full", "mixed")
DESIGNS = ("state", "base")
MAIN = ("full_accuracy", "mixed_accuracy", "full_surrogate", "mixed_surrogate")
ARMS = (*MAIN, *("random_" + arm for arm in MAIN), "full", "hold")


def probe_scores(probabilities, labels, temperature=0.05):
    """Support-panel mean scores, all oriented so larger means better."""
    p = np.asarray(probabilities, float)
    labels = np.asarray(labels, int)
    if p.ndim != 4 or p.shape[1] != 2 or p.shape[2:] != (len(labels), 2):
        raise ValueError("Expected panel x reader x query x class probabilities")
    if temperature <= 0 or not np.isfinite(p).all():
        raise ValueError("Finite probabilities and positive temperature required")
    linear = p[:, 0]
    truth = np.take_along_axis(linear, labels[None, :, None], axis=-1)[..., 0]
    soft = 1 / (1 + np.exp(-(truth - 0.5) / temperature))
    return np.array(
        [
            100 * np.mean(linear.argmax(-1) == labels),
            100 * soft.mean(),
            np.log(np.clip(truth, 1e-12, 1)).mean(),
            -np.square(1 - truth).mean(),
        ]
    )


def accuracy_slope(native, accuracy, target):
    """Training-only, positive, zero-intercept mapping to accuracy points."""
    if target == "accuracy":
        return 1.0
    native, accuracy = np.asarray(native), np.asarray(accuracy)
    denominator = float(native @ native)
    return max(0.0, float(native @ accuracy) / denominator) if denominator > 1e-24 else 0.0


def predict(model, x):
    return model["slope"] * model["regressor"].predict(x)


def threshold(predictions, rate):
    predictions = np.asarray(predictions, float)
    if not len(predictions) or not np.isfinite(predictions).all() or not 0 < rate < 1:
        raise ValueError("Finite predictions and an interior admission rate required")
    return float(max(0, np.quantile(predictions, 1 - rate)))


def development_schedule(settings, history, seed):
    """Equal parent sampling; treatment masks are independent of outcomes."""
    n, period = settings["stop"], settings["frames_per_clip"]
    rng = np.random.default_rng(view_seed(settings["development_namespace"], history, 0, seed))
    offset = int(rng.integers(0, 3))
    mask, rates, origins = np.ones(n, bool), np.ones(n), []
    for clip in range(settings["development_start"] // period, n // period):
        rate = settings["development_probabilities"][(clip + offset) % 3]
        indices = np.arange(clip * period, (clip + 1) * period)
        mask[indices] = rng.random(period) < rate
        rates[indices] = rate
        private = np.random.default_rng(view_seed(settings["development_namespace"] + "-parents", history, clip, seed))
        origins.extend(clip * period + stratum * 32 + int(private.integers(0, 32)) for stratum in range(3))
    return np.asarray(origins), mask, rates


def fit_forecasters(a, settings):
    """History validation selects models and one shared continuous surrogate."""
    validation = np.isin(a["histories"], settings["validation_histories"])
    assert validation.any() and (~validation).any()
    models, selection = {}, {}
    for recipe in RECIPES:
        for design in DESIGNS:
            for ti, target in enumerate(TARGETS):
                key = f"{recipe}/{design}/{target}"
                native, actual = a["targets"][:, ti], a["targets"][:, 0]
                is_recipe = a["recipes"] == recipe
                valid = is_recipe & validation
                predictions, native_predictions, losses, native_losses = {}, {}, {}, {}
                for family in settings["regressors"]:
                    predictions[family], native_predictions[family] = np.zeros(valid.sum()), np.zeros(valid.sum())
                    for seed in settings["trajectory_seeds"]:
                        fit = is_recipe & ~validation & (a["seeds"] == seed)
                        ask = valid & (a["seeds"] == seed)
                        regressor = FrozenRegressor(family, a[design][fit], native[fit], settings)
                        p = regressor.predict(a[design][ask])
                        take = a["seeds"][valid] == seed
                        native_predictions[family][take] = p
                        predictions[family][take] = p * accuracy_slope(native[fit], actual[fit], target)
                    losses[family] = float(np.mean((predictions[family] - actual[valid]) ** 2))
                    native_losses[family] = float(np.mean((native_predictions[family] - native[valid]) ** 2))
                winner = min(losses, key=losses.get)
                selection[key] = dict(
                    family=winner,
                    validation_accuracy_mse=losses,
                    validation_native_mse=native_losses,
                    predictions=predictions[winner].tolist(),
                    native_predictions=native_predictions[winner].tolist(),
                    actual_accuracy=actual[valid].tolist(),
                    native_targets=native[valid].tolist(),
                    validation_histories=a["histories"][valid].tolist(),
                    validation_seeds=a["seeds"][valid].tolist(),
                    threshold=threshold(predictions[winner], settings["max_validation_admission"]),
                    validation_rows=int(valid.sum()),
                )
                models[key] = {}
                for seed in settings["trajectory_seeds"]:
                    fit = is_recipe & (a["seeds"] == seed)
                    models[key][seed] = dict(
                        regressor=FrozenRegressor(winner, a[design][fit], native[fit], settings),
                        slope=accuracy_slope(native[fit], actual[fit], target),
                    )
    continuous_scores = {
        target: float(
            np.mean(
                [min(selection[f"{recipe}/base/{target}"]["validation_accuracy_mse"].values()) for recipe in RECIPES]
            )
        )
        for target in TARGETS[1:]
    }
    chosen = min(continuous_scores, key=continuous_scores.get)
    gates = {}
    for arm in MAIN:
        recipe, objective = arm.split("_")
        target = "accuracy" if objective == "accuracy" else chosen
        key = f"{recipe}/base/{target}"
        gates[arm] = dict(key=key, target=target, threshold=selection[key]["threshold"])
    return models, dict(models=selection, continuous_scores=continuous_scores, chosen_surrogate=chosen, gates=gates)
