"""Frozen singleton-effect forecasting with explicit known-transition controls."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def transition_features(origins, period=96):
    origins = np.asarray(origins)
    if np.any(origins < 0) or period <= 0:
        raise ValueError("Nonnegative origins and positive period required")
    return np.column_stack((origins % period == 0, origins % period)).astype(float)


def last_effect(y, histories, seeds, origins):
    previous = np.zeros_like(y)
    for i, (g, seed, t) in enumerate(zip(histories, seeds, origins, strict=True)):
        eligible = np.flatnonzero((histories == g) & (seeds == seed) & (origins + 1 < t))
        if len(eligible):
            previous[i] = y[eligible[np.argmax(origins[eligible])]]
    return previous


def forest_fit(x, y, query, settings):
    xc, xs, yc, ys = x.mean(0), x.std(0), y.mean(0), y.std(0)
    xs[xs < 1e-12], ys[ys < 1e-12] = 1, 1
    forest = ExtraTreesRegressor(
        n_estimators=settings["trees"],
        max_depth=settings["tree_depth"],
        min_samples_leaf=settings["tree_min_leaf"],
        max_features=1.0,
        random_state=settings["model_seed"],
        n_jobs=1,
    )
    forest.fit((x - xc) / xs, (y - yc) / ys)
    return yc + forest.predict((query - xc) / xs) * ys


def frozen_predictions(designs, y, seeds, origins, previous, settings):
    prefix, test = origins + 1 <= settings["prefix_end"], origins >= settings["test_start"]
    predictions = {
        name: np.zeros_like(y[test])
        for name in (
            *designs,
            "zero_effect",
            "phase_mean",
            "prefix_mean",
            "previous_effect",
            "majority_sign",
            "phase_majority_sign",
        )
    }
    for seed in np.unique(seeds):
        fit, query = prefix & (seeds == seed), test & (seeds == seed)
        slot = seeds[test] == seed
        if not fit.any() or not query.any():
            raise ValueError("Every trajectory seed needs prefix and test rows")
        for name, x in designs.items():
            predictions[name][slot] = forest_fit(x[fit], y[fit], x[query], settings)
        predictions["prefix_mean"][slot] = y[fit].mean(0)
        predictions["previous_effect"][slot] = previous[query]
        # A sign forecast has no magnitude interpretation; exclude from RMSE tables.
        signs = np.where(np.abs(y[fit]) <= 1e-10, 0, np.sign(y[fit]))
        majority = np.array([max((-1, 0, 1), key=lambda c: int((signs[:, j] == c).sum())) for j in range(y.shape[1])])
        predictions["majority_sign"][slot] = majority
        for phase in np.unique(origins[query] % settings["frames_per_clip"]):
            same = fit & (origins % settings["frames_per_clip"] == phase)
            dest = slot & (origins[test] % settings["frames_per_clip"] == phase)
            if not same.any():
                raise ValueError("Test phase missing from prefix")
            predictions["phase_mean"][dest] = y[same].mean(0)
            signs = np.where(np.abs(y[same]) <= 1e-10, 0, np.sign(y[same]))
            predictions["phase_majority_sign"][dest] = [
                max((-1, 0, 1), key=lambda c: int((signs[:, j] == c).sum())) for j in range(y.shape[1])
            ]
    return predictions


def cluster_intervals(y, prediction, control, histories, draws, seed):
    """Keep all optimizer seeds and serial arrivals from a source history together."""
    unique = np.unique(histories)
    rng = np.random.default_rng(seed)
    choices = rng.integers(len(unique), size=(draws, len(unique)))
    sizes = np.array([(histories == g).sum() for g in unique])
    result = {}
    for loss, power in (("mse", 2), ("mae", 1)):
        delta = np.abs(prediction - y) ** power - np.abs(control - y) ** power
        totals = np.array([delta[histories == g].sum(0) for g in unique])
        sampled = totals[choices].sum(1) / sizes[choices].sum(1)[:, None]
        result[loss] = np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist()
    return result
