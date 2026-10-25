"""Chronological linear/nonlinear forecasts of expected isolated image effects."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge


def view_seed(namespace, history, origin, draw):
    value = hashlib.sha256(f"{namespace}|{history}|{origin}|{draw}".encode()).digest()
    return int.from_bytes(value[:4], "little") % (2**31 - 1)


def fit_predict(family, x, y, query, settings):
    """Fit scaling only on admitted labels; use fixed, declared capacity."""
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    yc, ys = y.mean(0), y.std(0)
    ys[ys < 1e-12] = 1
    models = {
        "ridge": lambda: Ridge(alpha=settings["ridge_alpha"]),
        "rbf": lambda: KernelRidge(alpha=settings["rbf_alpha"], kernel="rbf", gamma=1 / x.shape[1]),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=settings["trees"],
            max_depth=settings["tree_depth"],
            min_samples_leaf=settings["tree_min_leaf"],
            max_features=1.0,
            random_state=settings["model_seed"],
            n_jobs=1,
        ),
    }
    model = models[family]()
    model.fit((x - center) / scale, (y - yc) / ys)
    return yc + model.predict((query - center) / scale) * ys


def chronological_forecasts(designs, y, origins, settings):
    """Predict entire same-time groups before revealing any of their labels."""
    prefix = origins + 1 <= settings["prefix_end"]
    inner = origins + 1 <= settings["validation_end"]
    validation = prefix & ~inner
    test = origins >= settings["test_start"]
    if not all(mask.any() for mask in (prefix, inner, validation, test)):
        raise ValueError("Nonempty chronological train, validation and test sets required")
    predictions, validation_scores, fits = {}, {}, {}
    denominator = np.sqrt(np.mean(y[validation] ** 2, axis=0)).clip(1e-15)
    for family in settings["models"]:
        for design in settings["designs"]:
            name, x = f"{family}_{design}", designs[design]
            v = fit_predict(family, x[inner], y[inner], x[validation], settings)
            validation_scores[name] = float((np.sqrt(np.mean((v - y[validation]) ** 2, axis=0)) / denominator).mean())
            predictions[f"frozen_{name}"] = fit_predict(family, x[prefix], y[prefix], x[test], settings)
            online = np.empty_like(y[test])
            for t in np.unique(origins[test]):
                admitted = origins + 1 < t
                current = origins[test] == t
                online[current] = fit_predict(family, x[admitted], y[admitted], x[test][current], settings)
                fits[str(int(t))] = int(admitted.sum())
            predictions[f"online_{name}"] = online
    selected = min(validation_scores, key=validation_scores.get)
    for mode in ("frozen", "online"):
        predictions[f"{mode}_selected"] = predictions[f"{mode}_{selected}"].copy()
    predictions["zero_effect"] = np.zeros_like(y[test])
    predictions["prefix_mean"] = np.broadcast_to(y[prefix].mean(0), y[test].shape).copy()
    predictions["online_mean"] = np.array([y[origins + 1 < t].mean(0) for t in origins[test]])
    return predictions, {
        "selected": selected,
        "validation_scores": validation_scores,
        "prefix_rows": int(prefix.sum()),
        "inner_rows": int(inner.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
        "online_admitted_rows": fits,
    }


def previous_effect(y, groups, origins):
    """Only the latest same-history effect whose update has finished is known."""
    result = np.zeros_like(y)
    for i, (g, t) in enumerate(zip(groups, origins, strict=True)):
        eligible = np.flatnonzero((groups == g) & (origins + 1 < t))
        if len(eligible):
            result[i] = y[eligible[np.argmax(origins[eligible])]]
    return result


def paired_intervals(y, prediction, baseline, groups, draws, seed):
    """Paired loss difference; negative endpoints favor the predictor."""
    labels = np.unique(groups)
    rng = np.random.default_rng(seed)
    weights = np.array(
        [np.bincount(rng.integers(len(labels), size=len(labels)), minlength=len(labels)) for _ in range(draws)]
    )
    result = {}
    for name, power in (("mse", 2), ("mae", 1)):
        delta = np.abs(y - prediction) ** power - np.abs(y - baseline) ** power
        sums = np.array([delta[groups == g].sum(0) for g in labels])
        counts = np.array([(groups == g).sum() for g in labels])
        sample = (weights @ sums) / (weights @ counts)[:, None]
        result[name] = np.quantile(sample, [0.025, 0.975], axis=0).T.tolist()
    return result
