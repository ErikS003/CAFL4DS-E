"""Probabilistic direction forecasts with causal prefix fitting and view-aware scores."""

from __future__ import annotations

import numpy as np
from hydra.utils import instantiate
from sklearn.metrics import roc_auc_score


def event_counts(delta, tolerance=1e-10):
    delta = np.asarray(delta)
    return np.column_stack(((delta < -tolerance).sum(1), (delta > tolerance).sum(1)))


def probabilities(counts):
    active = counts.sum(1)
    return np.divide(counts[:, 1], active, out=np.zeros(len(counts), dtype=float), where=active > 0)


def rates(counts, phases=None, query_phases=None):
    if phases is None:
        return float((counts[:, 1].sum() + 1) / (counts.sum() + 2))
    return np.array([rates(counts[np.asarray(phases) == phase]) for phase in query_phases])


def fit_probability(family, x, counts, query, settings):
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    if np.any(counts.sum(0) == 0):
        return np.full(len(query), rates(counts))
    xx = np.repeat((x - center) / scale, 2, axis=0)
    yy = np.tile([0, 1], len(x))
    weights = counts.ravel().astype(float) / 6
    model = instantiate(settings["estimators"][family])
    model.fit(xx, yy, sample_weight=weights)
    return model.predict_proba((query - center) / scale)[:, 1]


def score(counts, p, threshold=0.5):
    negative, positive = counts[:, 0].astype(float), counts[:, 1].astype(float)
    total = negative + positive
    decision = np.asarray(p) >= threshold
    tp, tn = positive[decision].sum(), negative[~decision].sum()
    fp, fn = negative[decision].sum(), positive[~decision].sum()
    recall = tp / positive.sum() if positive.sum() else None
    specificity = tn / negative.sum() if negative.sum() else None
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    brier = np.sum(positive * (1 - p) ** 2 + negative * p**2) / total.sum()
    p = np.clip(p, 1e-12, 1 - 1e-12)
    log_loss = -np.sum(positive * np.log(p) + negative * np.log1p(-p)) / total.sum()
    labels, prediction, weights = np.tile([0, 1], len(p)), np.repeat(p, 2), counts.ravel()
    auc = float(roc_auc_score(labels, prediction, sample_weight=weights)) if counts.sum(0).min() > 0 else None
    return dict(
        brier=float(brier),
        log_loss=float(log_loss),
        balanced_accuracy=None if balanced is None else float(balanced),
        accuracy=float((tp + tn) / total.sum()),
        increase_precision=float(tp / (tp + fp)) if tp + fp else None,
        increase_recall=None if recall is None else float(recall),
        decrease_recall=None if specificity is None else float(specificity),
        auc=auc,
        active_views=int(total.sum()),
        parents=len(counts),
        true_increase_rate=float(positive.sum() / total.sum()),
        predicted_increase_fraction=float(decision.mean()),
        confusion=dict(tp=float(tp), tn=float(tn), fp=float(fp), fn=float(fn)),
    )


def balanced_threshold(counts, p, settings):
    candidates = np.arange(settings["threshold_min"], settings["threshold_max"] + 1e-8, settings["threshold_step"])
    candidates = np.round(candidates, 8)
    scores = [score(counts, p, threshold)["balanced_accuracy"] for threshold in candidates]
    if all(v is None for v in scores):
        return 0.5
    return float(
        max(
            zip(candidates, scores),
            key=lambda item: (-1 if item[1] is None else item[1], -abs(item[0] - 0.5), -item[0]),
        )[0]
    )


def forecasts(designs, counts, seeds, origins, settings):
    inner = origins + 1 <= settings["inner_end"]
    prefix = origins + 1 <= settings["prefix_end"]
    validation, test = prefix & ~inner, origins >= settings["test_start"]
    out, validation_predictions, family_scores, thresholds = {}, {}, {}, {}
    for design, x in designs.items():
        for family in settings["families"]:
            name = f"{family}_{design}"
            v, prediction = np.zeros(validation.sum()), np.zeros(test.sum())
            for seed in np.unique(seeds):
                fit, valid = inner & (seeds == seed), validation & (seeds == seed)
                v[seeds[validation] == seed] = fit_probability(family, x[fit], counts[fit], x[valid], settings)
                fit, query = prefix & (seeds == seed), test & (seeds == seed)
                prediction[seeds[test] == seed] = fit_probability(family, x[fit], counts[fit], x[query], settings)
            out[name], validation_predictions[name] = prediction, v
            family_scores[name] = score(counts[validation], v)["brier"]
            thresholds[name] = balanced_threshold(counts[validation], v, settings)
    for name in ("prefix_rate", "phase_rate"):
        v, prediction = np.zeros(validation.sum()), np.zeros(test.sum())
        for seed in np.unique(seeds):
            for fit_mask, query_mask, dest in ((inner, validation, v), (prefix, test, prediction)):
                fit, query = fit_mask & (seeds == seed), query_mask & (seeds == seed)
                slot = seeds[query_mask] == seed
                dest[slot] = (
                    rates(counts[fit])
                    if name == "prefix_rate"
                    else rates(counts[fit], origins[fit] % settings["period"], origins[query] % settings["period"])
                )
        out[name], validation_predictions[name] = prediction, v
        thresholds[name] = balanced_threshold(counts[validation], v, settings)
    selected = {}
    for design in designs:
        name = min((f"{f}_{design}" for f in settings["families"]), key=family_scores.get)
        selected[design] = name
        out[f"selected_{design}"] = out[name].copy()
        thresholds[f"selected_{design}"] = thresholds[name]
    out["always_decrease"], out["always_increase"] = np.zeros(test.sum()), np.ones(test.sum())
    thresholds["always_decrease"], thresholds["always_increase"] = 0.5, 0.5
    return (
        out,
        dict(
            selected=selected,
            validation_brier=family_scores,
            thresholds=thresholds,
            prefix_parents=int(prefix.sum()),
            inner_parents=int(inner.sum()),
            validation_parents=int(validation.sum()),
            test_parents=int(test.sum()),
        ),
        validation_predictions,
    )


def paired_intervals(counts, p, threshold, control, control_threshold, groups, draws, seed):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    choices = rng.integers(len(unique), size=(draws, len(unique)))

    # Sufficient statistics preserve all view counts and paired predictions.
    def stats(probability, cut):
        decision = probability >= cut
        pos, neg = counts[:, 1], counts[:, 0]
        loss = pos * (1 - probability) ** 2 + neg * probability**2
        row = np.column_stack((loss, pos + neg, pos * decision, pos, neg * ~decision, neg))
        return np.array([row[groups == g].sum(0) for g in unique])

    a, b = stats(p, threshold)[choices].sum(1), stats(control, control_threshold)[choices].sum(1)
    brier = a[:, 0] / a[:, 1] - b[:, 0] / b[:, 1]
    balanced = 0.5 * (a[:, 2] / a[:, 3] + a[:, 4] / a[:, 5] - b[:, 2] / b[:, 3] - b[:, 4] / b[:, 5])
    finite = np.isfinite(balanced)
    return dict(
        brier_difference=np.quantile(brier, [0.025, 0.975]).tolist(),
        balanced_accuracy_difference=np.quantile(balanced[finite], [0.025, 0.975]).tolist() if finite.any() else None,
        valid_balanced_bootstrap_draws=int(finite.sum()),
    )
