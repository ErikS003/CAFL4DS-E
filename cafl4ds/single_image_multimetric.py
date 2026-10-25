"""Causal singleton forecasts that retain unchanged downstream measurements."""

from __future__ import annotations

import numpy as np
from hydra.utils import instantiate


def counts(delta, tolerance):
    delta = np.asarray(delta)
    if delta.ndim != 2 or not np.isfinite(delta).all():
        raise ValueError("Require finite parent-by-view changes")
    return np.column_stack(((delta < -tolerance).sum(1), (abs(delta) <= tolerance).sum(1), (delta > tolerance).sum(1)))


def rate(c):
    return (c.sum(0) + 1) / (c.sum() + 3)


def table(c, key, query):
    return np.array([rate(c[np.all(key == k, axis=1)]) if np.any(np.all(key == k, axis=1)) else rate(c) for k in query])


def fit(kind, family, x, delta, query, settings):
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    xx, qq = (x - center) / scale, (query - center) / scale
    model = instantiate(settings["classifiers" if kind == "classification" else "regressors"][family])
    if kind == "regression":
        y = delta.mean(1)
        yc, ys = y.mean(), y.std()
        if ys < 1e-12:
            return np.full(len(query), yc)
        model.fit(xx, (y - yc) / ys)
        return yc + ys * model.predict(qq)
    c = counts(delta, settings["sign_tolerance"])
    if (c.sum(0) > 0).sum() < 2:
        return np.tile(rate(c), (len(query), 1))
    weights = (c / delta.shape[1]).ravel()
    active = weights > 0
    model.fit(np.repeat(xx, 3, axis=0)[active], np.tile(np.arange(3), len(x))[active], sample_weight=weights[active])
    result = np.zeros((len(query), 3))
    result[:, model.classes_.astype(int)] = model.predict_proba(qq)
    return result


def sign_probability(p):
    return np.divide(p[:, 2], p[:, 0] + p[:, 2], out=np.full(len(p), 0.5), where=p[:, 0] + p[:, 2] > 0)


def class_score(c, p, threshold=0.5):
    n = c.sum()
    decision = p.argmax(1)
    confusion = np.array([c[decision == k].sum(0) for k in range(3)]).T
    totals = c.sum(0)
    recalls = np.divide(confusion.diagonal(), totals, out=np.full(3, np.nan), where=totals > 0)
    active = c[:, 0] + c[:, 2]
    q = sign_probability(p)
    up = q >= threshold
    recall_up = float(c[up, 2].sum() / totals[2]) if totals[2] else None
    recall_down = float(c[~up, 0].sum() / totals[0]) if totals[0] else None
    brier = ((p**2).sum(1) * c.sum(1) - 2 * (p * c).sum(1) + c.sum(1)) / 2
    changed = int(active.sum())
    return dict(
        parents=len(c),
        views=int(n),
        changed_views=changed,
        changed_fraction=float(changed / n),
        tie_fraction=float(totals[1] / n),
        class_counts=totals.astype(int).tolist(),
        accuracy=float(confusion.diagonal().sum() / n),
        multiclass_balanced_accuracy=float(np.nanmean(recalls)),
        brier=float(brier.sum() / n),
        log_loss=float(-(c * np.log(np.clip(p, 1e-12, 1))).sum() / n),
        change_brier=float((active * p[:, 1] ** 2 + c[:, 1] * (1 - p[:, 1]) ** 2).sum() / n),
        direction_balanced_accuracy=(recall_up + recall_down) / 2
        if recall_up is not None and recall_down is not None
        else None,
        direction_accuracy=float((c[up, 2].sum() + c[~up, 0].sum()) / changed) if changed else None,
        increase_recall=recall_up,
        decrease_recall=recall_down,
        direction_brier=float((c[:, 0] * q**2 + c[:, 2] * (1 - q) ** 2).sum() / changed) if changed else None,
        confusion=confusion.astype(int).tolist(),
    )


def regression_score(delta, prediction):
    error = prediction[:, None] - delta
    mean_error = prediction - delta.mean(1)
    return dict(
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(abs(error).mean()),
        mean_rmse=float(np.sqrt(np.mean(mean_error**2))),
        mean_mae=float(abs(mean_error).mean()),
    )


def threshold(c, p):
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 8)
    values = [class_score(c, p, t)["direction_balanced_accuracy"] for t in candidates]
    return float(
        max(
            zip(candidates, values, strict=True), key=lambda v: (-1 if v[1] is None else v[1], -abs(v[0] - 0.5), -v[0])
        )[0]
    )


def forecast(designs, delta, lag, origins, fitting_group, prefix_pool, test, settings):
    """Select on an early validation suffix; query labels never enter any fit."""
    prefix = prefix_pool & (origins + 1 <= settings["prefix_end"])
    inner = prefix_pool & (origins + 1 <= settings["inner_end"])
    valid = prefix & ~inner
    predictions, metadata = {}, {}
    for kind in ("classification", "regression"):
        candidates, validation, selection_scores = {}, {}, {}
        c = counts(delta[valid], settings["sign_tolerance"])
        for design, x in designs.items():
            for family in ("linear", "nonlinear"):
                name = family + "_" + design
                shape = (3,) if kind == "classification" else ()
                query = np.empty((test.sum(), *shape))
                v = np.empty((valid.sum(), *shape))
                for group in np.unique(fitting_group):
                    for admit, ask, dest in ((inner, valid, v), (prefix, test, query)):
                        fitting = admit & (fitting_group == group)
                        querying = ask & (fitting_group == group)
                        if querying.any():
                            dest[fitting_group[ask] == group] = fit(
                                kind, family, x[fitting], delta[fitting], x[querying], settings
                            )
                candidates[name], validation[name] = query, v
                selection_scores[name] = (
                    class_score(c, v)["brier"]
                    if kind == "classification"
                    else float(np.mean((v - delta[valid].mean(1)) ** 2))
                )
        selected = {}
        for design in designs:
            winner = min((f + "_" + design for f in ("linear", "nonlinear")), key=selection_scores.get)
            name = "selected_" + design
            selected[design] = winner
            candidates[name], validation[name] = candidates[winner].copy(), validation[winner].copy()

        # All controls use the same causal seed-specific early prefix.
        for name in ("prefix", "phase", "persistence", "phase_persistence"):
            shape = (3,) if kind == "classification" else ()
            q, v = np.empty((test.sum(), *shape)), np.empty((valid.sum(), *shape))
            for group in np.unique(fitting_group):
                for admit, ask, dest in ((inner, valid, v), (prefix, test, q)):
                    a, b = admit & (fitting_group == group), ask & (fitting_group == group)
                    if not b.any():
                        continue
                    ac = counts(delta[a], settings["sign_tolerance"])
                    phase, next_phase = (
                        (origins[a] % settings["period"])[:, None],
                        (origins[b] % settings["period"])[:, None],
                    )
                    lag_sign = np.where(abs(lag) <= settings["sign_tolerance"], 0, np.sign(lag))
                    sign, next_sign = lag_sign[a, None], lag_sign[b, None]
                    if kind == "classification":
                        if name == "prefix":
                            values = np.tile(rate(ac), (b.sum(), 1))
                        else:
                            key, ask_key = (phase, next_phase) if name == "phase" else (sign, next_sign)
                            if name == "phase_persistence":
                                key, ask_key = np.c_[sign, phase], np.c_[next_sign, next_phase]
                            values = table(ac, key, ask_key)
                    elif name in ("persistence", "phase_persistence"):
                        values = lag[b]
                    elif name == "prefix":
                        values = np.full(b.sum(), delta[a].mean())
                    else:
                        values = np.array([delta[a][phase[:, 0] == t].mean() for t in next_phase[:, 0]])
                    dest[fitting_group[ask] == group] = values
            candidates[name], validation[name] = q, v
        if kind == "classification":
            chosen = min(("persistence", "phase_persistence"), key=lambda n: class_score(c, validation[n])["brier"])
            candidates["calibrated_persistence"] = candidates[chosen].copy()
            validation["calibrated_persistence"] = validation[chosen].copy()
            cuts = {n: threshold(c, v) for n, v in validation.items()}
            candidates["no_change"] = np.tile([0.0, 1.0, 0.0], (test.sum(), 1))
            cuts["no_change"] = 0.5
            metadata[kind] = dict(
                selected=selected, validation_scores=selection_scores, thresholds=cuts, persistence_selected=chosen
            )
        else:
            multipliers = (0, 0.25, 0.5, 0.75, 1)
            factor = min(multipliers, key=lambda v: np.mean((v * lag[valid] - delta[valid].mean(1)) ** 2))
            candidates["calibrated_persistence"] = factor * lag[test]
            candidates["no_change"] = np.zeros(test.sum())
            metadata[kind] = dict(selected=selected, validation_scores=selection_scores, persistence_multiplier=factor)
        predictions[kind] = candidates
    metadata["rows"] = dict(
        prefix=int(prefix.sum()), inner=int(inner.sum()), validation=int(valid.sum()), test=int(test.sum())
    )
    return predictions, metadata


def intervals(delta, a, b, cut_a, cut_b, groups, settings, kind):
    """Bootstrap complete source histories, retaining views and all SSL seeds."""
    unique = np.unique(groups)
    choices = np.random.default_rng(settings["bootstrap_seed"]).integers(
        len(unique), size=(settings["bootstrap_draws"], len(unique))
    )
    if kind == "regression":
        ea, eb = a[:, None] - delta, b[:, None] - delta
        rows = np.column_stack(
            ((ea**2 - eb**2).sum(1), (abs(ea) - abs(eb)).sum(1), np.full(len(delta), delta.shape[1]))
        )
        grouped = np.array([rows[groups == g].sum(0) for g in unique])[choices].sum(1)
        return {
            name: np.quantile(grouped[:, i] / grouped[:, 2], [0.025, 0.975]).tolist()
            for i, name in enumerate(("mse", "mae"))
        }
    c = counts(delta, settings["sign_tolerance"])

    def stats(p, cut):
        q = sign_probability(p) >= cut
        loss = ((p**2).sum(1) * c.sum(1) - 2 * (p * c).sum(1) + c.sum(1)) / 2
        return np.column_stack((loss, c.sum(1), c[:, 2] * q, c[:, 2], c[:, 0] * ~q, c[:, 0]))

    aa, bb = [
        np.array([v[groups == g].sum(0) for g in unique])[choices].sum(1) for v in (stats(a, cut_a), stats(b, cut_b))
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * (aa[:, 2] / aa[:, 3] + aa[:, 4] / aa[:, 5] - bb[:, 2] / bb[:, 3] - bb[:, 4] / bb[:, 5])
    finite = np.isfinite(d)
    return dict(
        brier=np.quantile(aa[:, 0] / aa[:, 1] - bb[:, 0] / bb[:, 1], [0.025, 0.975]).tolist(),
        direction_balanced_accuracy=np.quantile(d[finite], [0.025, 0.975]).tolist() if finite.any() else None,
        valid_direction_draws=int(finite.sum()),
    )


def evaluate(delta, prediction, metadata, groups, origins, settings, primary, seeds=None):
    result = {}
    for context, mask in (
        ("all", np.ones(len(delta), bool)),
        ("clip_join", origins % settings["period"] == 0),
        ("within_clip", origins % settings["period"] != 0),
    ):
        c = counts(delta[mask], settings["sign_tolerance"])
        scores, comparisons = {}, {}
        for kind, models in prediction.items():
            scores[kind] = {}
            for name, values in models.items():
                scores[kind][name] = (
                    regression_score(delta[mask], values[mask])
                    if kind == "regression"
                    else class_score(c, values[mask], metadata[kind]["thresholds"][name])
                )
                if kind == "classification":
                    signed = mask & (values[:, [0, 2]].max(1) >= settings["confidence_threshold"])
                    scores[kind][name]["signed_confidence"] = dict(
                        parent_coverage=float(signed.sum() / mask.sum()),
                        score=class_score(counts(delta[signed], settings["sign_tolerance"]), values[signed], 0.5)
                        if signed.any()
                        else None,
                    )
            comparisons[kind] = {}
            for control in (primary[0], "calibrated_persistence", "phase", "no_change"):
                name = primary[1]
                cuts = metadata["classification"]["thresholds"]
                comparisons[kind][control] = intervals(
                    delta[mask],
                    models[name][mask],
                    models[control][mask],
                    cuts.get(name, 0.5),
                    cuts.get(control, 0.5),
                    groups[mask],
                    settings,
                    kind,
                )
        rs, cs = scores["regression"], scores["classification"]
        name = primary[1]
        magnitude = all(
            rs[name]["rmse"] < rs[k]["rmse"]
            and rs[name]["mae"] < rs[k]["mae"]
            and ci["mse"][1] < 0
            and ci["mae"][1] < 0
            for k, ci in comparisons["regression"].items()
        )
        sign = cs[name]["direction_balanced_accuracy"]
        direction = (
            sign is not None
            and cs[name]["changed_views"] >= 20
            and min(cs[name]["increase_recall"], cs[name]["decrease_recall"]) >= 0.1
        )
        for k in (primary[0], "calibrated_persistence", "phase"):
            ci, base = comparisons["classification"][k], cs[k]
            direction = bool(
                direction
                and base["direction_balanced_accuracy"] is not None
                and sign > base["direction_balanced_accuracy"]
                and cs[name]["brier"] < base["brier"]
                and ci["brier"][1] < 0
                and ci["direction_balanced_accuracy"] is not None
                and ci["direction_balanced_accuracy"][0] > 0
            )
        per_seed = {}
        if seeds is not None and len(np.unique(seeds)) > 1:
            for seed in np.unique(seeds):
                take = mask & (seeds == seed)
                per_seed[str(seed)] = dict(
                    classification=class_score(
                        counts(delta[take], settings["sign_tolerance"]),
                        prediction["classification"][name][take],
                        metadata["classification"]["thresholds"][name],
                    ),
                    regression=regression_score(delta[take], prediction["regression"][name][take]),
                )
        result[context] = dict(
            scores=scores,
            comparisons=comparisons,
            per_seed=per_seed,
            exploratory_screens=dict(magnitude=bool(magnitude), direction=bool(direction)),
        )
    return result
