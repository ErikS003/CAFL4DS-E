"""Separate measured health information from prospective regression-based gating."""

from __future__ import annotations

import numpy as np

METRICS = ("variance", "covariance", "rank", "cka", "cosine")
DOMAINS = ("balanced", "day", "night")
PENALTY = 10.0
MODELS = ("mean", "cheap", "history", "measured", "predicted", "plugin", "response_only", "measured_interaction")


def observations(health, parent_updates, past_losses):
    """Only health at 0/4/8 and losses from updates 0..7 may enter runtime inputs."""
    h = np.asarray(health, float)
    if h.shape != (3, 5) or len(past_losses) != 8 or not np.isfinite(h).all():
        raise ValueError("Three past health readings and eight past losses required")
    scale = np.r_[h[2, :3], np.ones(2)]
    scale = np.maximum(scale, 1e-12)
    return np.r_[parent_updates + 8, np.mean(past_losses[-4:]), h[2], (h[2] - h[1]) / scale, (h[1] - h[0]) / scale]


def response(before, after):
    """Relative spread/rank changes; absolute changes in local-parent drift."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    return (b - a) / np.r_[np.maximum(a[:3], 1e-12), np.ones(2)]


def fit(x, y):
    """Fixed multi-output ridge; all normalization is fitted on supplied rows."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = (x - center) / scale
    intercept = y.mean(0)
    beta = np.linalg.solve(z.T @ z + PENALTY * np.eye(x.shape[1]), z.T @ (y - intercept))
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "beta": beta.tolist(),
        "intercept": intercept.tolist(),
        "rows": len(y),
    }


def predict(model, x):
    return np.asarray(model["intercept"]) + ((np.asarray(x) - model["center"]) / model["scale"]) @ np.asarray(
        model["beta"]
    )


def augmented(x, health_response, interaction=False):
    """Observed and forecast responses use identical feature maps."""
    base = np.c_[x, health_response]
    return np.c_[base, x[:, 2:7] * health_response] if interaction else base


def crossfit(x, health_response, utility, groups):
    """No held-history health outcome or utility may enter prospective predictions."""
    x, r, y, groups = map(np.asarray, (x, health_response, utility, groups))
    unique = sorted(set(groups))
    if len(unique) < 4 or x.shape[1] != 17 or r.shape != (len(x), 5) or y.shape != (len(x), 3):
        raise ValueError("At least four source groups and aligned fixed feature shapes required")
    forecasts = {name: np.zeros_like(y, dtype=float) for name in MODELS}
    predicted_health = np.zeros_like(r, dtype=float)
    folds = {}
    for g in unique:
        training, held = groups != g, groups == g
        outer_health = fit(x[training], r[training])
        predicted_health[held] = predict(outer_health, x[held])
        generated = np.zeros_like(r, dtype=float)
        inner_folds = {}
        for k in unique:
            if k == g:
                continue
            selected = groups == k
            allowed = training & ~selected
            model = fit(x[allowed], r[allowed])
            generated[selected] = predict(model, x[selected])
            inner_folds[str(k)] = {"excluded_groups": [str(g), str(k)], "model": model}
        fit_features = {
            "cheap": x[:, :2],
            "history": x,
            "measured": augmented(x, r),
            "predicted": augmented(x, generated),
            "response_only": r,
            "measured_interaction": augmented(x, r, True),
        }
        query_features = {
            **fit_features,
            "predicted": augmented(x, predicted_health),
        }
        utility_models = {}
        forecasts["mean"][held] = y[training].mean(0)
        for name, features in fit_features.items():
            model = fit(features[training], y[training])
            utility_models[name] = model
            forecasts[name][held] = predict(model, query_features[name][held])
        forecasts["plugin"][held] = predict(utility_models["measured"], augmented(x[held], predicted_health[held]))
        folds[str(g)] = {
            "excluded_group": str(g),
            "fit_groups": [str(k) for k in unique if k != g],
            "outer_health": outer_health,
            "inner_health": inner_folds,
            "generated_training_health": generated[training].tolist(),
            "utility_models": utility_models,
        }
    return {
        "predictions": {k: v.tolist() for k, v in forecasts.items()},
        "predicted_health": predicted_health.tolist(),
        "folds": folds,
    }


def accept(predictions, threshold=0.25):
    """A point-prediction shadow policy; these are not calibrated safety bounds."""
    p = np.asarray(predictions)
    return (p[:, 0] >= threshold) & (p[:, 1] >= -1) & (p[:, 2] >= -1)


def source_errors(actual, predicted, groups):
    a, p, g = np.asarray(actual), np.asarray(predicted), np.asarray(groups)
    return {str(k): float(np.mean((a[g == k] - p[g == k]) ** 2)) for k in sorted(set(g))}


def summarize(data, fitted):
    """Keep source histories as uncertainty units and all gates prespecified."""
    from scipy.stats import t

    x, r, y = (np.array(data[k]) for k in ("x", "health_response", "utility"))
    groups = np.array(data["groups"])
    unique = sorted(set(groups))
    predictions = {k: np.array(v) for k, v in fitted["predictions"].items()}

    def interval(values, family=16):
        a = np.asarray(values)
        mean = float(a.mean())
        width = float(t.isf(0.05 / (2 * family), len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
        return {"mean": mean, "ci95": [mean - width, mean + width], "source_values": a.tolist()}

    errors, scores = {}, {}
    for model, p in predictions.items():
        errors[model] = source_errors(y[:, 0], p[:, 0], groups)
        scores[model] = {"mse_pp2": float(np.mean(list(errors[model].values()))), "source_mse": errors[model]}
    errors["zero"] = source_errors(y[:, 0], np.zeros(len(y)), groups)
    scores["zero"] = {"mse_pp2": float(np.mean(list(errors["zero"].values()))), "source_mse": errors["zero"]}
    comparisons = {}
    for model in ("measured", "predicted"):
        comparisons[model] = {}
        for baseline in ("mean", "history"):
            values = [errors[baseline][g] - errors[model][g] for g in unique]
            reduction = 1 - scores[model]["mse_pp2"] / max(scores[baseline]["mse_pp2"], 1e-30)
            comparisons[model][baseline] = {
                "reduction_fraction": reduction,
                "adjusted": interval(values),
                "ordinary": interval(values, 1),
                "passed": bool(reduction >= 0.1 and interval(values)["ci95"][0] > 0),
            }
    choices = {name: accept(predictions[name]) for name in ("history", "predicted", "measured")}
    outcomes = {name: choices[name][:, None] * y for name in choices}
    outcomes.update({"accept_all": y, "hold_all": np.zeros_like(y)})
    expected_random = np.zeros_like(y)
    for g in unique:
        selected = groups == g
        expected_random[selected] = np.mean(choices["predicted"][selected]) * np.mean(y[selected], axis=0)
    outcomes["matched_random_expectation"] = expected_random
    policies = {
        k: {
            "mean_gain_vs_hold_pp": v.mean(0).tolist(),
            "accepted_rows": int(choices[k].sum())
            if k in choices
            else (len(y) if k == "accept_all" else 0 if k == "hold_all" else None),
        }
        for k, v in outcomes.items()
    }
    policy_contrasts = {}
    for baseline in ("accept_all", "hold_all", "history", "matched_random_expectation"):
        policy_contrasts[baseline] = {}
        for index, domain in enumerate(DOMAINS):
            values = [
                float(np.mean(outcomes["predicted"][groups == g, index] - outcomes[baseline][groups == g, index]))
                for g in unique
            ]
            policy_contrasts[baseline][domain] = {**interval(values), "ordinary_ci95": interval(values, 1)["ci95"]}
    identities = data["identities"]
    parent_ids = sorted({(i["stream"], i["checkpoint"]) for i in identities})
    parent_means = [
        np.mean(y[[j for j, i in enumerate(identities) if (i["stream"], i["checkpoint"]) == key]], axis=0)
        for key in parent_ids
    ]
    population = {
        "beneficial_parents": sum(v[0] >= 0.5 and min(v[1:]) >= -1 for v in parent_means),
        "harmful_balanced_parents": sum(v[0] <= -0.5 for v in parent_means),
    }
    gates = {
        "two_sided_population": all(v >= 2 for v in population.values()),
        "measured_health_information": all(v["passed"] for v in comparisons["measured"].values()),
        "regression_health_pipeline": all(v["passed"] for v in comparisons["predicted"].values()),
        "shadow_policy_gain": all(
            v["balanced"]["mean"] >= 0.25 and v["balanced"]["ci95"][0] > 0 for v in policy_contrasts.values()
        ),
        "shadow_domain_retention": all(
            v[d]["ci95"][0] >= -1 for v in policy_contrasts.values() for d in ("day", "night")
        ),
    }
    health = {
        m: {
            "mse": float(
                np.mean(list(source_errors(r[:, j], np.array(fitted["predicted_health"])[:, j], groups).values()))
            ),
            "zero_mse": float(np.mean(list(source_errors(r[:, j], np.zeros(len(r)), groups).values()))),
            "recent_trend_mse": float(np.mean(list(source_errors(r[:, j], 2 * x[:, 7 + j], groups).values()))),
        }
        for j, m in enumerate(METRICS)
    }
    stability = []
    for key in parent_ids:
        indices = [j for j, i in enumerate(identities) if (i["stream"], i["checkpoint"]) == key]
        seed_values = y[indices, 0]
        panel_values = np.array(data["panel_utility"])[indices, :, 0].mean(0)
        stability.append(
            {
                "stream": key[0],
                "checkpoint": key[1],
                "mean_balanced_pp": float(seed_values.mean()),
                "seed_values_pp": seed_values.tolist(),
                "panel_means_pp": panel_values.tolist(),
                "positive_across_seeds_and_panels": bool(min(seed_values) > 0 and min(panel_values) > 0),
                "negative_across_seeds_and_panels": bool(max(seed_values) < 0 and max(panel_values) < 0),
            }
        )
    tradeoff = {}
    for threshold in (0.0, 0.25, 0.5, 1.0):
        selected = accept(predictions["predicted"], threshold)
        tradeoff[str(threshold)] = {
            "accepted": int(selected.sum()),
            "mean_gain_vs_hold_pp": (selected[:, None] * y).mean(0).tolist(),
        }
    good = (y[:, 0] >= 0.5) & (np.min(y[:, 1:], axis=1) >= -1)
    bad = (y[:, 0] <= -0.5) | (np.min(y[:, 1:], axis=1) < -1)
    a = choices["predicted"]
    classification = {
        "good_rows": int(good.sum()),
        "bad_rows": int(bad.sum()),
        "neutral_rows": int((~good & ~bad).sum()),
        "accepted_bad": int((a & bad).sum()),
        "rejected_good": int((~a & good).sum()),
        "acceptance_is_not_confidence": True,
    }
    return {
        "scores": scores,
        "comparisons": comparisons,
        "health_prediction": health,
        "policies": policies,
        "policy_contrasts": policy_contrasts,
        "population": population,
        "classification": classification,
        "parent_sign_stability": stability,
        "threshold_tradeoff_secondary": tradeoff,
        "gates": gates,
        "development_screen_passed": all(gates.values()),
        "deployable_filter": False,
        "fresh_confirmation": False,
        "measured_health_is_post_action_diagnostic": True,
        "primary_scalar_family": 16,
    }
