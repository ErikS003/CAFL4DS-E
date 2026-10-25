"""Task-anchored, rotation-invariant retention health and held-history pacing diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np

from cafl4ds.health_regression_gate import fit, predict, observations
from cafl4ds.prediction_benchmark import correlation
from cafl4ds.collapse_risk_regression import interval


def plain(value):
    """Normalize scientific containers once, at the strict JSON boundary."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    return value


def unit_rows(z):
    x = np.asarray(z, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite embedding matrix required")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def invariant_ridge(x, y, query, penalty=1.0):
    """One fitting-only scalar scale preserves orthogonal-coordinate invariance."""
    x, query, y = unit_rows(x), unit_rows(query), np.asarray(y, float)
    center = x.mean(0)
    scale = float(np.sqrt(np.mean(np.sum((x - center) ** 2, axis=1))))
    if scale < 1e-12:
        scale = 1.0
    z = (x - center) / scale
    intercept = y.mean(0)
    beta = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y - intercept))
    return intercept + (query - center) / scale @ beta, {
        "center": center,
        "scale": scale,
        "intercept": intercept,
        "beta": beta,
        "rows": len(x),
        "penalty": penalty,
    }


def bank_folds(identities, count=4):
    if len(set(identities)) != len(identities) or len(identities) % count:
        raise ValueError("Unique identities with balanced fold sizes required")
    order = sorted(range(len(identities)), key=lambda i: hashlib.sha256(str(identities[i]).encode()).digest())
    folds = np.empty(len(order), dtype=int)
    folds[order] = np.arange(len(order)) % count
    return folds


class FunctionalRetention:
    """A runtime scorer using reference predictions, unlabeled features and enrollment domains."""

    def __init__(self, targets, domains, assignment, penalty=1.0):
        self.targets = np.asarray(targets, float)
        self.domains = np.asarray(domains)
        self.assignment = np.asarray(assignment)
        self.penalty = penalty
        if (
            self.targets.ndim != 2
            or len(self.targets) != len(self.domains)
            or len(self.targets) != len(self.assignment)
        ):
            raise ValueError("Aligned reference targets and enrollment required")
        self.variance = np.array([self.targets[self.domains == d].var(0).mean() for d in (0, 1)])
        if min(self.variance) < 1e-12:
            raise ValueError("Reference score variance is uninformative")

    def score(self, features):
        if len(features) != len(self.targets):
            raise ValueError("Reference image identities must remain aligned")
        predictions = np.zeros_like(self.targets)
        models = {}
        for fold in sorted(set(self.assignment)):
            held = self.assignment == fold
            predictions[held], models[str(fold)] = invariant_ridge(
                np.asarray(features)[~held], self.targets[~held], np.asarray(features)[held], self.penalty
            )
        error = ((predictions - self.targets) ** 2).mean(1)
        health = np.array([error[self.domains == d].mean() / self.variance[d] for d in (0, 1)])
        return {"health": health, "predictions": predictions, "models": models}


def prospective_features(geometry, functional, parent_updates, losses):
    """At update eight, exclude all later health and loss values by construction."""
    h = np.asarray(functional, float)[:3]
    if h.shape != (3, 2):
        raise ValueError("Two-domain functional readings at zero/four/eight required")
    base = observations(np.asarray(geometry)[:3], parent_updates, list(losses)[:8])
    return np.r_[base, h[2], h[2] - h[1], h[1] - h[0]]


def accept(predicted_damage, threshold=1.0):
    return np.asarray(predicted_damage) <= threshold


def crossfit(data):
    """Prospective inputs, post-action interpretation and response-only risk composition."""
    x, real, sham, damage, groups = (
        np.asarray(data[k]) for k in ("x", "response", "sham_response", "damage", "groups")
    )
    if x.shape[1] != 23 or real.shape != (len(x), 2) or sham.shape != real.shape:
        raise ValueError("Frozen prospective feature and domain-response shapes required")
    forecasts = {name: np.zeros(len(x)) for name in ("mean", "geometry", "direct", "measured", "sham", "chain")}
    health = np.zeros_like(real)
    folds = {}
    for group in sorted(set(groups)):
        test, train = groups == group, groups != group
        forecasts["mean"][test] = damage[train].mean()
        designs = {
            "geometry": x[:, :17],
            "direct": x,
            "measured": np.c_[x[:, :17], real],
            "sham": np.c_[x[:, :17], sham],
        }
        models = {}
        for name, design in designs.items():
            models[name] = fit(design[train], damage[train])
            forecasts[name][test] = predict(models[name], design[test])
        health_model, risk_model = fit(x[train], real[train]), fit(real[train], damage[train])
        health[test] = predict(health_model, x[test])
        forecasts["chain"][test] = predict(risk_model, health[test])
        folds[str(group)] = {
            "fitting_rows": np.flatnonzero(train),
            "held_rows": np.flatnonzero(test),
            "utility_models": models,
            "health_model": health_model,
            "risk_model": risk_model,
        }
    return {"forecasts": forecasts, "predicted_health": health, "folds": folds}


def summarize(data, fitted, teacher_checks, synthetic_checks, controls):
    groups, y, response = (np.asarray(data[k]) for k in ("groups", "utility", "response"))
    damage = np.asarray(data["damage"])
    unique = sorted(set(groups))

    def errors(actual, predictions):
        return {
            str(g): float(np.mean((np.asarray(actual)[groups == g] - np.asarray(predictions)[groups == g]) ** 2))
            for g in unique
        }

    risk_errors = {name: errors(damage, p) for name, p in fitted["forecasts"].items()}
    mse = {name: float(np.mean(list(e.values()))) for name, e in risk_errors.items()}
    comparisons = {}
    for baseline in ("mean", "geometry", "sham"):
        bounds = interval([risk_errors[baseline][g] - risk_errors["measured"][g] for g in unique], family=3)
        reduction = 1 - mse["measured"] / max(mse[baseline], 1e-30)
        comparisons[baseline] = {
            "fractional_reduction": reduction,
            "interval": bounds,
            "passed": bool(reduction >= 0.1 and bounds["low"] > 0),
        }
    health_errors = {
        "regression": errors(response, fitted["predicted_health"]),
        "persistence": errors(response, np.zeros_like(response)),
        "trend": errors(response, 6 * np.asarray(data["x"])[:, 19:21]),
    }
    health_mse = {name: float(np.mean(list(v.values()))) for name, v in health_errors.items()}
    health_comparisons = {}
    for baseline in ("persistence", "trend"):
        ci = interval([health_errors[baseline][g] - health_errors["regression"][g] for g in unique], family=2)
        reduction = 1 - health_mse["regression"] / max(health_mse[baseline], 1e-30)
        health_comparisons[baseline] = {
            "fractional_reduction": reduction,
            "interval": ci,
            "passed": bool(reduction >= 0.1 and ci["low"] > 0),
        }
    accepted = {name: accept(fitted["forecasts"][name]) for name in ("chain", "direct")}
    outcomes = {name: a[:, None] * y for name, a in accepted.items()}
    outcomes.update({"always_train": y, "always_hold": np.zeros_like(y)})
    random = np.zeros_like(y)
    for g in unique:
        select = groups == g
        random[select] = accepted["chain"][select].mean() * y[select].mean(0)
    outcomes["matched_random"] = random
    policy_comparisons = {}
    for name in ("always_train", "always_hold", "direct", "matched_random"):
        policy_comparisons[name] = {
            domain: interval(
                [float(np.mean(outcomes["chain"][groups == g, j] - outcomes[name][groups == g, j])) for g in unique],
                family=12,
            )
            for j, domain in enumerate(("balanced", "day", "night"))
        }
    good = (y[:, 0] >= 0.5) & (y[:, 1:].min(1) >= -1)
    bad = damage > 1
    classification = {
        "useful_rows": int(good.sum()),
        "harmful_rows": int(bad.sum()),
        "useful_acceptance": float(accepted["chain"][good].mean()) if good.any() else 0.0,
        "harmful_rejection": float((~accepted["chain"][bad]).mean()) if bad.any() else 0.0,
    }
    parent_ids = sorted({(r["stream"], r["checkpoint"]) for r in data["identities"]})
    means = [
        y[[i for i, r in enumerate(data["identities"]) if (r["stream"], r["checkpoint"]) == key]].mean(0)
        for key in parent_ids
    ]
    population = {
        "beneficial_parents": int(sum(v[0] >= 0.5 and min(v[1:]) >= -1 for v in means)),
        "harmful_parents": int(sum(-min(v[1:]) > 1 for v in means)),
    }
    signs = np.array([v["risk_model"]["beta"] for v in fitted["folds"].values()])
    control_checks = {
        name: float(np.mean([max(r["response"]) for r in controls if r["checkpoint"] == name]))
        for name in sorted({r["checkpoint"] for r in controls})
    }
    gates = {
        "teacher": bool(teacher_checks["passed"]),
        "synthetic_instrument": bool(synthetic_checks["passed"]),
        "selected_controls": bool(
            control_checks["day_end"] <= 0.05 and all(v >= 0.05 for k, v in control_checks.items() if k != "day_end")
        ),
        "measured_interpretation": all(v["passed"] for v in comparisons.values()),
        "risk_direction": bool(np.all(np.sum(signs >= 0, axis=0) >= 5)),
        "forecast": all(v["passed"] for v in health_comparisons.values()),
        "two_sided_population": all(v >= 2 for v in population.values()),
        "shadow_gain": all(
            v["balanced"]["mean"] >= 0.25 and v["balanced"]["low"] > 0 for v in policy_comparisons.values()
        ),
        "shadow_retention": all(v[d]["low"] >= -1 for v in policy_comparisons.values() for d in ("day", "night")),
        "useful_and_harmful_choices": bool(
            classification["useful_acceptance"] >= 0.8 and classification["harmful_rejection"] >= 0.5
        ),
    }
    return plain(
        {
            "gates": gates,
            "development_ready": all(gates.values()),
            "live_filter_qualified": False,
            "risk_mse": mse,
            "risk_source_mse": risk_errors,
            "measured_comparisons": comparisons,
            "health_mse": health_mse,
            "health_source_mse": health_errors,
            "health_comparisons": health_comparisons,
            "policy_comparisons": policy_comparisons,
            "policies": {
                name: {
                    "mean_gain_vs_hold_pp": v.mean(0),
                    "accepted": int(accepted[name].sum()) if name in accepted else None,
                }
                for name, v in outcomes.items()
            },
            "classification": classification,
            "population": population,
            "risk_coefficients": signs,
            "selected_control_worst_changes": control_checks,
            "secondary_thresholds": {
                str(threshold): {
                    "accepted": int(np.sum(np.array(fitted["forecasts"]["chain"]) <= threshold)),
                    "mean_gain_vs_hold_pp": ((np.array(fitted["forecasts"]["chain"]) <= threshold)[:, None] * y).mean(
                        0
                    ),
                }
                for threshold in (0.5, 2.0)
            },
            "descriptive_measured_health_damage_correlation": correlation(np.max(response, axis=1), damage),
        }
    )
