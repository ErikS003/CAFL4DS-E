"""Measured functional health as an incremental predictor of conditional capability."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from cafl4ds.collapse_risk_regression import ridge_fit
from cafl4ds.functional_retention import plain

CANDIDATES = ("dino", "mae")
MODELS = ("mean", "controls", "dino", "mae", "dino_sham", "mae_sham")
CONTROL_NAMES = ("day_end", "night_end", "age_128", "past_day_fraction", "past_loss_mean8", "past_loss_slope8")


def controls(context, age, trace):
    """Loss inputs strictly precede the paired update block."""
    if context not in ("day_end", "night_end", "return_early") or age < 8:
        raise ValueError("Known parent context and eight past losses required")
    loss = np.asarray(trace, float)[age - 8 : age, 0]
    if len(loss) != 8 or not np.isfinite(loss).all():
        raise ValueError("Missing finite causal loss prefix")
    tick = np.arange(8) - 3.5
    day_updates = min(age, 128) + max(age - 256, 0)
    return np.r_[
        context == "day_end",
        context == "night_end",
        age / 128,
        day_updates / age,
        loss.mean(),
        tick @ loss / (tick @ tick),
    ]


def design_matrices(x, health):
    """Health is row by parent/endpoint by day/night, with shams separate."""
    designs = {"controls": np.asarray(x, float)}
    for name in MODELS[2:]:
        h = np.asarray(health[name], float)
        if h.shape != (len(x), 2, 2) or not np.isfinite(h).all():
            raise ValueError("Aligned finite paired health required")
        designs[name] = np.c_[x, h[:, 0], h[:, 1] - h[:, 0]]
    return designs


def crossed_damage(before, after, labels, domains, panel_counts, query_counts):
    """Preserve the paired resampling law of the qualified target."""
    delta = 100 * ((np.asarray(before).argmax(-1) == labels).astype(float) - (np.asarray(after).argmax(-1) == labels))
    weighted = np.asarray(panel_counts, float) @ delta / len(before)
    result = []
    for domain in (0, 1):
        score = np.zeros(len(panel_counts))
        for label in (0, 1):
            selected = (domains == domain) & (labels == label)
            if not np.all(query_counts[:, selected].sum(1) == selected.sum()):
                raise ValueError("Invalid stratified query counts")
            score += np.einsum("rq,rq->r", weighted[:, selected], query_counts[:, selected]) / (2 * selected.sum())
        result.append(score)
    return np.stack(result, axis=-1)


class HealthCapabilityBridge:
    def __init__(self, penalty=10.0, minimum_reduction=0.1, maximum_rmse_pp=1.0, maximum_bias_pp=0.5):
        self.penalty = penalty
        self.minimum_reduction = minimum_reduction
        self.maximum_rmse_pp = maximum_rmse_pp
        self.maximum_bias_pp = maximum_bias_pp

    def crossfit(self, x, health, damage, groups):
        damage, groups = np.asarray(damage, float), np.asarray(groups)
        if damage.shape != (len(x), 2) or len(set(groups)) < 4:
            raise ValueError("Two-domain outcomes and at least four source histories required")
        designs = design_matrices(x, health)
        predictions = {name: np.zeros_like(damage) for name in MODELS}
        operators = {name: np.zeros((len(x), len(x))) for name in MODELS}
        folds = {}
        for group in sorted(set(groups)):
            train, held = groups != group, groups == group
            train_i, held_i = np.flatnonzero(train), np.flatnonzero(held)
            operators["mean"][np.ix_(held_i, train_i)] = 1 / train.sum()
            predictions["mean"][held] = damage[train].mean(0)
            models = {}
            for name, design in designs.items():
                predictions[name][held], models[name] = ridge_fit(
                    design[train], damage[train], design[held], self.penalty
                )
                operator, _ = ridge_fit(design[train], np.eye(train.sum()), design[held], self.penalty)
                operators[name][np.ix_(held_i, train_i)] = operator
            folds[str(group)] = {"train": train_i, "held": held_i, "models": models}
        for name in MODELS:
            np.testing.assert_allclose(operators[name] @ damage, predictions[name], atol=1e-10)
        return {"predictions": predictions, "operators": operators, "folds": folds}

    def summarize(self, damage, groups, fitted, crossed, observability, teacher_checks, instrument_checks):
        y, groups = np.asarray(damage), np.asarray(groups)
        unique = sorted(set(groups))
        predictions = fitted["predictions"]
        errors, metrics = {}, {}
        for name in MODELS:
            residual = np.asarray(predictions[name]) - y
            errors[name] = np.array([(residual[groups == g] ** 2).mean(0) for g in unique])
            metrics[name] = {
                "mse": errors[name].mean(0),
                "rmse_pp": np.sqrt(errors[name].mean(0)),
                "mae_pp": np.abs(residual).mean(0),
                "bias_pp": residual.mean(0),
                "source_mse": errors[name],
                "harm_counts": [
                    {
                        "tp": int(((predictions[name][:, d] > 1) & (y[:, d] > 1)).sum()),
                        "fp": int(((predictions[name][:, d] > 1) & (y[:, d] <= 1)).sum()),
                        "fn": int(((predictions[name][:, d] <= 1) & (y[:, d] > 1)).sum()),
                        "tn": int(((predictions[name][:, d] <= 1) & (y[:, d] <= 1)).sum()),
                    }
                    for d in (0, 1)
                ],
            }
        comparisons, gates = {}, {}
        bootstrap_errors, sensitivity = {}, {}
        for name in MODELS:
            predicted = np.einsum("ij,rjd->rid", fitted["operators"][name], crossed)
            bootstrap_errors[name] = ((predicted - crossed) ** 2).mean(1)
        for candidate in CANDIDATES:
            comparisons[candidate] = {}
            for baseline in ("mean", "controls", candidate + "_sham"):
                differences = errors[baseline] - errors[candidate]
                average = differences.mean(0)
                half = t.ppf(1 - 0.05 / (2 * 12), len(unique) - 1) * differences.std(0, ddof=1) / np.sqrt(len(unique))
                reduction = 1 - metrics[candidate]["mse"] / np.maximum(metrics[baseline]["mse"], 1e-30)
                wins = (differences > 0).sum(0)
                comparisons[candidate][baseline] = {
                    "fractional_mse_reduction": reduction,
                    "source_mse_difference": differences,
                    "lower": average - half,
                    "upper": average + half,
                    "sources_improved": wins,
                    "passed": (reduction >= self.minimum_reduction) & (average - half > 0) & (wins >= 5),
                }
                reduction_samples = 1 - bootstrap_errors[candidate] / np.maximum(bootstrap_errors[baseline], 1e-30)
                rmse_difference = np.sqrt(bootstrap_errors[baseline]) - np.sqrt(bootstrap_errors[candidate])
                sensitivity[candidate + "_vs_" + baseline] = {
                    "fractional_mse_reduction": np.quantile(reduction_samples, [0.025, 0.5, 0.975], axis=0),
                    "rmse_improvement_pp": np.quantile(rmse_difference, [0.025, 0.5, 0.975], axis=0),
                }
            calibration = (metrics[candidate]["rmse_pp"] <= self.maximum_rmse_pp) & (
                np.abs(metrics[candidate]["bias_pp"]) <= self.maximum_bias_pp
            )
            predictive = np.all([v["passed"] for v in comparisons[candidate].values()], axis=0)
            observable = np.asarray(observability[candidate]["passed"])
            semantic = bool(teacher_checks[candidate]["passed"])
            instrument = bool(instrument_checks[candidate]["passed"])
            qualified = predictive & calibration & observable & semantic & instrument
            gates[candidate] = {
                "incremental_prediction": predictive,
                "calibration": calibration,
                "observable": observable,
                "semantic": semantic,
                "instrument": instrument,
                "qualified_domains": qualified,
                "qualified": bool(qualified.all()),
            }
        for name in MODELS:
            residual_samples = np.einsum("ij,rjd->rid", fitted["operators"][name], crossed) - crossed
            sensitivity[name + "_bias_pp"] = np.quantile(residual_samples.mean(1), [0.025, 0.5, 0.975], axis=0)
        return plain(
            {
                "metrics": metrics,
                "comparisons": comparisons,
                "gates": gates,
                "measurement_sensitivity": sensitivity,
                "qualified": any(g["qualified"] for g in gates.values()),
                "source_count": len(unique),
                "decision": "prepare_prospective_regression_protocol"
                if any(g["qualified"] for g in gates.values())
                else "stop_before_prospective_forecasting_or_threshold_tuning",
            }
        )
