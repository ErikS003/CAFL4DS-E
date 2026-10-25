"""Fixed old-data regressions and paired night-target confirmation statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from cafl4ds.collapse_risk_regression import ridge_fit
from cafl4ds.functional_retention import plain

MODELS = ("mean", "controls", "dino", "sham")


def designs(controls, health, sham):
    x = np.asarray(controls, float)
    result = {"controls": x}
    for name, h in (("dino", health), ("sham", sham)):
        h = np.asarray(h, float)
        if h.shape != (len(x), 2, 2):
            raise ValueError("Paired day/night starting and endpoint health required")
        result[name] = np.c_[x, h[:, 0], h[:, 1] - h[:, 0]]
    return result


def prediction(model, x):
    return np.asarray(model["intercept"]) + ((np.asarray(x) - model["center"]) / model["scale"]) @ np.asarray(
        model["beta"]
    )


def resamples(labels, panels=256, repetitions=2000, seed=74551):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    panel = rng.multinomial(panels, np.full(panels, 1 / panels), size=repetitions).astype(np.uint16)
    query = np.zeros((repetitions, len(labels)), np.uint16)
    for c in (0, 1):
        selected = np.flatnonzero(labels == c)
        if not len(selected):
            raise ValueError("Both night classes required")
        query[:, selected] = rng.multinomial(len(selected), np.full(len(selected), 1 / len(selected)), size=repetitions)
    return panel, query


def compare(before, after, labels, panel_counts, query_counts):
    """Matched classifiers and class-balanced conditional night accuracy gains."""
    labels = np.asarray(labels)
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.shape[1:] != (len(labels), 2):
        raise ValueError("Aligned panels by night queries by two probabilities required")
    delta = 100 * ((after.argmax(-1) == labels).astype(float) - (before.argmax(-1) == labels))
    panel_gain = np.mean([delta[:, labels == c].mean(1) for c in (0, 1)], axis=0)
    if not np.all(panel_counts.sum(1) == len(before)):
        raise ValueError("Invalid panel counts")
    weighted = panel_counts.astype(float) @ delta / len(before)
    query_sample, crossed = np.zeros(len(panel_counts)), np.zeros(len(panel_counts))
    for c in (0, 1):
        selected = labels == c
        if not np.all(query_counts[:, selected].sum(1) == selected.sum()):
            raise ValueError("Invalid class-stratified query counts")
        weight = query_counts[:, selected] / (2 * selected.sum())
        query_sample += weight @ delta[:, selected].mean(0)
        crossed += np.sum(weight * weighted[:, selected], axis=1)

    def interval(values):
        bounds = np.quantile(values, [0.025, 0.975])
        return {"lower": bounds[0], "upper": bounds[1], "half_width": (bounds[1] - bounds[0]) / 2}

    result = {
        "gain_pp": panel_gain.mean(),
        "panel_gain_pp": panel_gain,
        "positive_panel_fraction": np.mean(panel_gain > 1e-8),
        "negative_panel_fraction": np.mean(panel_gain < -1e-8),
        "support": interval(panel_counts @ panel_gain / len(before)),
        "query": interval(query_sample),
        "crossed": interval(crossed),
    }
    return plain(result), -crossed


class NightHealthConfirmation:
    def __init__(self, penalty=10.0, minimum_reduction=0.1, required_source_wins=5):
        self.penalty = penalty
        self.minimum_reduction = minimum_reduction
        self.required_source_wins = required_source_wins

    def fit_development(self, data):
        """The only fitting API takes the frozen old development arrays."""
        y = np.asarray(data["damage"], float)[:, 1]
        matrices = designs(data["controls"], data["dino"], data["dino_sham"])
        models = {"mean": {"value": float(y.mean()), "rows": len(y)}}
        for name, x in matrices.items():
            _, models[name] = ridge_fit(x, y, x, self.penalty)
        return plain(models)

    def predict(self, models, controls, health, sham):
        matrices = designs(controls, health, sham)
        result = {name: prediction(models[name], x) for name, x in matrices.items()}
        result["mean"] = np.full(len(controls), models["mean"]["value"])
        return result

    def summarize(self, records, forecasts, crossed):
        groups = np.array([r["source"] for r in records])
        unique = sorted(set(groups))
        if len(unique) != 6 or len(records) != 18:
            raise ValueError("Exactly six confirmation histories and eighteen contrasts required")
        y = -np.array([r["target"]["gain_pp"] for r in records])
        errors, metrics, bootstrap = {}, {}, {}
        resolved = np.array([r["target"]["crossed"]["upper"] < -1 for r in records])
        for name in MODELS:
            predicted = np.asarray(forecasts[name], float)
            residual = predicted - y
            errors[name] = np.array([(residual[groups == g] ** 2).mean() for g in unique])
            alarm = predicted > 1
            harm = y > 1
            metrics[name] = {
                "mse": errors[name].mean(),
                "rmse_pp": np.sqrt(errors[name].mean()),
                "mae_pp": np.abs(residual).mean(),
                "bias_pp": residual.mean(),
                "source_mse": errors[name],
                "harm_counts": {
                    "tp": int((alarm & harm).sum()),
                    "fp": int((alarm & ~harm).sum()),
                    "fn": int((~alarm & harm).sum()),
                    "tn": int((~alarm & ~harm).sum()),
                },
                "resolved_harm_detected": int((alarm & resolved).sum()),
                "resolved_harm_missed": int((~alarm & resolved).sum()),
                "other_alarms": int((alarm & ~resolved).sum()),
            }
            residual_samples = predicted[None] - crossed
            bootstrap[name] = (residual_samples**2).mean(1)
            metrics[name]["conditional_rmse_range_pp"] = np.quantile(np.sqrt(bootstrap[name]), [0.025, 0.975])
        comparisons = {}
        for baseline in ("mean", "controls", "sham"):
            delta = errors[baseline] - errors["dino"]
            half = t.ppf(1 - 0.05 / 6, 5) * delta.std(ddof=1) / np.sqrt(6)
            reduction = 1 - metrics["dino"]["mse"] / max(metrics[baseline]["mse"], 1e-30)
            wins = int((delta > 0).sum())
            comparisons[baseline] = {
                "fractional_mse_reduction": reduction,
                "source_mse_improvements": delta,
                "lower": delta.mean() - half,
                "upper": delta.mean() + half,
                "sources_improved": wins,
                "conditional_reduction_range": np.quantile(
                    1 - bootstrap["dino"] / np.maximum(bootstrap[baseline], 1e-30), [0.025, 0.975]
                ),
                "passed": bool(
                    reduction >= self.minimum_reduction
                    and wins >= self.required_source_wins
                    and delta.mean() - half > 0
                ),
            }
        support_fraction = np.mean([r["target"]["support"]["half_width"] <= 0.5 for r in records])
        crossed_fraction = np.mean([r["target"]["crossed"]["half_width"] <= 1 for r in records])
        useful = sorted(
            {r["source"] for r in records if r["target"]["gain_pp"] >= 0.25 and r["target"]["crossed"]["lower"] > 0}
        )
        harmful = sorted({r["source"] for r in records if r["target"]["crossed"]["upper"] < -1})
        gates = {
            "support_precision": bool(support_fraction >= 0.8),
            "crossed_precision": bool(crossed_fraction >= 0.8),
            "two_sided_opportunity": len(useful) >= 2 and len(harmful) >= 2,
            "incremental_transfer": all(c["passed"] for c in comparisons.values()),
        }
        passed = all(gates.values())
        return plain(
            {
                "metrics": metrics,
                "comparisons": comparisons,
                "gates": gates,
                "association_confirmed": passed,
                "calibration_passed": bool(metrics["dino"]["rmse_pp"] <= 1 and abs(metrics["dino"]["bias_pp"]) <= 0.5),
                "support_precision_fraction": support_fraction,
                "crossed_precision_fraction": crossed_fraction,
                "useful_sources": useful,
                "harmful_sources": harmful,
                "filter_qualified": False,
                "decision": "prepare_separate_observability_and_calibration_protocol"
                if passed
                else "stop_night_transfer_claim",
                "scope": "new_clip_identities_reused_query_pool_measured_endpoint_health",
            }
        )
