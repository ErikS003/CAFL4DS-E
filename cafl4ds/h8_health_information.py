"""Eight-update measured-health information, with complete-history exclusion."""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from cafl4ds.collapse_risk_regression import ridge_fit
from cafl4ds.prediction_benchmark import health_values

SIGNALS = ("rank", "variance", "covariance", "cka", "cosine")
CONTEXTS = (("day_end", 128), ("night_end", 256), ("return_early", 288))
PRIMARY = "geometry"


def geometry_features(features, warm, domains):
    """Global/day/night geometry; positive-valued metrics use warm-relative logs."""
    features, warm, domains = np.asarray(features), np.asarray(warm), np.asarray(domains)
    values = []
    for selected in (np.ones(len(domains), bool), domains == 0, domains == 1):
        state = health_values(features[selected], warm[selected])
        anchor = health_values(warm[selected], warm[selected])
        values.extend(
            np.log(max(state[k], 1e-30) / max(anchor[k], 1e-30)) if k in SIGNALS[:3] else state[k] for k in SIGNALS
        )
    result = np.asarray(values)
    if result.shape != (15,) or not np.isfinite(result).all():
        raise ValueError("Finite global/day/night geometry required")
    return result


def training_shuffle(groups, contexts, held_group, seed):
    """Permute whole training-history response profiles, never borrowing held features."""
    groups, contexts = np.asarray(groups), np.asarray(contexts)
    sources = sorted(set(groups) - {held_group})
    if len(sources) < 3:
        raise ValueError("At least three training histories required")
    rng = np.random.default_rng(seed)
    while True:
        donors = rng.permutation(sources)
        if np.all(donors != sources):
            break
    donor_map = dict(zip(sources, donors, strict=True))
    indices = np.arange(len(groups))
    for i in np.flatnonzero(groups != held_group):
        match = np.flatnonzero((groups == donor_map[groups[i]]) & (contexts == contexts[i]))
        if len(match) != 1:
            raise ValueError("One row per history/context required")
        indices[i] = match[0]
    return indices


def source_errors(residual, groups):
    """Equal history weights; repeated contexts do not become new source groups."""
    return np.stack([(np.asarray(residual)[groups == g] ** 2).mean(0) for g in sorted(set(groups))])


def point_admission(predictions, minimum_gain=0.25, budget=1.0):
    """An offline point screen only; it has no forecast safety guarantee."""
    p = np.asarray(predictions)
    return (p.mean(-1) >= minimum_gain) & (p.min(-1) >= -budget)


class H8HealthInformation:
    def __init__(self, penalty=10.0, sham_count=32, sham_seed=81201, minimum_reduction=0.1):
        if penalty <= 0 or sham_count < 1 or not 0 < minimum_reduction < 1:
            raise ValueError("Positive ridge penalty, shams and fractional reduction required")
        self.penalty = penalty
        self.sham_count = sham_count
        self.sham_seed = sham_seed
        self.minimum_reduction = minimum_reduction

    def crossfit(self, controls, geometry, functional, y, groups, contexts):
        """All health is measured after the block; none is a prospective policy input."""
        x, h, y, groups, contexts = map(np.asarray, (controls, geometry, y, groups, contexts))
        if h.shape != (len(x), 2, 15) or y.shape != (len(x), 2):
            raise ValueError("Aligned parent/endpoint geometry and two-domain outcomes required")
        if len(set(groups)) < 4 or not np.isfinite(np.r_[x.ravel(), h.ravel(), y.ravel()]).all():
            raise ValueError("Finite data and at least four histories required")
        if len({(g, c) for g, c in zip(groups, contexts, strict=True)}) != len(groups):
            raise ValueError("Duplicate history/context row")
        parent, response = h[:, 0], h[:, 1] - h[:, 0]
        designs = {
            "controls": x,
            "geometry_parent": np.c_[x, parent],
            "geometry_response": np.c_[x, response],
            "geometry": np.c_[x, parent, response],
        }
        for name, values in functional.items():
            values = np.asarray(values)
            if values.shape != (len(x), 2, 2):
                raise ValueError("Paired functional negative controls required")
            designs[name] = np.c_[x, values[:, 0], values[:, 1] - values[:, 0]]
        names = ["mean", *designs, *[f"geometry_sham_{i:02d}" for i in range(self.sham_count)]]
        operators = {n: np.zeros((len(x), len(x))) for n in names}
        folds = {}
        for fold_index, group in enumerate(sorted(set(groups))):
            train, held = groups != group, groups == group
            ti, hi = np.flatnonzero(train), np.flatnonzero(held)
            operators["mean"][np.ix_(hi, ti)] = 1 / train.sum()
            models, shuffles = {}, {}
            for name in names[1:]:
                if name.startswith("geometry_sham_"):
                    seed = self.sham_seed + 1000 * fold_index + int(name[-2:])
                    permutation = training_shuffle(groups, contexts, group, seed)
                    design = np.c_[x, parent, response[permutation]]
                    shuffles[name] = permutation
                else:
                    design = designs[name]
                operator, model = ridge_fit(design[train], np.eye(train.sum()), design[held], self.penalty)
                operators[name][np.ix_(hi, ti)] = operator
                # Persist actual fitted coefficients as well as the linear outcome operator.
                _, actual = ridge_fit(design[train], y[train], design[held], self.penalty)
                models[name] = {"actual": actual, "operator_model": model}
            folds[str(group)] = {"train": ti, "held": hi, "models": models, "shuffles": shuffles}
        predictions = {n: op @ y for n, op in operators.items()}
        return {"operators": operators, "predictions": predictions, "folds": folds}

    def summarize(self, y, groups, fitted, crossed, rows, target_ready):
        y, groups, crossed = np.asarray(y), np.asarray(groups), np.asarray(crossed)
        unique = sorted(set(groups))
        metrics, errors, sensitivity_errors = {}, {}, {}
        useful = np.array([r["useful"] for r in rows])
        harmful = np.array([r["harmful"] for r in rows])
        for name, p in fitted["predictions"].items():
            residual = np.asarray(p) - y
            errors[name] = source_errors(residual, groups)
            mask = point_admission(p)
            sensitivity_residual = np.einsum("ij,rjd->rid", fitted["operators"][name], crossed) - crossed
            sensitivity_errors[name] = np.mean(sensitivity_residual**2, axis=1)
            metrics[name] = {
                "mse": errors[name].mean(0),
                "rmse_pp": np.sqrt(errors[name].mean(0)),
                "bias_pp": residual.mean(0),
                "mae_pp": abs(residual).mean(0),
                "source_mse": errors[name],
                "bias_measurement_range_pp": np.quantile(sensitivity_residual.mean(1), [0.025, 0.975], axis=0),
                "point_screen": {
                    "accepted_rows": np.flatnonzero(mask),
                    "useful_accepted": int(np.sum(mask & useful)),
                    "useful_total": int(useful.sum()),
                    "resolved_harm_rejected": int(np.sum(~mask & harmful)),
                    "resolved_harm_total": int(harmful.sum()),
                    "scope": "offline_parent_budget_point_screen_no_uncertainty_or_warm_screen",
                },
            }
        shams = [n for n in metrics if n.startswith("geometry_sham_")]
        # Best sham is chosen independently per domain: deliberately stringent specificity benchmark.
        winners = [min(shams, key=lambda n: metrics[n]["mse"][d]) for d in (0, 1)]
        sham_source = np.column_stack([errors[n][:, d] for d, n in enumerate(winners)])
        sham_sensitivity = np.minimum.reduce([sensitivity_errors[n] for n in shams])
        comparisons = {}
        for baseline in ("controls", "geometry_parent", "best_response_sham"):
            e = sham_source if baseline == "best_response_sham" else errors[baseline]
            b = sham_sensitivity if baseline == "best_response_sham" else sensitivity_errors[baseline]
            delta = e - errors[PRIMARY]
            mean = delta.mean(0)
            # An exploratory source-level diagnostic, not an independent-source safety certificate.
            half = t.ppf(1 - 0.05 / (2 * 6), len(unique) - 1) * delta.std(0, ddof=1) / np.sqrt(len(unique))
            reduction = 1 - errors[PRIMARY].mean(0) / np.maximum(e.mean(0), 1e-30)
            samples = 1 - sensitivity_errors[PRIMARY] / np.maximum(b, 1e-30)
            bounds = np.quantile(samples, [0.025, 0.975], axis=0)
            wins = (delta > 0).sum(0)
            passed = (reduction >= self.minimum_reduction) & (mean - half > 0) & (wins >= 5) & (bounds[0] > 0)
            comparisons[baseline] = {
                "fractional_mse_reduction": reduction,
                "source_difference": delta,
                "exploratory_adjusted_source_range": [mean - half, mean + half],
                "measurement_reduction_range": bounds,
                "sources_improved": wins,
                "passed_domains": passed,
            }
        calibration = (metrics[PRIMARY]["rmse_pp"] <= 1.0) & (abs(metrics[PRIMARY]["bias_pp"]) <= 0.5)
        predictive = np.all([r["passed_domains"] for r in comparisons.values()], axis=0)
        bridge_pass = bool(target_ready and calibration.all() and predictive.all())
        return {
            "metrics": metrics,
            "comparisons": comparisons,
            "best_response_sham_names": winners,
            "calibration_domains": calibration,
            "incremental_domains": predictive,
            "target_ready": bool(target_ready),
            "bridge_pass": bridge_pass,
            "functional_negative_controls_eligible": False,
            "filter_qualified": False,
            "controller_activated": False,
            "source_groups": len(unique),
            "rows": len(y),
            "decision": "prepare_separate_preaction_forecasting_protocol"
            if bridge_pass
            else "stop_before_preaction_forecasting_or_admission",
            "scope": "reused_development_measured_postaction_health_not_fresh_confirmation",
        }
