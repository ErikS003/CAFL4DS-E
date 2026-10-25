"""Parent-only forecasts and held-history stopping diagnostics; never an online policy."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import t

from cafl4ds.collapse_risk_regression import ridge_fit


HORIZONS = np.array([8, 16, 24, 32])
REAL_MODELS = ("horizon_mean", "controls", "health")


def past_losses(trace, age=128):
    """Read only the eight completed updates strictly before the decision."""
    values = np.asarray(trace, float)[age - 8 : age, 0]
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("Eight finite past losses required")
    ticks = np.arange(8) - 3.5
    return np.array([values.mean(), ticks @ values / (ticks @ ticks)])


def design(losses, geometry):
    losses, geometry = np.asarray(losses, float), np.asarray(geometry, float)
    if losses.shape != (6, 2) or geometry.shape != (6, 15):
        raise ValueError("Six parent-only loss and geometry profiles required")
    if not np.isfinite(np.r_[losses.ravel(), geometry.ravel()]).all():
        raise ValueError("Finite parent inputs required")
    h = np.tile(HORIZONS / 32, 6)
    cheap = np.c_[h, h**2, h[:, None] * np.repeat(losses, 4, axis=0)]
    return {"controls": cheap, "health": np.c_[cheap, h[:, None] * np.repeat(geometry, 4, axis=0)]}


def mean_operator(train, query):
    """One training-history mean at each horizon, with no source pooling across time."""
    return ((query[:, None] % 4) == (train[None, :] % 4)).astype(float) / (len(train) / 4)


def select_duration(values):
    """Forecast-only point/lower-bound screen, checking every visited boundary."""
    values = np.asarray(values, float)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("Four finite two-domain forecasts required")
    allowed = [0] + [k for k in range(1, 5) if values[:k].min() >= -1 and values[k - 1].mean() >= 0.25]
    return max(allowed, key=lambda k: (np.tanh(values[k - 1].mean() / 0.5) if k else 0, -k))


def score_choices(choices, truth, lower, upper):
    truth, lower, upper = map(np.asarray, (truth, lower, upper))
    selected, states, point_ok = [], [], []
    for i, k in enumerate(choices):
        selected.append(truth[i, k - 1] if k else np.zeros(2))
        states.append(
            "within" if k == 0 or lower[i, :k].min() >= -1 else "violation" if upper[i, :k].min() < -1 else "unknown"
        )
        point_ok.append(bool(k == 0 or truth[i, :k].min() >= -1))
    gains = np.mean(selected, axis=0)
    return {
        "updates_by_history": [8 * int(k) for k in choices],
        "total_updates": 8 * int(sum(choices)),
        "mean_gain_pp": [float(gains.mean()), *gains.tolist()],
        "source_domain_gain_pp": selected,
        "path_status": states,
        "within_histories": states.count("within"),
        "violation_histories": states.count("violation"),
        "unknown_histories": states.count("unknown"),
        "point_budget_histories": int(sum(point_ok)),
    }


def with_random(choices, truth, lower, upper):
    result = score_choices(choices, truth, lower, upper)
    random = [score_choices(p, truth, lower, upper) for p in itertools.permutations(choices)]
    result["matched_random"] = {
        "assignment_permutations": len(random),
        "total_updates": result["total_updates"],
        "mean_gain_pp": np.mean([r["mean_gain_pp"] for r in random], axis=0),
        "expected_violations": float(np.mean([r["violation_histories"] for r in random])),
        "expected_unknown": float(np.mean([r["unknown_histories"] for r in random])),
    }
    result["balanced_advantage_over_random_pp"] = float(
        result["mean_gain_pp"][0] - result["matched_random"]["mean_gain_pp"][0]
    )
    return result


class PreactionStopping:
    def __init__(self, penalty=10.0, sham_count=32, sham_seed=91601):
        if penalty != 10 or sham_count < 1:
            raise ValueError("Fixed ridge penalty and positive sham count required")
        self.penalty, self.sham_count, self.sham_seed = penalty, sham_count, sham_seed

    def crossfit(self, losses, geometry, targets):
        matrices = design(losses, geometry)
        y = np.asarray(targets, float)
        if y.shape != (6, 4, 2) or not np.isfinite(y).all():
            raise ValueError("Six complete histories and four two-domain targets required")
        y = y.reshape(24, 2)
        names = ["zero", *REAL_MODELS, *[f"health_sham_{k:02d}" for k in range(self.sham_count)]]
        operators = {name: np.zeros((24, 24)) for name in names}
        predictions = {name: np.zeros((24, 2)) for name in names}
        bands = {name: np.zeros((24, 2)) for name in REAL_MODELS}
        folds = []
        for held in range(6):
            training_groups = np.array([g for g in range(6) if g != held])
            train = np.flatnonzero(np.repeat(np.arange(6), 4) != held)
            query = np.arange(4 * held, 4 * held + 4)
            fold = {"held_history": held, "train_rows": train, "held_rows": query, "models": {}, "sham_donors": {}}
            for name in names[1:]:
                if name == "horizon_mean":
                    operator = mean_operator(train, query)
                    fold["models"][name] = {"kind": "mean_at_each_horizon"}
                else:
                    if name.startswith("health_sham_"):
                        rng = np.random.default_rng(self.sham_seed + 1000 * held + int(name[-2:]))
                        while True:
                            donors = rng.permutation(training_groups)
                            if np.all(donors != training_groups):
                                break
                        sham = np.array(geometry, copy=True)
                        sham[training_groups] = np.asarray(geometry)[donors]
                        matrix = design(losses, sham)["health"]
                        fold["sham_donors"][name] = dict(zip(map(str, training_groups), map(int, donors), strict=True))
                    else:
                        matrix = matrices[name]
                    operator, _ = ridge_fit(matrix[train], np.eye(len(train)), matrix[query], self.penalty)
                    _, model = ridge_fit(matrix[train], y[train], matrix[query], self.penalty)
                    fold["models"][name] = model
                operators[name][np.ix_(query, train)] = operator
                predictions[name][query] = operator @ y[train]
                if name in REAL_MODELS:
                    residuals, inner_folds = [], []
                    for inner in training_groups:
                        inner_query = np.arange(4 * inner, 4 * inner + 4)
                        inner_train = train[train // 4 != inner]
                        if name == "horizon_mean":
                            forecast = mean_operator(inner_train, inner_query) @ y[inner_train]
                        else:
                            forecast, _ = ridge_fit(
                                matrices[name][inner_train], y[inner_train], matrices[name][inner_query], self.penalty
                            )
                        residuals.append(abs(forecast - y[inner_query]))
                        inner_folds.append({"held_history": int(inner), "train_rows": inner_train})
                    bands[name][query] = np.max(residuals, axis=0)
                    fold["models"][name]["inner_folds"] = inner_folds
            folds.append(fold)
        return {"predictions": predictions, "operators": operators, "envelope_half_width_pp": bands, "folds": folds}

    def summarize(self, targets, fitted, lower, upper, oracle):
        truth = np.asarray(targets, float)
        y = truth.reshape(24, 2)
        metrics, errors = {}, {}
        for name, forecast in fitted["predictions"].items():
            residual = np.asarray(forecast) - y
            error = (residual**2).reshape(6, 4, 2)
            errors[name] = error.mean(1)
            metrics[name] = {
                "rmse_pp": np.sqrt(error.mean((0, 1))),
                "bias_pp": residual.mean(0),
                "source_mse": errors[name],
                "horizon_rmse_pp": np.sqrt(error.mean(0)),
            }
        shams = [n for n in metrics if n.startswith("health_sham_")]
        winners = [min(shams, key=lambda n: errors[n][:, d].mean()) for d in range(2)]
        sham_errors = np.column_stack([errors[n][:, d] for d, n in enumerate(winners)])
        comparisons = {}
        for name in ("horizon_mean", "controls", "best_geometry_sham"):
            base = sham_errors if name == "best_geometry_sham" else errors[name]
            delta = base - errors["health"]
            mean = delta.mean(0)
            half = t.ppf(1 - 0.05 / (2 * 6), 5) * delta.std(0, ddof=1) / np.sqrt(6)
            reduction = 1 - errors["health"].mean(0) / np.maximum(base.mean(0), 1e-30)
            passed = (reduction >= 0.1) & ((delta > 0).sum(0) >= 5) & (mean - half > 0)
            comparisons[name] = {
                "fractional_mse_reduction": reduction,
                "histories_improved": (delta > 0).sum(0),
                "adjusted_source_range": [mean - half, mean + half],
                "passed_domains": passed,
            }
        rules = {
            "hold": with_random([0] * 6, truth, lower, upper),
            "always_train": with_random([4] * 6, truth, lower, upper),
        }
        rules["acquired_oracle"] = with_random(oracle, truth, lower, upper)
        for name in REAL_MODELS:
            p = np.asarray(fitted["predictions"][name]).reshape(6, 4, 2)
            width = np.asarray(fitted["envelope_half_width_pp"][name]).reshape(6, 4, 2)
            for suffix, forecast in (("point", p), ("envelope", p - width)):
                rules[f"{name}_{suffix}"] = with_random([select_duration(v) for v in forecast], truth, lower, upper)
            metrics[name]["residual_envelope_coverage_fraction"] = (abs(p - truth) <= width).mean((0, 1))
            metrics[name]["mean_envelope_half_width_pp"] = width.mean((0, 1))
        calibration = (metrics["health"]["rmse_pp"] <= 1) & (abs(metrics["health"]["bias_pp"]) <= 0.5)
        return {
            "metrics": metrics,
            "comparisons": comparisons,
            "best_sham_per_domain": winners,
            "forecast_calibration_domains": calibration,
            "exploratory_prediction_gate": bool(
                calibration.all() and np.all([v["passed_domains"] for v in comparisons.values()])
            ),
            "rules": rules,
            "held_history_folds": 6,
            "forecast_rows": 24,
            "fresh_confirmation": False,
            "policy_qualified": False,
            "controller_activated": False,
            "new_ssl_updates": 0,
            "new_canonical_classifier_fits": 0,
        }
