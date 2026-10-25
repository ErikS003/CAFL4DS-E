"""Held-history forecasts and warning costs on cached full-training trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import t

from cafl4ds.dnd_health_stopping import reading

INSTRUMENTS = ("rank", "variance", "cka", "combined")
MODELS = ("constant", "age", "rank", "variance", "cka", "all")
TARGETS = ("worst_loss", "balanced_gain", "day_gain", "night_gain")


def features_and_alarms(banks: Mapping[int, np.ndarray], monitor: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed reference and only the present and preceding scheduled bank."""
    levels = np.array([reading(banks[132], banks[p]) for p in monitor])
    changes = np.vstack([np.zeros(3), np.diff(levels, axis=0)])
    features = np.stack([levels, changes], axis=-1).reshape(len(monitor), 6)
    exceeds = levels >= 0.1 - 1e-12
    exceeds[0] = False
    persistent = exceeds & np.vstack([np.zeros(3, dtype=bool), exceeds[:-1]])
    combined = exceeds.any(1) & np.r_[False, exceeds[:-1].any(1)]
    return features, np.column_stack([persistent, combined])


def horizon_steps(monitor: tuple[int, ...], points: tuple[int, ...], horizon: int) -> np.ndarray:
    """Keep a full exact horizon, without interpolating or clipping late dates."""
    return np.array([p for p in monitor if p >= 140 and p + horizon <= monitor[-1] and p + horizon in points])


def outcome_targets(curve: np.ndarray, points: tuple[int, ...], steps: np.ndarray, horizon: int) -> np.ndarray:
    """Measure current-capability loss and signed final learning from continuing."""
    curve = np.asarray(curve, float)
    result = []
    for step in steps:
        start, end = points.index(int(step)), points.index(int(step + horizon))
        loss = max(0.0, float((curve[start, 1:] - curve[start + 1 : end + 1, 1:]).max()))
        result.append(np.r_[loss, curve[end] - curve[start]])
    return np.asarray(result)


def interval(values: np.ndarray) -> dict[str, Any]:
    """Give descriptive paired-history t intervals, never row-level inference."""
    values = np.asarray(values, float)
    half = t.ppf(0.975, len(values) - 1) * values.std(0, ddof=1) / np.sqrt(len(values))
    mean = values.mean(0)
    return {"mean": mean, "ci95": np.stack([mean - half, mean + half], axis=-1), "history_values": values}


def crossfit(features: np.ndarray, targets: np.ndarray, penalty: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Learn clock fixed effects and ridge corrections using other histories only."""
    features, targets = np.asarray(features, float), np.asarray(targets, float)
    if features.shape[:2] != targets.shape[:2] or targets.shape[-1] != 4 or penalty <= 0:
        raise ValueError("Matched history/date arrays, four targets and positive penalty required")
    predictions = np.empty((len(MODELS), *targets.shape))
    folds = []
    for heldout in range(len(targets)):
        train = np.array([i for i in range(len(targets)) if i != heldout])
        tx, ty = features[train], targets[train]
        xmean, ymean = tx.mean(0), ty.mean(0)
        predictions[0, heldout] = ty.mean((0, 1))
        predictions[1, heldout] = ymean
        for model_index, columns in enumerate(((0, 1), (2, 3), (4, 5), tuple(range(6))), 2):
            center = (tx - xmean)[..., columns].reshape(-1, len(columns))
            scale = center.std(0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            z = center / scale
            residual = (ty - ymean).reshape(-1, 4)
            coef = np.linalg.solve(z.T @ z / len(z) + penalty * np.eye(len(columns)), z.T @ residual / len(z))
            held_z = (features[heldout] - xmean)[..., columns] / scale
            predictions[model_index, heldout] = ymean + held_z @ coef
            folds.append(
                {
                    "heldout": heldout,
                    "train": train,
                    "model": MODELS[model_index],
                    "columns": columns,
                    "feature_mean_by_clock": xmean[..., columns],
                    "outcome_mean_by_clock": ymean,
                    "scale": scale,
                    "coef": coef,
                }
            )
    predictions[..., 0] = np.maximum(predictions[..., 0], 0)
    return predictions, folds


def matched_age_selection(age_risk: np.ndarray, warnings: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Select exactly the same number of dates per history without using outcomes."""
    selected = np.zeros_like(warnings, dtype=bool)
    for i, row in enumerate(warnings):
        order = np.lexsort((steps, -age_risk[i]))
        selected[i, order[: int(row.sum())]] = True
    return selected


def warning_statistics(
    targets: np.ndarray, weights: np.ndarray, material: float, balanced_cost: float, domain_cost: float
) -> dict[str, Any]:
    """Describe isolated warning opportunities, including exact random expectations."""
    weights = np.asarray(weights, float)
    total = float(weights.sum())
    harm = targets[..., 0] >= material
    learning = (targets[..., 1] > balanced_cost) | (targets[..., 2:] > domain_cost).any(-1)
    useful = harm & ~learning
    return {
        "warning_count": total,
        "warning_fraction": total / weights.size,
        "harm_precision": float((weights * harm).sum() / total) if total else None,
        "harm_recall": float((weights * harm).sum() / harm.sum()) if harm.any() else None,
        "useful_hold_precision": float((weights * useful).sum() / total) if total else None,
        "material_learning_fraction": float((weights * learning).sum() / total) if total else None,
        "mean_targets_when_warned_pp": (targets * weights[..., None]).sum((0, 1)) / total if total else None,
        "mean_positive_balanced_gain_when_warned_pp": float((np.maximum(targets[..., 1], 0) * weights).sum() / total)
        if total
        else None,
        "protected_loss_per_decision_pp": (targets[..., 0] * weights).mean(1),
        "forgone_positive_learning_per_decision_pp": (np.maximum(targets[..., 1], 0) * weights).mean(1),
    }


class SignalDiagnostic:
    """A fixed, exploratory analysis; it does not implement a live controller."""

    def __init__(
        self,
        penalty: float = 1.0,
        material_loss_pp: float = 1.0,
        balanced_cost_pp: float = 0.5,
        domain_cost_pp: float = 1.0,
    ) -> None:
        """Keep fixed regularization and descriptive harm/cost thresholds."""
        self.penalty = penalty
        self.material = material_loss_pp
        self.balanced_cost = balanced_cost_pp
        self.domain_cost = domain_cost_pp

    def analyze(
        self, windows: dict[int, dict[str, np.ndarray]], primary_horizon: int
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
        """Produce held-history errors and warning comparisons for all declared models."""
        summary: dict[str, Any] = {
            "primary_horizon": primary_horizon,
            "horizons": {},
            "exploratory_reuse": True,
            "controller_qualified": False,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
        }
        arrays, folds = {}, []
        for horizon, data in windows.items():
            predictions, fitted = crossfit(data["features"], data["targets"], self.penalty)
            arrays[f"predictions_{horizon}"] = predictions
            folds.extend({"horizon": horizon, **fold} for fold in fitted)
            truth = data["targets"]
            error = predictions - truth[None]
            rmse = np.sqrt((error**2).mean(2))
            mae = np.abs(error).mean(2)
            models = {}
            for j, name in enumerate(MODELS):
                improvement = rmse[1, :, 0] - rmse[j, :, 0]
                relative = float(improvement.mean() / max(rmse[1, :, 0].mean(), 1e-12))
                ci = interval(improvement)
                models[name] = {
                    "rmse_pp": interval(rmse[j]),
                    "mae_pp": interval(mae[j]),
                    "risk_rmse_improvement_vs_age": ci,
                    "fractional_risk_rmse_improvement": relative,
                    "histories_improved": int((improvement > 0).sum()),
                    "exploratory_screen": bool(
                        j >= 2 and relative >= 0.1 and ci["ci95"][0] > 0 and (improvement > 0).sum() >= 6
                    ),
                    "risk_warning": warning_statistics(
                        truth,
                        predictions[j, ..., 0] >= self.material,
                        self.material,
                        self.balanced_cost,
                        self.domain_cost,
                    ),
                }
            alarms = {}
            for j, name in enumerate(INSTRUMENTS):
                mask = data["alarms"][..., j]
                age = matched_age_selection(predictions[1, ..., 0], mask, data["steps"])
                random = np.broadcast_to(mask.mean(1)[:, None], mask.shape)
                actual, age_stats, random_stats = [
                    warning_statistics(truth, weights, self.material, self.balanced_cost, self.domain_cost)
                    for weights in (mask, age, random)
                ]
                alarms[name] = {
                    "actual": actual,
                    "matched_age": age_stats,
                    "matched_uniform_expected": random_stats,
                    "warning_counts_by_history": mask.sum(1),
                    "protected_loss_advantage_vs_age": interval(
                        actual["protected_loss_per_decision_pp"] - age_stats["protected_loss_per_decision_pp"]
                    ),
                    "positive_learning_cost_vs_age": interval(
                        actual["forgone_positive_learning_per_decision_pp"]
                        - age_stats["forgone_positive_learning_per_decision_pp"]
                    ),
                }
                arrays[f"matched_age_{horizon}_{name}"] = age
            panel_harm = data["support_targets"][..., 0] >= self.material
            reader_disagreement = float((panel_harm.any(-1) != panel_harm.all(-1)).mean())
            summary["horizons"][str(horizon)] = {
                "steps": data["steps"],
                "histories": len(truth),
                "decisions_per_history": truth.shape[1],
                "total_decisions": truth.shape[0] * truth.shape[1],
                "harm_prevalence": float((truth[..., 0] >= self.material).mean()),
                "support_draw_harm_disagreement_fraction": reader_disagreement,
                "models": models,
                "alarms": alarms,
            }
        summary["statistical_model_fits"] = len(folds)
        summary["primary_candidates"] = [
            name
            for name, value in summary["horizons"][str(primary_horizon)]["models"].items()
            if value["exploratory_screen"]
        ]
        return summary, arrays, folds
