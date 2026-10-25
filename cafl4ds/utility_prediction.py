"""Grouped utility prediction with seed/time holdouts and predeclared feature ablations."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def repeat_reliability(rows: list[dict[str, Any]], horizon: int, target: str) -> dict[str, Any]:
    """Separate shared reference offsets from noise that actually changes candidate ordering."""
    groups: dict[tuple[int, int], dict[int, dict[int, float]]] = defaultdict(dict)
    for row in rows:
        repeats = groups[row["seed"], row["step"]]
        repeats.setdefault(row["repeat"], {})[row["candidate"]] = row["gain"][str(horizon)]["development"][target]
    centered_sd, correlations = [], []
    for repeats in groups.values():
        candidates = sorted(next(iter(repeats.values())))
        if any(sorted(values) != candidates for values in repeats.values()):
            raise ValueError("Repeated contexts must measure the same candidates")
        values = np.array([[outcomes[candidate] for candidate in candidates] for outcomes in repeats.values()])
        if len(values) < 2:
            continue
        centered = values - values.mean(axis=1, keepdims=True)
        centered_sd.extend(centered.std(axis=0).tolist())
        for first, second in combinations(values, 2):
            if np.ptp(first) > 0 and np.ptp(second) > 0:
                correlations.append(float(spearmanr(first, second).statistic))
    return {
        "mean_centered_repeat_std": float(np.mean(centered_sd)) if centered_sd else None,
        "mean_pairwise_repeat_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "repeat_rank_pairs": len(correlations),
    }


def aggregate_repeats(rows: list[dict[str, Any]], horizon: int, target: str) -> list[dict[str, Any]]:
    """Treat repeated counterfactual contexts as measurements of one candidate, not IID rows."""
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["seed"], row["step"], row["candidate"]].append(row)
    result = []
    for key, repeats in sorted(groups.items()):
        first = repeats[0]
        if any(row["features"] != first["features"] for row in repeats):
            raise ValueError("Pre-decision features changed across intervention repeats")
        outcomes = {
            role: np.array([row["gain"][str(horizon)][role][target] for row in repeats])
            for role in ("development", "holdout")
        }
        result.append(
            {
                "seed": key[0],
                "step": key[1],
                "candidate": key[2],
                "visit": first["visit"],
                "arrival_id": first["arrival_id"],
                "features": first["features"],
                "development": float(outcomes["development"].mean()),
                "holdout": float(outcomes["holdout"].mean()),
                "repeat_std": float(outcomes["development"].std()),
                "repeat_sign_agreement": float(
                    (np.sign(outcomes["development"]) == np.sign(outcomes["development"].mean())).mean()
                ),
                "repeats": len(repeats),
            }
        )
    return result


def split_examples(examples: list[dict[str, Any]], seeds: list[int]) -> dict[str, list[int]]:
    """Hold out the last visit for new images and whole seeds for unseen learning states."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Need distinct fit, validation and test seeds")
    roles: dict[str, list[int]] = {key: [] for key in ("fit", "validation", "test_seed", "test_time")}
    for index, row in enumerate(examples):
        if row["seed"] not in seeds:
            raise ValueError("Unregistered seed")
        if row["visit"] == 3:
            roles["test_time"].append(index)
        elif row["visit"] in (0, 1, 2):
            role = "fit" if row["seed"] in seeds[:-2] else "validation" if row["seed"] == seeds[-2] else "test_seed"
            roles[role].append(index)
        else:
            raise ValueError("Unrecognized recurrence visit")
    if not all(roles.values()):
        raise ValueError("Need early and final-visit snapshots in fit, validation and test seeds")
    if len({index for indices in roles.values() for index in indices}) != len(examples):
        raise ValueError("Prediction splits overlap")
    # Time test candidates are unique arrivals, not remeasurements of fitted images.
    fit_images = {examples[i]["arrival_id"] for i in roles["fit"] + roles["validation"]}
    time_images = {examples[i]["arrival_id"] for i in roles["test_time"]}
    if fit_images & time_images:
        raise ValueError("Training candidate images leaked into the time holdout")
    return roles


def feature_matrix(examples: list[dict[str, Any]], kind: str) -> tuple[np.ndarray, list[str]]:
    """Use image/health interactions to let a shared health state change image ordering."""
    all_keys = sorted(examples[0]["features"])
    image = [key for key in all_keys if not key.startswith(("health_", "gradient_", "context_"))]
    health = [key for key in all_keys if key.startswith("health_") and not key.startswith("health_gradient_")]
    gradient = [key for key in all_keys if key.startswith("gradient_")]
    basic = ["reconstruction_mean", "coverage_deficit"]
    drift = basic + ["health_cosine_drift"]
    variants = {
        "loss_only": ["reconstruction_mean"],
        "image_only": image,
        "health_only": health,
        "loss_coverage": basic,
        "loss_coverage_health": drift,
        "loss_coverage_drift": drift,
        "loss_coverage_drift_gradient": drift + gradient,
        "gradient_only": gradient,
        "image_gradient": image + gradient,
        "image_health": image + health,
        "image_health_interactions": image + health,
    }
    if kind not in variants:
        raise ValueError(f"Unknown ablation: {kind}")
    if kind in ("loss_coverage_drift_gradient", "gradient_only", "image_gradient") and len(gradient) != 2:
        raise ValueError("Gradient signals were not collected")
    keys = variants[kind]
    matrix = np.array([[row["features"][key] for key in keys] for row in examples], dtype=np.float64)
    if kind in ("loss_coverage_drift", "loss_coverage_drift_gradient"):
        cross_keys = [f"{key}*health_cosine_drift" for key in ("reconstruction_mean", "coverage_deficit")]
        cross = np.array(
            [
                [
                    row["features"][key] * row["features"]["health_cosine_drift"]
                    for key in ("reconstruction_mean", "coverage_deficit")
                ]
                for row in examples
            ]
        )
        matrix, keys = np.concatenate((matrix, cross), axis=1), keys + cross_keys
    if kind == "image_health_interactions":
        cross_keys = [f"{i}*{h}" for i in image for h in health]
        cross = np.array(
            [[row["features"][i] * row["features"][h] for i in image for h in health] for row in examples],
            dtype=np.float64,
        )
        matrix, keys = np.concatenate((matrix, cross), axis=1), keys + cross_keys
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite predictor inputs")
    return matrix, keys


def ranking_metrics(examples: list[dict[str, Any]], scores: np.ndarray, role: str, top_k: int = 1) -> dict[str, Any]:
    """Rank only within a common pre-decision snapshot, after averaging context repeats."""
    if len(examples) != len(scores) or top_k < 1:
        raise ValueError("Invalid ranking dimensions")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["seed"], row["step"]].append(index)
    records = []
    for (seed, step), indices in sorted(groups.items()):
        truth = np.array([examples[i][role] for i in indices])
        predicted = scores[indices]
        k = min(top_k, len(indices))
        selected = np.argsort(-predicted, kind="stable")[:k]
        cutoff = predicted[selected[-1]]
        strict, tied = predicted > cutoff, predicted == cutoff
        weights = strict.astype(float)
        weights[tied] = (k - strict.sum()) / tied.sum()
        oracle = np.argsort(-truth, kind="stable")[:k]
        correlation = None
        if np.ptp(truth) > 0 and np.ptp(predicted) > 0:
            correlation = float(spearmanr(truth, predicted).statistic)
        chosen, best, random = float(truth @ weights / k), float(truth[oracle].mean()), float(truth.mean())
        records.append(
            {
                "seed": seed,
                "step": step,
                "candidates": len(indices),
                "k": k,
                "selected_gain": chosen,
                "oracle_gain": best,
                "random_expected_gain": random,
                "gain_over_random": chosen - random,
                "regret": best - chosen,
                "rank_correlation": correlation,
                "selected_positive_fraction": float((truth > 0) @ weights / k),
            }
        )
    correlations = [record["rank_correlation"] for record in records if record["rank_correlation"] is not None]
    return {
        "snapshots": len(records),
        "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
        **{
            key: float(np.mean([record[key] for record in records]))
            for key in (
                "selected_gain",
                "oracle_gain",
                "random_expected_gain",
                "gain_over_random",
                "regret",
                "selected_positive_fraction",
            )
        },
        "by_snapshot": records,
    }
