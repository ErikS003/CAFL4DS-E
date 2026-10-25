"""Cached stopping-choice ceilings with domain constraints and matched mean work."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.stats import t

DOSES = np.array([0, 4, 8, 16, 32])
TOL = 1e-10


def query_halves(query: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each query class/domain cell by fixed identity hash, not outcomes."""
    a = np.zeros(len(query), dtype=bool)
    for domain in (0, 1):
        for label in (0, 1):
            indices = np.flatnonzero((labels == label) & (domains == domain))
            if len(indices) != 100:
                raise ValueError("Four cells of one hundred queries required")
            ranked = sorted(
                indices,
                key=lambda i: hashlib.sha256(f"stopping-headroom:20261014:{int(query[i])}".encode()).hexdigest(),
            )
            a[ranked[:50]] = True
    return a, ~a


def oracle_choices(scores: np.ndarray, retention: bool) -> np.ndarray:
    """Choose maximum balanced accuracy, ties favor fewer updates."""
    selected = []
    for row in scores:
        feasible = np.all(row[:, 1:] - row[0, 1:] >= -1 - TOL, axis=1) if retention else np.ones(5, dtype=bool)
        best = row[feasible, 0].max()
        selected.append(int(np.flatnonzero(feasible & (row[:, 0] >= best - TOL))[0]))
    return np.array(selected)


def fixed_choice(scores: np.ndarray) -> int:
    """Select the common balanced-accuracy optimum on the selection partition."""
    means = scores[:, :, 0].mean(0)
    return int(np.flatnonzero(means >= means.max() - TOL)[0])


def mixture(scores: np.ndarray, budget: float) -> np.ndarray:
    """Find a state-independent distribution under expected work/domain limits."""
    average = scores.mean(0)
    constraints = np.vstack((DOSES, -average[:, 1], -average[:, 2]))
    limits = np.array([budget, 1 - average[0, 1], 1 - average[0, 2]])
    result = linprog(
        -average[:, 0], A_ub=constraints, b_ub=limits, A_eq=np.ones((1, 5)), b_eq=[1], bounds=(0, 1), method="highs"
    )
    if not result.success:
        raise ValueError(f"State-independent mixture failed: {result.message}")
    return result.x


def interval(values: np.ndarray, family: int = 1) -> dict[str, Any]:
    """Report conditional six-source heterogeneity, not selection-bias correction."""
    values = np.asarray(values, dtype=float)
    width = t.ppf(1 - 0.05 / (2 * family), 5) * values.std(ddof=1) / np.sqrt(6)
    return {
        "mean_pp": float(values.mean()),
        "ci95_pp": [float(values.mean() - width), float(values.mean() + width)],
        "source_values_pp": values.tolist(),
    }


def evaluation(scores: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    """Read a frozen choice on the evaluation partition."""
    chosen = scores[np.arange(len(scores)), choices]
    delta = chosen - scores[:, 0]
    return {
        "scores": chosen.tolist(),
        "mean_accuracy": chosen.mean(0).tolist(),
        "mean_updates": float(DOSES[choices].mean()),
        "dose_counts": {str(int(d)): int(sum(DOSES[choices] == d)) for d in DOSES},
        "mean_effect_vs_hold": delta.mean(0).tolist(),
        "domain_loss_gt_1_cases": int(np.any(delta[:, 1:] < -1 - TOL, axis=1).sum()),
        "worst_domain_effect_pp": float(delta[:, 1:].min()),
        "mean_worst_domain_effect_pp": float(delta[:, 1:].min(1).mean()),
    }


def group_values(values: np.ndarray) -> np.ndarray:
    """Average the three checkpoint cases within each of six source histories."""
    return np.asarray(values).reshape(6, 3, -1).mean(1)


def analysis(tables: dict[str, np.ndarray]) -> dict[str, Any]:
    """Keep pooled hindsight, query-half sensitivity and cost matching separate."""
    full = tables["all"]
    fixed = fixed_choice(full)
    hindsight = {}
    for name, retention in (("unconstrained", False), ("retention_aware", True)):
        choices = oracle_choices(full, retention)
        selected = full[np.arange(18), choices]
        delta = group_values(selected - full[:, fixed])
        hindsight[name] = {
            "choices": choices.tolist(),
            "evaluation": evaluation(full, choices),
            "vs_best_fixed": {m: interval(delta[:, i]) for i, m in enumerate(("balanced", "day", "night"))},
        }
    folds = {}
    for name, selection, testing in (("A_to_B", tables["A"], tables["B"]), ("B_to_A", tables["B"], tables["A"])):
        choices = oracle_choices(selection, True)
        unconstrained = oracle_choices(selection, False)
        chosen = testing[np.arange(18), choices]
        budget = float(DOSES[choices].mean())
        distribution = mixture(selection, budget)
        mixture_scores = np.einsum("k,nkm->nm", distribution, testing)
        fixed = fixed_choice(selection)
        gaps = group_values(chosen - mixture_scores)
        holds = group_values(chosen - testing[:, 0])
        contrasts = {
            f"vs_mixture_{m}": {"ordinary": interval(gaps[:, i]), "adjusted": interval(gaps[:, i], 10)}
            for i, m in enumerate(("balanced", "day", "night"))
        }
        contrasts.update(
            {
                f"vs_hold_{m}": {"ordinary": interval(holds[:, i + 1]), "adjusted": interval(holds[:, i + 1], 10)}
                for i, m in enumerate(("day", "night"))
            }
        )
        evidence = {
            "material_gain": float(gaps[:, 0].mean()) >= 0.5,
            "positive_adjusted_gain": contrasts["vs_mixture_balanced"]["adjusted"]["ci95_pp"][0] > 0,
            "retention_bounds": all(
                contrasts[k]["adjusted"]["ci95_pp"][0] >= -1 for k in contrasts if k != "vs_mixture_balanced"
            ),
            "positive_sources": int(sum(gaps[:, 0] > 0)) >= 4,
        }
        folds[name] = {
            "retention_choices": choices.tolist(),
            "oracle": evaluation(testing, choices),
            "unconstrained_choices": unconstrained.tolist(),
            "unconstrained_oracle": evaluation(testing, unconstrained),
            "best_fixed_dose": int(DOSES[fixed]),
            "best_fixed": evaluation(testing, np.full(18, fixed)),
            "mixture_probabilities": distribution.tolist(),
            "mixture_mean_updates": float(distribution @ DOSES),
            "mixture_mean_accuracy": mixture_scores.mean(0).tolist(),
            "mixture_expected_case_scores": mixture_scores.tolist(),
            "mixture_domain_loss_gt_1_cases": int(
                np.any(mixture_scores[:, 1:] - testing[:, 0, 1:] < -1 - TOL, axis=1).sum()
            ),
            "contrasts": contrasts,
            "screen": evidence,
            "investigation_screen_passed": all(evidence.values()),
        }
    return {
        "fixed_curve": [{"dose": int(d), **evaluation(full, np.full(18, k))} for k, d in enumerate(DOSES)],
        "best_pooled_fixed_dose": int(DOSES[fixed_choice(full)]),
        "hindsight": hindsight,
        "query_half_folds": folds,
        "investigation_screen_passed": all(f["investigation_screen_passed"] for f in folds.values()),
        "controller_qualified": False,
        "fresh_confirmation": False,
    }
