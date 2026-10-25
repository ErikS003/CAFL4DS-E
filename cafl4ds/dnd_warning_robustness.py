"""Sensitivity of fixed warning decisions to reader draws and source omission."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.dnd_signal_diagnostic import interval, matched_age_selection, outcome_targets, warning_statistics

READERS = {
    "draw_61001": (0,),
    "draw_61007": (1,),
    "draw_61013": (2,),
    "pair_61001_61007": (0, 1),
    "pair_61001_61013": (0, 2),
    "pair_61007_61013": (1, 2),
    "pooled": (0, 1, 2),
}
INSTRUMENTS = ("rank", "variance", "cka", "combined")
COHORTS = {"all": tuple(range(8)), **{f"omit_{i:02d}": tuple(j for j in range(8) if j != i) for i in range(8)}}


def reader_targets(
    panels: np.ndarray, draws: tuple[int, ...], points: tuple[int, ...], steps: np.ndarray, horizon: int
) -> np.ndarray:
    """Average the selected scores before computing nonlinear loss targets."""
    curves = panels[:, :, list(draws)].mean(2)
    return np.array([outcome_targets(curve, points, steps, horizon) for curve in curves])


def recalibrated_age_risk(targets: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """A recipient and an omitted source never calibrate that recipient's age score."""
    if len(indices) < 2:
        raise ValueError("At least two retained source histories required")
    result = np.zeros(targets.shape[:2])
    for i in indices:
        others = [j for j in indices if j != i]
        result[i] = targets[others, :, 0].mean(0)
    return result


def metrics(targets: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Keep conditional warning quality and per-source positive learning costs."""
    result = warning_statistics(targets, weights, 1, 0.5, 1)
    result["positive_learning_per_decision_pp"] = (np.maximum(targets[..., 1:], 0) * weights[..., None]).mean(1)
    return result


def comparison(actual: dict, control: dict) -> dict[str, Any]:
    """Compare equally many hypothetical warning dates, retaining source units."""
    return {
        "loss_advantage_pp": interval(
            actual["protected_loss_per_decision_pp"] - control["protected_loss_per_decision_pp"]
        ),
        "positive_learning_cost_pp": interval(
            actual["positive_learning_per_decision_pp"] - control["positive_learning_per_decision_pp"]
        ),
    }


def sign_check(comparisons: dict) -> bool:
    """Require beneficial signs against both non-frozen controls in every domain."""
    if not {"age_recalibrated", "uniform"}.issubset(comparisons):
        raise ValueError("Both matched controls are required")
    return all(
        row["loss_advantage_pp"]["mean"] > 1e-12
        and np.all(np.asarray(row["positive_learning_cost_pp"]["mean"]) <= 1e-12)
        for name, row in comparisons.items()
        if name in ("age_recalibrated", "uniform")
    )


def robustness_screen(cells: list[dict]) -> dict[str, Any]:
    """Refuse pooled-only or source-sensitive success at the primary setting."""
    required_readers = ("draw_61001", "draw_61007", "draw_61013", "pooled")
    primary = {(r["reader"], r["cohort"]): r for r in cells if r["horizon"] == 32 and r["instrument"] == "combined"}
    required = [(reader, cohort) for reader in required_readers for cohort in COHORTS]
    if any(key not in primary for key in required):
        raise ValueError("The complete 36-cell primary robustness grid is required")
    canonical = primary[("pooled", "all")]
    interval_pass = {
        name: bool(canonical["comparisons"][name]["loss_advantage_pp"]["ci95"][0] > 0)
        for name in ("age_recalibrated", "uniform")
    }
    failures = [
        {"reader": reader, "cohort": cohort}
        for reader, cohort in required
        if not primary[(reader, cohort)]["sign_pass"]
    ]
    return {
        "passed": not failures and all(interval_pass.values()),
        "required_cells": len(required),
        "sign_cells_passed": len(required) - len(failures),
        "failed_sign_cells": failures,
        "pooled_interval_pass": interval_pass,
        "singleton_full_sign_pass": all(primary[(reader, "all")]["sign_pass"] for reader in required_readers[:3]),
        "pooled_omission_sign_pass": all(
            primary[("pooled", cohort)]["sign_pass"] for cohort in COHORTS if cohort != "all"
        ),
        "crossed_singleton_omission_sign_pass": all(
            primary[(reader, cohort)]["sign_pass"]
            for reader in required_readers[:3]
            for cohort in COHORTS
            if cohort != "all"
        ),
    }


class WarningRobustness:
    """A cached sensitivity audit of one fixed warning, without new policy fitting."""

    def analyze(
        self,
        panels: np.ndarray,
        points: tuple[int, ...],
        steps: dict[int, np.ndarray],
        alarms: dict[int, np.ndarray],
        frozen_age: dict[int, np.ndarray],
        first: list[dict],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Evaluate all declared reader/source combinations and preserve every result."""
        cells, agreement, actual, saved = [], {}, [], {}
        for horizon, dates in steps.items():
            all_targets = np.array(
                [reader_targets(panels, draws, points, dates, horizon) for draws in READERS.values()]
            )
            saved[f"targets_{horizon}"] = all_targets
            allocated = np.zeros((7, 9, 4, *alarms[horizon].shape[:2]), dtype=bool)
            age_scores = np.zeros((7, 9, *alarms[horizon].shape[:2]))
            for r, reader in enumerate(READERS):
                targets = all_targets[r]
                for c, (cohort, kept) in enumerate(COHORTS.items()):
                    ids = list(kept)
                    age_scores[r, c] = recalibrated_age_risk(targets, kept)
                    for j, instrument in enumerate(INSTRUMENTS):
                        warning = alarms[horizon][ids, :, j]
                        age = matched_age_selection(age_scores[r, c, ids], warning, dates)
                        allocated[r, c, j, ids] = age
                        uniform = np.broadcast_to(warning.mean(1)[:, None], warning.shape)
                        weights = {
                            "health": warning,
                            "age_frozen": frozen_age[horizon][j, ids],
                            "age_recalibrated": age,
                            "uniform": uniform,
                        }
                        stats = {name: metrics(targets[ids], mask) for name, mask in weights.items()}
                        contrasts = {
                            name: comparison(stats["health"], stats[name]) for name in weights if name != "health"
                        }
                        cells.append(
                            {
                                "horizon": horizon,
                                "reader": reader,
                                "cohort": cohort,
                                "instrument": instrument,
                                "histories": kept,
                                "warning_counts": warning.sum(1),
                                "metrics": stats,
                                "comparisons": contrasts,
                                "sign_pass": sign_check(contrasts),
                            }
                        )
            saved[f"age_selections_{horizon}"] = allocated
            saved[f"age_scores_{horizon}"] = age_scores
            singleton = all_targets[:3]
            harm_votes = (singleton[..., 0] >= 1).sum(0)
            useful_votes = (
                (singleton[..., 0] >= 1) & (singleton[..., 1] <= 0.5) & (singleton[..., 2:] <= 1).all(-1)
            ).sum(0)
            saved[f"harm_votes_{horizon}"] = harm_votes
            saved[f"useful_votes_{horizon}"] = useful_votes
            agreement[str(horizon)] = {
                "eligible_harm_vote_counts": [int((harm_votes == k).sum()) for k in range(4)],
                "mean_singleton_max_minus_pooled_max_pp": interval(
                    (singleton[..., 0].mean(0) - all_targets[-1, ..., 0]).mean(1)
                ),
                "instruments": {},
            }
            for j, instrument in enumerate(INSTRUMENTS):
                mask = alarms[horizon][..., j]
                agreement[str(horizon)]["instruments"][instrument] = {
                    "warning_count": int(mask.sum()),
                    "harm_vote_counts": [int(((harm_votes == k) & mask).sum()) for k in range(4)],
                    "useful_vote_counts": [int(((useful_votes == k) & mask).sum()) for k in range(4)],
                    "pooled_harm_count": int(((all_targets[-1, ..., 0] >= 1) & mask).sum()),
                }
            for row in (row for row in first if row["horizon"] == horizon):
                i = int(row["history"].split("_")[-1])
                available = row["stop"] in dates
                for r, reader in enumerate(READERS):
                    value = all_targets[r, i, list(dates).index(row["stop"])] if available else None
                    useful = bool(value[0] >= 1 and value[1] <= 0.5 and (value[2:] <= 1).all()) if available else None
                    actual.append(
                        {
                            "history": row["history"],
                            "horizon": horizon,
                            "stop": row["stop"],
                            "reader": reader,
                            "full_horizon_available": available,
                            "targets_pp": value,
                            "useful_hold": useful,
                        }
                    )
        summary = {
            "primary_horizon": 32,
            "primary_instrument": "combined",
            "readers": READERS,
            "cohorts": COHORTS,
            "cells": cells,
            "agreement": agreement,
            "actual_first_alarms": actual,
            "robustness_screen": robustness_screen(cells),
            "exploratory_reuse": True,
            "fresh_source_confirmation": False,
            "controller_qualified": False,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
            "new_regression_fits": 0,
        }
        return summary, saved
