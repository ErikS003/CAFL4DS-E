"""P1.2 adapted-versus-frozen comparison summaries.

The execution harness lives in ``scripts/run_adaptation_experiment.py``. This module keeps the
condition-matrix reduction and seed aggregation pure so the scientific comparison can be tested
without training a model.
"""

from __future__ import annotations

import statistics
from typing import Any

from cafl4ds.eval import PerEraProbe


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for an empty collection."""
    return statistics.fmean(values) if values else None


def condition_report(
    adapted: PerEraProbe,
    frozen: PerEraProbe,
    era_names: dict[int, str],
) -> dict[str, Any]:
    """Compare adapted and frozen probe accuracy on current and earlier regimes.

    Args:
        adapted: Probe-on-past evaluator recorded during adaptation.
        frozen: Evaluator recorded on the init-matched frozen twin at the same era checkpoints.
        era_names: Human-readable condition name for each era.

    Returns:
        The two accuracy matrices, a long-form matched history, current/past trajectory summaries,
        final per-condition gains, and standard forgetting summaries.
    """
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    checkpoints = sorted(set(adapted.matrix) & set(frozen.matrix))
    for after_era in checkpoints:
        live_row, frozen_row = adapted.matrix[after_era], frozen.matrix[after_era]
        common = sorted(set(live_row) & set(frozen_row))
        for eval_era in common:
            history.append(
                {
                    "after_era": after_era,
                    "eval_era": eval_era,
                    "era_name": era_names.get(eval_era, f"era{eval_era}"),
                    "is_current": eval_era == after_era,
                    "adapted_acc": live_row[eval_era],
                    "frozen_acc": frozen_row[eval_era],
                    "gain": live_row[eval_era] - frozen_row[eval_era],
                }
            )
        earlier = [era for era in common if era < after_era]
        current = after_era if after_era in common else None
        trajectory.append(
            {
                "after_era": after_era,
                "current_adapted": live_row.get(current) if current is not None else None,
                "current_frozen": frozen_row.get(current) if current is not None else None,
                "current_gain": (live_row[current] - frozen_row[current] if current is not None else None),
                "past_adapted_mean": _mean([live_row[era] for era in earlier]),
                "past_frozen_mean": _mean([frozen_row[era] for era in earlier]),
                "past_gain_mean": _mean([live_row[era] - frozen_row[era] for era in earlier]),
            }
        )

    final_after = checkpoints[-1] if checkpoints else None
    final_rows = [row for row in history if row["after_era"] == final_after]
    return {
        "probe": adapted.probe,
        "adapted_matrix": adapted.matrix,
        "frozen_matrix": frozen.matrix,
        "history": history,
        "trajectory": trajectory,
        "final_per_condition": final_rows,
        "adapted_summary": adapted.summary(),
        "frozen_summary": frozen.summary(),
    }


def _metric_summary(values: list[float]) -> dict[str, Any]:
    """Summarize paired gains without pretending three seeds establish significance."""
    return {
        "gains": values,
        "mean_gain": _mean(values),
        "stdev_gain": statistics.stdev(values) if len(values) > 1 else 0.0,
        "wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
        "n": len(values),
    }


def aggregate_few_shot(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-shot adapted-minus-frozen gains from matched seed reports."""
    shots = sorted({int(shot) for report in seed_reports for shot in report.get("few_shot", {})})
    return {
        str(shot): _metric_summary([float(report["few_shot"][str(shot)]["gain"]) for report in seed_reports])
        for shot in shots
    }


def aggregate_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate matched global and condition-specific gains across seeds.

    Args:
        seed_reports: Per-seed P1.2 reports produced by the experiment script.

    Returns:
        Exploratory multi-seed summaries for global probes and final per-condition accuracy.
    """
    global_summary = {
        probe: _metric_summary([float(report["global"][probe]["gain"]) for report in seed_reports])
        for probe in ("knn", "linear")
    }
    final_gains = [float(row["gain"]) for report in seed_reports for row in report["conditions"]["final_per_condition"]]
    per_era: dict[str, dict[str, Any]] = {}
    era_names = {
        str(row["eval_era"]): row["era_name"]
        for report in seed_reports
        for row in report["conditions"]["final_per_condition"]
    }
    for era, name in sorted(era_names.items(), key=lambda item: int(item[0])):
        gains = [
            float(row["gain"])
            for report in seed_reports
            for row in report["conditions"]["final_per_condition"]
            if int(row["eval_era"]) == int(era)
        ]
        per_era[era] = {"era_name": name, **_metric_summary(gains)}
    return {
        "study": "P1.2",
        "n_seeds": len(seed_reports),
        "seeds": [int(report["seed"]) for report in seed_reports],
        "global": global_summary,
        "final_condition_gain": _metric_summary(final_gains),
        "final_by_condition": per_era,
        "interpretation": {
            "exploratory": True,
            "adaptation_beats_frozen_all_seeds": {
                probe: summary["wins"] == summary["n"] and summary["n"] > 0 for probe, summary in global_summary.items()
            },
        },
    }
