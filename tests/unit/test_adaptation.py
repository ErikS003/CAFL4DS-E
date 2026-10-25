"""Tests for the P1.2 adapted-versus-frozen report reductions."""

from __future__ import annotations

import pytest

from cafl4ds.adaptation import aggregate_seed_reports, condition_report
from cafl4ds.eval import PerEraProbe


def _probe(matrix: dict[int, dict[int, float]]) -> PerEraProbe:
    """Build a reduction-only evaluator without materializing images."""
    probe = object.__new__(PerEraProbe)
    probe.probe = "linear"
    probe.matrix = matrix
    return probe


def test_condition_report_separates_current_and_past_gains() -> None:
    """Current-condition and earlier-condition comparisons stay explicit in the artifact."""
    adapted = _probe({0: {0: 0.7}, 1: {0: 0.6, 1: 0.8}})
    frozen = _probe({0: {0: 0.5}, 1: {0: 0.5, 1: 0.55}})
    report = condition_report(adapted, frozen, {0: "day", 1: "night"})
    assert report["trajectory"][1]["current_gain"] == pytest.approx(0.25)
    assert report["trajectory"][1]["past_gain_mean"] == pytest.approx(0.1)
    assert {row["era_name"] for row in report["final_per_condition"]} == {"day", "night"}


def test_aggregate_reports_counts_paired_seed_wins() -> None:
    """The ensemble summary reports effect sizes and signs without a false significance claim."""
    seeds = []
    for seed, gain in enumerate((0.1, 0.0, -0.05)):
        seeds.append(
            {
                "seed": seed,
                "global": {
                    "knn": {"gain": gain},
                    "linear": {"gain": gain + 0.02},
                },
                "conditions": {
                    "final_per_condition": [
                        {"eval_era": 0, "era_name": "day", "gain": gain},
                        {"eval_era": 1, "era_name": "night", "gain": gain / 2},
                    ]
                },
            }
        )
    aggregate = aggregate_seed_reports(seeds)
    assert aggregate["global"]["knn"]["wins"] == 1
    assert aggregate["global"]["knn"]["ties"] == 1
    assert aggregate["global"]["knn"]["losses"] == 1
    assert aggregate["interpretation"]["exploratory"] is True
    assert aggregate["interpretation"]["adaptation_beats_frozen_all_seeds"]["knn"] is False
