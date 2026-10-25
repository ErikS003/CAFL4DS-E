"""Verify the frozen gate cannot promote a model using only one successful comparison."""

import copy
from typing import Any

import pytest

from scripts.report_utility_confirmation import HYPOTHESES, assess


def fixture_report() -> dict[str, Any]:
    """Create a small report with successful frozen comparisons."""
    fold = {"tests": {}, "matched_contrasts": {}, "frozen_domain_tests": {}}
    for role in ("test_seed", "test_time"):
        fold["tests"][role] = {}
        fold["matched_contrasts"][role] = {}
        fold["frozen_domain_tests"][role] = {"A": {}, "B": {}}
        for model, _, contrast in HYPOTHESES.values():
            fold["tests"][role][model] = {
                "gain_over_random": 0.2,
                "by_snapshot": [{"step": step, "gain_over_random": 0.2} for step in (36, 40, 44)],
            }
            fold["matched_contrasts"][role][contrast] = {
                "mean_gain_difference": 0.1,
                "positive_snapshot_fraction": 1.0,
                "by_seed": {"6617": 0.1},
            }
            for domain in ("A", "B"):
                fold["frozen_domain_tests"][role][domain][model] = {"gain_over_random": 0.2}
    return {"horizon": 4, "primary_origin": 36, "folds": {"36": fold}}


@pytest.mark.parametrize("role", ["test_seed", "test_time"])
@pytest.mark.parametrize("comparison", ["random", "baseline"])
def test_gate_requires_both_positive_comparisons_on_both_holdouts(role: str, comparison: str) -> None:
    """A tie against random or loss against baseline blocks only that hypothesis."""
    report = fixture_report()
    model, _, contrast = HYPOTHESES["raw_context"]
    if comparison == "random":
        report["folds"]["36"]["tests"][role][model]["gain_over_random"] = 0.0
    else:
        report["folds"]["36"]["matched_contrasts"][role][contrast]["mean_gain_difference"] = -0.01
    original = copy.deepcopy(report)
    result = assess(report)
    assert result["qualified_hypotheses"] == ["centered_health"]
    assert report == original


def test_transition_diagnostic_is_not_a_posthoc_replacement_for_the_gate() -> None:
    """Preserve an unfavorable transition diagnostic even with a passing mean gate."""
    report = fixture_report()
    model = HYPOTHESES["centered_health"][0]
    report["folds"]["36"]["tests"]["test_time"][model]["by_snapshot"][0]["gain_over_random"] = -0.4
    result = assess(report)
    assert result["qualified_hypotheses"] == ["raw_context", "centered_health"]
    assert result["hypotheses"]["centered_health"]["transition_gain_over_random"] == -0.4


def test_confirmation_rejects_a_substituted_short_horizon() -> None:
    """The sensitivity horizon cannot replace the nominated four-update outcome."""
    report = fixture_report()
    report["horizon"] = 1
    with pytest.raises(ValueError, match="four-update"):
        assess(report)
