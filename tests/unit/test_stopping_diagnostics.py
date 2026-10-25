"""Distinguish point-budget, conditional and fixed-reference stopping promises."""

import copy

from cafl4ds.stopping_diagnostics import diagnostic
from tests.unit.test_cumulative_budget import episode, measured


def test_uncertain_original_reference_only_excluded_from_conditional_promise():
    rows = [episode(str(i)) for i in range(6)]
    for e in rows:
        e["points"][0]["warm"] = measured((0, 0.9, -0.9), width=0.3)
        e["points"][1]["acquired"] = measured((0.6, 0.6, 0.6))
    point = diagnostic(rows, "dual_point")
    acquired = diagnostic(rows, "acquired_conditional")
    assert point["informed_stop"]["blocks"] == acquired["informed_stop"]["blocks"] == [1] * 6
    assert not point["policy_qualified"] and not acquired["policy_qualified"]


def test_point_budget_cannot_buy_other_domain_loss_and_matching_preserves_work():
    rows = [episode(str(i)) for i in range(6)]
    for i, e in enumerate(rows):
        e["points"][1]["acquired"] = measured((0.4, 0.4, 0.4))
        e["points"][2]["acquired"] = measured((1, 3.5, -1.5))
    original = copy.deepcopy(rows)
    result = diagnostic(rows, "dual_point")
    assert result["informed_stop"]["blocks"] == [1] * 6
    assert result["matched_random_stop"]["ssl_updates"] == result["informed_stop"]["ssl_updates"]
    assert rows == original
