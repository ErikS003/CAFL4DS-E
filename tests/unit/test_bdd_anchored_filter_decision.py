"""Protect domain guardrails and separate efficiency from content contribution."""

from scripts.report_bdd_anchored_filter import decision


def contrasts() -> dict:
    """Build an otherwise qualifying small synthetic set of paired intervals."""
    return {name: {metric: {"ci95_pp": [.1, .3]} for metric in ("balanced", "A", "B")}
            for name in ("uniform_minus_full", "anchored_minus_full", "uniform_minus_frozen",
                         "anchored_minus_frozen", "anchored_minus_uniform", "anchored_minus_random")}


def test_uniform_efficiency_can_pass_without_added_content_value() -> None:
    """A simpler useful filter must not be hidden by a failed content upgrade."""
    rows = contrasts()
    rows["anchored_minus_uniform"]["balanced"]["ci95_pp"] = [-.2, .4]
    result = decision(rows, {"uniform": .4, "anchored": .4})
    assert result["efficiency_qualified"]["uniform"]
    assert not result["added_content_value_qualified"]
    assert not result["controller_qualified"]
    assert not result["independent_confirmation"]


def test_night_loss_cannot_hide_behind_balanced_gain() -> None:
    """Each candidate independently fails its night guardrail at the strict margin."""
    rows = contrasts()
    for candidate in ("uniform", "anchored"):
        rows[f"{candidate}_minus_full"]["B"]["ci95_pp"] = [-1, .2]
    result = decision(rows, {"uniform": .4, "anchored": .4})
    assert result["decision"] == "no_qualified_static_efficiency"
    assert result["added_content_value_qualified"]


def test_content_gain_does_not_override_cost_or_frozen_failure() -> None:
    """Neither an expensive candidate nor unresolved adaptation qualifies."""
    rows = contrasts()
    rows["uniform_minus_frozen"]["balanced"]["ci95_pp"] = [0, .2]
    result = decision(rows, {"uniform": .4, "anchored": .249})
    assert result["decision"] == "no_qualified_static_efficiency"
    assert result["added_content_value_qualified"]
