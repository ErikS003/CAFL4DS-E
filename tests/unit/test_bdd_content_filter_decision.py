"""The scientific gate must count both domains, both comparators and monitoring cost."""

from __future__ import annotations

from typing import Any

from scripts.report_bdd_content_filter import decision


def passing_contrasts() -> dict[str, Any]:
    """Return a synthetic positive matched-work and noninferiority result."""
    return {name: {metric: {"ci95_pp": [.1, .5]} for metric in ("balanced", "A", "B")}
            for name in ("content_minus_random", "content_minus_fixed", "content_minus_full", "content_minus_frozen")}


def test_static_success_does_not_qualify_controller_or_independent_transfer() -> None:
    """An efficient static selector cannot silently promote a later research claim."""
    result = decision(passing_contrasts(), .3)
    assert result["decision"] == "content_half_qualified_for_separate_policy_test"
    assert result["controller_qualified"] is False
    assert result["independent_confirmation"] is False


def test_balanced_success_cannot_hide_failed_nighttime_guardrail() -> None:
    """The previous experiment's nighttime failure cannot be averaged away."""
    contrasts = passing_contrasts()
    contrasts["content_minus_full"]["B"]["ci95_pp"] = [-1.01, -.1]
    assert decision(contrasts, .3)["decision"] == "content_half_not_qualified"


def test_both_comparators_and_overhead_screen_are_required() -> None:
    """Positive fixed-relative quality does not rescue unresolved random-relative value."""
    contrasts = passing_contrasts()
    contrasts["content_minus_random"]["balanced"]["ci95_pp"] = [0.0, .5]
    assert decision(contrasts, .3)["decision"] == "content_half_not_qualified"
    assert decision(passing_contrasts(), .2499)["decision"] == "content_half_not_qualified"
