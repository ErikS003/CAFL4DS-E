"""Safety gates must abstain when accuracy or nighttime evidence fails."""

from scripts.run_health_gate import TARGETS, select


def test_night_guard_rejects_accuracy_only_gain() -> None:
    """A tempting accuracy forecast cannot override the nighttime guardrail."""
    predictions = {k: {1.0: 0.0, 0.5: 0.002, 0.0: 0.001} for k in TARGETS}
    predictions["mean_night"][0.5] = -0.001
    margins = dict.fromkeys(TARGETS, 0.0)
    assert select(predictions, margins, "positive_accuracy", 0.0) == 0.5
    assert select(predictions, margins, "night_guard", 0.0) == 0.0


def test_empirical_margin_abstains_and_uses_strict_accuracy_threshold() -> None:
    """Uncertain or exactly threshold-sized benefits keep the reference action."""
    predictions = {k: {1.0: 0.0, 0.5: 0.001, 0.0: 0.0005} for k in TARGETS}
    margins = dict.fromkeys(TARGETS, 0.001)
    assert select(predictions, margins, "empirical_margin", 0.0005) == 1.0
    margins = dict.fromkeys(TARGETS, 0.0)
    assert select(predictions, margins, "empirical_margin", 0.001) == 1.0
    assert select(predictions, margins, "empirical_margin", 0.0005) == 0.5
    predictions["robust_night"][0.5] = -0.001
    assert select(predictions, margins, "empirical_margin", 0.0005) == 1.0
