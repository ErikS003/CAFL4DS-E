"""Check disagreement, relative retention and connection to the replay actuator."""

from __future__ import annotations

import numpy as np

from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.replay_controller import ReplayAllocationController
from cafl4ds.replay_response import ResponseModel


def model(low: float, high: float) -> ResponseModel:
    """Construct an intercept-only categorical forecast with known benefits."""
    return ResponseModel((), np.array([]), np.array([]), np.array([low, high]), 1.0)


def rows() -> list[dict]:  # type: ignore[type-arg]
    """Supply features without any response outcome."""
    return [{"state": "decision", "action": a, "features": {}} for a in (8, 16, 24)]


def test_agreement_applies_to_actual_filter() -> None:
    """A consistently beneficial, night-safe proposal changes the actuator."""
    controller = ReplayAllocationController((model(0.4, 0.1),) * 3, (model(0.2, 0.1),) * 3, 16)
    selector = ReplayRatio(16)
    assert controller.apply(selector, rows())["action"] == 8
    assert selector.replay_count == 8


def test_one_training_pool_disagreement_abstains() -> None:
    """The full model cannot overrule a negative single-pool benefit forecast."""
    controller = ReplayAllocationController(
        (model(0.4, 0.1), model(-0.1, 0.1), model(0.3, 0.1)), (model(0.2, 0.1),) * 3, 16
    )
    assert controller.decide(rows())["action"] == 16


def test_night_guard_compares_with_training_constant() -> None:
    """Being safe relative to16 is insufficient when the actual baseline is24."""
    controller = ReplayAllocationController((model(0.5, 0.2),) * 3, (model(0.1, 0.3),) * 3, 24)
    assert controller.decide(rows())["action"] == 24
