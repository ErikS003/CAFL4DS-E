"""Behavioral regression tests for the failure modes seen in N78--N82."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cafl4ds.single_image_heuristic import HeuristicController

CFG = json.loads(
    (Path(__file__).parents[2] / "cafl4ds/configs/single_image_heuristic_controller_proposal.json").read_text()
)


def test_pessimistic_predictions_still_get_real_opportunities():
    c = HeuristicController()
    p, rate, reason = c.choose(0, 100, 0, -0.001, 0.05, CFG)
    assert (p, rate, reason) == (0.1, 0.25, "unfavorable")
    assert c.choose(1, -100, 0.5, -0.001, 0.2, CFG) == (0.25, 0.25, "fallback")


def test_covariance_without_other_damage_does_not_freeze():
    c = HeuristicController()
    anchor = np.ones(7)
    current = anchor.copy()
    current[2] = 5
    for _ in range(4):
        assert not c.health(current, anchor, CFG)
    assert c.mode == "normal"
    current[6] = 1.1
    c.health(current, anchor, CFG)
    c.health(current, anchor, CFG)
    assert c.mode == "caution"
    assert c.choose(0, -1, 0, -0.001, 0, CFG)[1] == 1 / 16
    c.health(anchor, anchor, CFG)
    c.health(anchor, anchor, CFG)
    assert c.mode == "normal"


def test_burst_precedes_positive_gate_and_one_commit_does_not_clear_stall():
    c = HeuristicController(recent_commits=[255])
    for i in range(256, 272):
        assert c.choose(i, -100, 0, -0.001, 0.99, CFG) == (1.0, 1 / 16, "burst")
        c.recent_commits.append(i)
    assert c.burst_remaining == 0
    assert c.choose(272, 1, 0, -0.001, 0.99, CFG)[1] == 0


def test_recovery_is_bounded_and_survives_serialization():
    c = HeuristicController()
    c.recover(384, CFG)
    assert c.choose(384, -100, 0, -0.001, 0.1, CFG) == (0.25, 1 / 64, "recovery")
    restored = HeuristicController(**json.loads(json.dumps(asdict(c))))
    assert restored.choose(640, -100, 0, -0.001, 0, CFG) == c.choose(640, -100, 0, -0.001, 0, CFG)
    assert c.mode == "caution"


def test_task_brake_waits_for_confirmation_but_handles_severe_damage():
    c = HeuristicController()
    anchor = np.array([0.61, 0.62, 0.60])
    assert c.task(anchor - 0.03, anchor, anchor, CFG) == "continue"
    assert c.task(anchor - 0.03, anchor, anchor, CFG) == "rollback"
    assert c.task(anchor - 0.06, anchor, anchor, CFG) == "rollback"
    assert c.task(anchor + 0.02, anchor, anchor, CFG) == "continue"
    assert c.task(anchor + 0.02, anchor, anchor, CFG) == "backup"


def test_numerical_cooldown_is_not_rewound_by_learner_restore():
    c = HeuristicController()
    c.recover(10, CFG, numerical=True)
    assert c.choose(73, -1, 0, -0.001, 0, CFG)[2] == "cooldown"
    assert c.choose(74, -1, 0, -0.001, 0, CFG)[2] == "recovery"
