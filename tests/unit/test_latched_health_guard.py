"""Activation, retained monitoring, chronology and transactional state-cutoff behavior."""

import pytest

from cafl4ds.latched_health_guard import LatchedHealthGuard


def test_unreadable_tasks_do_not_block_learning_and_activation_cannot_alarm():
    guard = LatchedHealthGuard()
    assert guard.observe(0, [1, 1], [0.1, 0.1])["active"] == [False, False]
    assert guard.observe(4, [2, 2], [0.3, 0.1])["action"] == "continue"
    assert guard.active.tolist() == [True, False]


def test_learned_task_stays_protected_after_readability_disappears():
    guard = LatchedHealthGuard()
    guard.observe(0, [1, 0.6], [0.1, 0.4])
    result = guard.observe(4, [1, 0.7], [0.1, 0.01])
    assert result["action"] == "pause" and result["cutoff"] == 4
    assert result["violated_domains"] == [1]


def test_transactional_guard_selects_previous_accepted_checkpoint():
    guard = LatchedHealthGuard(mode="rollback")
    guard.observe(0, [1, 0.6], [0.1, 0.4])
    guard.observe(4, [1, 0.58], [0.1, 0.4])
    result = guard.observe(8, [1, 0.7], [0.1, 0.4])
    assert result["cutoff"] == 4


def test_best_observed_health_is_retained_and_unknown_task_is_ignored():
    guard = LatchedHealthGuard()
    guard.observe(0, [0.8, 0.8], [0.4, 0.1])
    guard.observe(4, [0.6, 10], [0.4, 0.1])
    assert guard.observe(8, [0.7, 20], [0.4, 0.1])["action"] == "pause"


def test_stopped_guard_rejects_future_data_and_backwards_time_is_invalid():
    guard = LatchedHealthGuard()
    guard.observe(4, [0.6, 0.6], [0.4, 0.4])
    with pytest.raises(ValueError):
        guard.observe(0, [0.6, 0.6], [0.4, 0.4])
    guard.observe(8, [0.8, 0.8], [0.1, 0.1])
    with pytest.raises(ValueError):
        guard.observe(16, [0, 0], [1, 1])
