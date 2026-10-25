"""Admission clocks, rate recovery, repeated alarms and exact work controls."""

import random

import numpy as np
import pytest

from cafl4ds.dnd_health_rate import HealthRate, static_mask


def test_quarter_half_three_quarters_full_recovery_counts_arrivals() -> None:
    """The ramp accepts8/16/24/32 complete batches in successive32-arrival blocks."""
    control = HealthRate()
    control.last_alarm = 148
    result = [control.admit(step) for step in range(149, 277)]
    for i, (expected, numerator) in enumerate(((8, 1), (16, 2), (24, 3), (32, 4))):
        block = result[32 * i : 32 * (i + 1)]
        assert sum(take for take, _ in block) == expected
        assert all(rate == numerator for _, rate in block)


def test_confirmed_alarm_changes_only_the_next_arrival_and_keeps_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observing after an update cannot retrospectively reject that update."""
    monkeypatch.setattr("cafl4ds.dnd_health_rate.reading", lambda a, b: np.array([0.2, 0, 0]))
    control = HealthRate()
    bank = np.arange(60).reshape(20, 3)
    control.observe(132, bank)
    assert control.admit(140) == (True, 4)
    assert not control.observe(140, bank)
    assert control.admit(148) == (True, 4)
    assert control.observe(148, bank)
    assert control.admit(149) == (False, 1)
    np.testing.assert_array_equal(control.reference, bank)
    with pytest.raises(ValueError, match="rebased"):
        control.observe(132, bank)


def test_persistent_unhealthy_readings_reset_the_ramp_without_freezing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated alarms may maintain quarter admission, with training still active."""
    monkeypatch.setattr("cafl4ds.dnd_health_rate.reading", lambda a, b: np.array([0, 0.2, 0]))
    control = HealthRate()
    bank = np.ones((20, 3))
    control.observe(132, bank)
    control.observe(140, bank)
    assert control.observe(148, bank)
    decisions = []
    for step in range(149, 181):
        decisions.append(control.admit(step))
        if step in (156, 164, 172, 180):
            control.observe(step, bank)
    assert [row["step"] for row in control.events if row["alarm"]] == [148, 164, 180]
    assert sum(take for take, _ in decisions) == 8
    assert all(rate == 1 for _, rate in decisions)


def test_auxiliary_evaluations_never_increment_alarm_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dense offline checkpoint cannot create a control pulse."""
    monkeypatch.setattr("cafl4ds.dnd_health_rate.reading", lambda a, b: np.array([0, 0, 0.2]))
    control = HealthRate()
    bank = np.ones((20, 3))
    control.observe(132, bank)
    assert not control.observe(140, bank)
    assert not control.observe(144, bank)
    assert control.streak == 1
    assert control.observe(148, bank)


def test_clear_observations_interrupt_a_streak_and_allow_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone unhealthy reading cannot reset gradual recovery."""
    values = iter(([0.2, 0, 0], [0, 0, 0], [0.2, 0, 0], [0.2, 0, 0]))
    monkeypatch.setattr("cafl4ds.dnd_health_rate.reading", lambda a, b: np.array(next(values)))
    control = HealthRate()
    control.observe(132, np.ones((20, 3)))
    fired = [control.observe(step, np.ones((20, 3))) for step in (140, 148, 156, 164)]
    assert fired == [False, False, False, True]


def test_rate_decisions_do_not_consume_numpy_or_python_learner_randomness() -> None:
    """Selection is deterministic and has no hidden stochastic coupling."""
    control = HealthRate()
    control.last_alarm = 148
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    for step in range(149, 449):
        control.admit(step)
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(after[1], numpy_before[1])


def test_even_pacing_preserves_each_adaptive_work_budget_and_shared_prefix() -> None:
    """Matched controls use an exact count even at awkward rates and suffix lengths."""
    for fork in (148, 180, 396, 404, 436):
        for count in (0, 1, (448 - fork) // 4, (448 - fork) // 2, 448 - fork):
            mask = static_mask(fork, count)
            assert mask[:fork].all() and mask[fork:].sum() == count
            assert len(mask) == 448


def test_quarter_mask_trains_one_complete_batch_in_four_after_the_alarm() -> None:
    """Quarter image admission preserves the existing32-image optimization batch."""
    mask = static_mask(148)
    assert mask[:148].all()
    np.testing.assert_array_equal(mask[148:160], [False, False, False, True] * 3)
