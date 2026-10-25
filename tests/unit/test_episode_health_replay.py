"""Check causal episode alarms, persistence, complete restore, and equal-dose schedules."""

import numpy as np
import pytest

from cafl4ds.filters.episode_health_replay import EpisodeHealthReplay, timing_schedules
from scripts.run_paired_update_response import fingerprint


def feed(monkeypatch, control, step, values):
    """Supply deterministic health readings while exercising the real state machine."""
    monkeypatch.setattr("cafl4ds.filters.episode_health_replay.reading", lambda *_: np.array(values))
    return control.observe(step, np.ones((4, 3)))


def test_persistent_bad_episode_does_not_restart_timed_recovery(monkeypatch):
    """A latched bad state emits once while the original ramp completes normally."""
    control = EpisodeHealthReplay()
    control.observe(132, np.ones((4, 3)))
    assert not feed(monkeypatch, control, 140, [0.1, 0, 0])
    assert feed(monkeypatch, control, 148, [0, 0.1, 0])
    for step in range(156, 253, 8):
        assert not feed(monkeypatch, control, step, [0, 0, 0.2])
    assert control.last_alarm == 148 and control.latched
    for step, mix in {149: (8, 24), 180: (8, 24), 181: (16, 16), 213: (24, 8), 245: (32, 0)}.items():
        assert control.split(step) == mix
    assert sum(e["alarm"] for e in control.events) == 1


def test_hysteresis_needs_consecutive_low_checks_and_new_high_checks(monkeypatch):
    """Intermediate readings break the low streak; rearming alone does not emit an alarm."""
    control = EpisodeHealthReplay()
    control.observe(132, np.ones((4, 3)))
    values = [0.2, 0.2, 0.05, 0.07, 0.05, 0.05, 0.2, 0.07, 0.2, 0.2]
    alarms = [feed(monkeypatch, control, 140 + 8 * i, [v, 0, 0]) for i, v in enumerate(values)]
    assert [i for i, fired in enumerate(alarms) if fired] == [1, 9]
    assert [e["step"] for e in control.events if e["rearmed"]] == [180]
    assert control.last_alarm == 212
    assert control.split(213) == (8, 24)


def test_restore_preserves_latch_and_independent_reference(monkeypatch):
    """State restore includes intermediate low streak and all subsequent decisions."""
    first, second = EpisodeHealthReplay(), EpisodeHealthReplay()
    first.observe(132, np.ones((4, 3)))
    for step, value in ((140, 0.2), (148, 0.2), (156, 0.03)):
        feed(monkeypatch, first, step, [value, 0, 0])
    state = first.state_dict()
    second.load_state_dict(state)
    assert fingerprint(first.state_dict()) == fingerprint(second.state_dict())
    assert not np.shares_memory(first.reference, second.reference)
    for step, value in ((164, 0.04), (172, 0.2), (180, 0.2)):
        assert feed(monkeypatch, first, step, [value, 0, 0]) == feed(monkeypatch, second, step, [value, 0, 0])
    assert fingerprint(first.state_dict()) == fingerprint(second.state_dict())


def test_invalid_or_duplicate_observations_fail_without_rebasing():
    """Invalid measurements cannot be treated as health or replace a fixed reference."""
    control = EpisodeHealthReplay()
    with pytest.raises(ValueError, match="Missing"):
        control.observe(140, np.ones((4, 3)))
    control.observe(132, np.ones((4, 3)))
    with pytest.raises(ValueError, match="advance"):
        control.observe(132, np.ones((4, 3)))
    with pytest.raises(ValueError, match="Finite"):
        control.observe(140, np.full((4, 3), np.nan))
    assert control.last_observation == 132
    assert not control.observe(141, np.full((4, 3), np.nan))


def test_timing_controls_keep_exact_counts_and_startup():
    """The incomplete final block remains fixed and all schedules preserve allocation counts."""
    values = [32] * 132 + ([8] * 32 + [16] * 32 + [24] * 32 + [32] * 32) * 30 + [32] * 124
    controls = timing_schedules(values)
    assert len(controls) == 4
    for name, schedule in controls.items():
        assert schedule[:132] == values[:132] and sorted(schedule) == sorted(values)
        assert schedule != values
        if name.startswith("perm"):
            assert schedule[4092:] == values[4092:]
    assert timing_schedules(values) == controls
