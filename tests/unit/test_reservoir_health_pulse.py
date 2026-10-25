"""Check pulse causality, retention invariance and exact-dose timing diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.historical_reservoir import HistoricalReservoir
from cafl4ds.filters.reservoir_health_pulse import PulseReservoir, ReservoirHealthPulse, shifted_pulses
from scripts.run_paired_update_response import fingerprint


def frames(step):
    """Make incoming identities directly observable in test pixels."""
    return torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1)


def test_pulse_boundaries_return_to_normal_even_while_latched():
    """An alarm affects the next 32 arrivals and never changes the observed arrival."""
    control = ReservoirHealthPulse()
    assert control.split(148) == (16, 16)
    control.last_alarm = 148
    control.latched = True
    with pytest.raises(ValueError, match="after"):
        control.split(148)
    assert control.split(149) == control.split(180) == (8, 24)
    assert control.split(181) == (16, 16)
    control.last_alarm = 196
    assert control.split(197) == control.split(228) == (8, 24)
    assert control.split(229) == (16, 16)


def test_normal_mixing_matches_immutable_constant_reservoir():
    """The new variable allocator preserves the successful constant recipe exactly."""
    a, b = HistoricalReservoir(), PulseReservoir()
    global_before = torch.get_rng_state().clone()
    for step in range(30):
        images, ctx = frames(step), SimpleNamespace(step=step)
        assert torch.equal(a.observe(images, ctx), b.observe(images, ctx, 16))
        assert a.last == b.last
    assert fingerprint(a.state_dict()) == fingerprint(b.state_dict())
    assert torch.equal(global_before, torch.get_rng_state())


def test_variable_dose_keeps_identical_resident_pixels_and_both_rng_states():
    """Extra replay changes nested subsets without changing causal memory or randomness."""
    a, b = PulseReservoir(), PulseReservoir()
    for step in range(50):
        images, ctx = frames(step), SimpleNamespace(step=step)
        x, y = a.observe(images, ctx, 16), b.observe(images, ctx, 24)
        assert len(x) == len(y) == 32
        assert set(b.last["current_ids"]) <= set(a.last["current_ids"])
        assert set(a.last["replay_ids"]) <= set(b.last["replay_ids"])
        assert fingerprint(a.state_dict()) == fingerprint(b.state_dict())
        assert all(i < step * 32 for i in b.last["replay_ids"])


def test_shifted_schedules_preserve_startup_and_truncated_terminal_dose():
    """Shifts preserve exactly the same added replay even for an incomplete last pulse."""
    values = np.full(4096, 16)
    values[0] = 32
    values[148:180] = 8
    values[4080:] = 8
    schedules = shifted_pulses(values)
    assert list(schedules) == ["shift_256", "shift_512", "shift_1024"]
    for offset, (name, actual) in zip((256, 512, 1024), schedules.items(), strict=True):
        np.testing.assert_array_equal(actual[:132], values[:132])
        np.testing.assert_array_equal(actual[132:], np.roll(values[132:], offset))
        assert sum(actual) == sum(values)
        assert actual.count(8) == 48
        assert name == f"shift_{offset}"


def test_state_roundtrip_preserves_future_variable_allocations():
    """Restoring a reservoir and controller reproduces later pulse decisions and samples."""
    memory = PulseReservoir()
    for step in range(12):
        memory.observe(frames(step), SimpleNamespace(step=step), 24)
    other = PulseReservoir()
    other.load_state_dict(memory.state_dict())
    for step in range(12, 16):
        ctx = SimpleNamespace(step=step)
        assert torch.equal(memory.observe(frames(step), ctx, 16), other.observe(frames(step), ctx, 16))
    control = ReservoirHealthPulse()
    control.last_alarm = 148
    control.latched = True
    restored = ReservoirHealthPulse()
    restored.load_state_dict(control.state_dict())
    assert restored.split(180) == control.split(180)
    assert fingerprint(restored.state_dict()) == fingerprint(control.state_dict())


def test_invalid_dose_rejected_without_memory_mutation():
    """An undeclared allocation cannot silently change memory eligibility."""
    memory = PulseReservoir()
    before = fingerprint(memory.marker())
    with pytest.raises(ValueError, match="16 or 24"):
        memory.observe(frames(0), SimpleNamespace(step=0), 8)
    assert fingerprint(memory.marker()) == before


def test_real_geometry_observations_latch_rearm_and_act_only_after_checks():
    """An actual geometry alarm connects to the timed pulse without early release on rearming."""
    reference = np.random.default_rng(17).normal(size=(64, 4))
    collapsed = np.repeat(reference[:1], 64, axis=0)
    control = ReservoirHealthPulse()
    assert not control.observe(132, reference)
    assert not control.observe(140, collapsed)
    assert control.split(148) == (16, 16)
    assert control.observe(148, collapsed)
    assert control.split(149) == (8, 24)
    assert not control.observe(156, reference)
    assert not control.observe(164, reference)
    assert not control.latched
    assert control.split(165) == (8, 24)
    assert not control.observe(172, collapsed)
    assert control.observe(180, collapsed)
    assert control.split(181) == control.split(212) == (8, 24)
    assert control.split(213) == (16, 16)


def test_timing_screen_requires_every_control_and_primary_protection():
    """A favorable comparator cannot hide a stronger control or failed protection margin."""
    from scripts.verify_reservoir_health_pulse import protection, timing_screen  # noqa: PLC0415

    candidate = np.zeros(13)
    candidate[:3] = [60, 60, 60]
    candidate[-1] = 7
    baseline = candidate.copy()
    baseline[-1] = 9
    controls = {name: baseline.copy() for name in ("a", "b", "c")}
    assert protection(candidate, baseline)["passed"]
    assert timing_screen(candidate, controls, True)["passed"]
    assert not timing_screen(candidate, controls, False)["passed"]
    controls["c"][-1] = 6.9
    assert not timing_screen(candidate, controls, True)["passed"]
    controls["c"][-1] = 8
    controls["c"][1] = 61.1
    assert not timing_screen(candidate, controls, True)["passed"]
