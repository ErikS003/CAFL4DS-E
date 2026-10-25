"""Check the user's 8/24 to 16/16 to base-24/8 recovery and matched controls."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.reservoir_health_pulse import PulseReservoir
from cafl4ds.filters.reservoir_health_recovery import RecoveryReservoir, ReservoirHealthRecovery, shifted_pulses
from scripts.run_paired_update_response import fingerprint


def frames(step):
    """Expose incoming identities in small deterministic pixels."""
    return torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1)


def test_stage_boundaries_and_latched_return_to_base():
    """Keep each recovery stage for exactly 32 future arrivals, including while latched."""
    control = ReservoirHealthRecovery()
    assert control.split(148) == (24, 8)
    control.last_alarm, control.latched = 148, True
    with pytest.raises(ValueError, match="after"):
        control.split(148)
    assert control.split(149) == control.split(180) == (8, 24)
    assert control.split(181) == control.split(212) == (16, 16)
    assert control.split(213) == (24, 8)
    assert control.latched


def test_fresh_alarm_restarts_middle_stage():
    """A fresh episode resets the heavy stage instead of finishing an older recovery."""
    control = ReservoirHealthRecovery()
    control.last_alarm = 148
    assert control.split(196) == (16, 16)
    control.last_alarm = 196
    assert control.split(197) == control.split(228) == (8, 24)
    assert control.split(229) == control.split(260) == (16, 16)
    assert control.split(261) == (24, 8)


def test_existing_memory_recipe_unchanged_at_shared_doses():
    """New allocation support preserves prior reservoir pixels and randomness."""
    prior, current = PulseReservoir(), RecoveryReservoir()
    before = torch.get_rng_state().clone()
    for step in range(30):
        old = 16 if step % 2 else 24
        ctx = SimpleNamespace(step=step)
        assert torch.equal(prior.observe(frames(step), ctx, old), current.observe(frames(step), ctx, old))
        assert prior.last == current.last
    assert fingerprint(prior.state_dict()) == fingerprint(current.state_dict())
    assert torch.equal(before, torch.get_rng_state())


def test_all_three_mixes_have_identical_memory_rng_and_nested_samples():
    """A 24/8 baseline and both recovery stages differ only in nested trained subsets."""
    memories = [RecoveryReservoir() for _ in range(3)]
    for step in range(50):
        for memory, old in zip(memories, (8, 16, 24), strict=True):
            mixed = memory.observe(frames(step), SimpleNamespace(step=step), old)
            assert len(mixed) == 32
            assert all(i < step * 32 for i in memory.last["replay_ids"])
        a, b, c = memories
        assert set(c.last["current_ids"]) <= set(b.last["current_ids"]) <= set(a.last["current_ids"])
        assert c.last["replay_ids"][:8] == b.last["replay_ids"][:8] == a.last["replay_ids"]
        assert fingerprint(a.state_dict()) == fingerprint(b.state_dict()) == fingerprint(c.state_dict())


def test_shifts_preserve_both_stages_and_truncated_terminal_dose():
    """Preserve startup and exact counts at each split, rather than only total replay."""
    values = np.full(4096, 24)
    values[0] = 32
    values[148:180] = 8
    values[180:212] = 16
    values[4080:] = 8
    for offset, actual in zip((256, 512, 1024), shifted_pulses(values).values(), strict=True):
        np.testing.assert_array_equal(actual[:132], values[:132])
        np.testing.assert_array_equal(actual[132:], np.roll(values[132:], offset))
        assert sorted(actual) == sorted(values.tolist())
        assert actual.count(8) == 48 and actual.count(16) == 32


def test_state_roundtrip_preserves_both_recovery_stages():
    """Restore complete memory and controller states before resuming future allocations."""
    memory = RecoveryReservoir()
    for step in range(12):
        memory.observe(frames(step), SimpleNamespace(step=step), 8)
    other = RecoveryReservoir()
    other.load_state_dict(memory.state_dict())
    for step, old in zip(range(12, 15), (24, 16, 8), strict=True):
        ctx = SimpleNamespace(step=step)
        assert torch.equal(memory.observe(frames(step), ctx, old), other.observe(frames(step), ctx, old))
    control = ReservoirHealthRecovery()
    control.last_alarm, control.latched = 148, True
    restored = ReservoirHealthRecovery()
    restored.load_state_dict(control.state_dict())
    for step in (180, 181, 212, 213):
        assert restored.split(step) == control.split(step)
    assert fingerprint(restored.state_dict()) == fingerprint(control.state_dict())


def test_invalid_dose_rejected_before_memory_mutation():
    """Undeclared replay allocations cannot alter retained history."""
    memory = RecoveryReservoir()
    before = fingerprint(memory.marker())
    with pytest.raises(ValueError, match="8, 16 or 24"):
        memory.observe(frames(0), SimpleNamespace(step=0), 0)
    assert fingerprint(memory.marker()) == before


def test_real_geometry_alarm_rearming_does_not_end_recovery_early():
    """Exercise the actual episode trigger with causal geometry observations."""
    reference = np.random.default_rng(17).normal(size=(64, 4))
    collapsed = np.repeat(reference[:1], 64, axis=0)
    control = ReservoirHealthRecovery()
    assert not control.observe(132, reference)
    assert not control.observe(140, collapsed)
    assert control.split(148) == (24, 8)
    assert control.observe(148, collapsed)
    assert control.split(149) == (8, 24)
    assert not control.observe(156, reference)
    assert not control.observe(164, reference)
    assert not control.latched
    assert control.split(180) == (8, 24)
    assert control.split(181) == (16, 16)
    assert control.split(213) == (24, 8)


def test_independent_reconstruction_matches_constant_and_staged_records():
    """Check the independent audit across a complete synthetic arrival stream."""
    from scripts.verify_reservoir_health_recovery import reconstruct  # noqa: PLC0415

    class Stream:
        def batch(self, index):
            return frames(index).numpy()

    reference = np.random.default_rng(17).normal(size=(64, 4))
    collapsed = np.repeat(reference[:1], 64, axis=0)
    records = {}
    for arm in ("constant", "health"):
        memory, control = RecoveryReservoir(), ReservoirHealthRecovery()
        control.observe(132, reference)
        raw, trace, monitors = {"132_monitor_features": reference}, [], {}
        for step in range(1, 4097):
            current, old = (24, 8) if arm == "constant" else control.split(step)
            if step == 1:
                current, old = 32, 0
            images = frames(step - 1)
            mixed = memory.observe(images, SimpleNamespace(step=step - 1), 8 if step == 1 else old)
            import hashlib  # noqa: PLC0415

            trace.append(
                {
                    "step": step,
                    "mix": [current, old],
                    "input_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                    "train_sha256": hashlib.sha256(mixed.numpy().tobytes()).hexdigest(),
                    "diet": dict(memory.last),
                    "response": [1, 2],
                }
            )
            if step > 132 and control.scheduled(step):
                features = collapsed if step in (140, 148, 220, 228) else reference
                raw[f"{step}_monitor_features"] = features
                monitors[str(step)] = {"alarm": control.observe(step, features)}
        record = {
            "trace": trace,
            "events": control.events,
            "monitors": monitors,
            "schedule": [32] + [24] * 4095 if arm == "constant" else None,
        }
        records[arm] = record
        result = reconstruct(record, records["constant"], raw, Stream())
        assert result["last_alarm"] == 228
        assert result["ids"] == memory.state_dict()["ids"]
        assert result["summary"]["heavy_arrivals"] == (64 if arm == "health" else 0)
        assert result["summary"]["middle_arrivals"] == (64 if arm == "health" else 0)
        assert result["summary"]["replay_exposures"] == 32760 + (1536 if arm == "health" else 0)
