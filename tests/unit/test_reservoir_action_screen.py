"""Verify paired recovery stages, conservative opportunity gates and independent pixel audit."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.reservoir_health_recovery import RecoveryReservoir
from cafl4ds.reservoir_action_screen import allocation, opportunity_screen
from scripts.run_paired_update_response import fingerprint


def test_exact_intervention_and_common_tail_boundaries():
    """The primary horizon includes both intervention stages and 64 common arrivals."""
    assert allocation(1, "recovery") == allocation(32, "recovery") == (8, 24)
    assert allocation(33, "recovery") == allocation(64, "recovery") == (16, 16)
    assert allocation(65, "recovery") == allocation(128, "recovery") == (24, 8)
    assert all(allocation(h, "base") == (24, 8) for h in range(1, 129))


def test_undeclared_offsets_and_actions_fail():
    """Invalid allocations cannot silently expand the frozen action family."""
    for offset, arm in ((0, "base"), (129, "recovery"), (32, "unknown")):
        with pytest.raises(ValueError, match="Require"):
            allocation(offset, arm)
    with pytest.raises(ValueError, match="eight parents"):
        opportunity_screen(np.zeros((9, 3, 3)))


def test_material_opposite_preferences_pass_only_with_two_each():
    """A useful screening result needs repeatable cases in both action directions."""
    values = np.zeros((8, 3, 3))
    values[:2] = 1
    values[2:4] = -1
    result = opportunity_screen(values)
    assert result["passed"]
    assert result["robust_recovery_count"] == result["robust_base_count"] == 2
    values[3] = 0
    assert not opportunity_screen(values)["passed"]


def test_a_dominant_static_action_is_not_an_adaptive_opportunity():
    """Useful recovery everywhere favors a static choice, not a health selector."""
    result = opportunity_screen(np.ones((8, 3, 3)))
    assert not result["passed"] and result["robust_recovery_count"] == 8
    assert result["optimistic_balanced_headroom_over_best_static"] == 0


def test_positive_balanced_gain_cannot_hide_domain_harm():
    """A gain larger than the practical margin still fails its domain budget."""
    values = np.ones((8, 3, 3))
    values[:, :, 1] = 4
    values[:, :, 2] = -2
    result = opportunity_screen(values)
    assert result["pooled_recovery_count"] == result["robust_recovery_count"] == 0


def test_support_sign_instability_rejects_pooled_preference():
    """A positive pooled result is not a robust case when one support draw reverses."""
    values = np.ones((8, 3, 3))
    values[:, 0] = -0.1
    result = opportunity_screen(values)
    assert result["pooled_recovery_count"] == 8 and result["robust_recovery_count"] == 0


def test_single_support_domain_violation_is_visible():
    """Support averaging cannot erase a declared per-draw domain violation."""
    values = np.ones((8, 3, 3))
    values[:, 0, 2] = -1.1
    result = opportunity_screen(values)
    assert result["pooled_recovery_count"] == 8 and result["robust_recovery_count"] == 0


def test_hindsight_headroom_is_only_a_descriptive_ceiling():
    """The envelope can look positive even if no robust action case qualifies."""
    values = np.zeros((8, 3, 3))
    values[:4] = 0.2
    values[4:] = -0.2
    result = opportunity_screen(values)
    assert result["optimistic_balanced_headroom_over_best_static"] == pytest.approx(0.1)
    assert not result["passed"] and not result["controller_qualified"]


def test_restored_reservoir_forks_match_primitive_pixels_and_common_tail():
    """Independent reconstruction validates no-reseed pairing, dosage and causal eligibility."""
    from scripts.verify_reservoir_action_screen import primitive_diet  # noqa: PLC0415

    class Stream:
        def batch(self, index):
            return np.arange(index * 32, (index + 1) * 32).reshape(32, 1, 1, 1)

    stream = Stream()
    original = RecoveryReservoir()
    before = torch.get_rng_state().clone()
    for step in range(192):
        original.observe(torch.from_numpy(stream.batch(step)), SimpleNamespace(step=step), 8)
    parent = {"memory": original.state_dict()}
    records, memories = {}, {}
    for arm in ("base", "recovery"):
        memory = RecoveryReservoir()
        memory.load_state_dict(parent["memory"])
        trace = []
        for offset in range(1, 129):
            step = 192 + offset
            pixels = stream.batch(step - 1)
            current, past = allocation(offset, arm)
            mixed = memory.observe(torch.from_numpy(pixels), SimpleNamespace(step=step - 1), past)
            trace.append(
                {
                    "step": step,
                    "offset": offset,
                    "current": current,
                    "past": past,
                    "diet": dict(memory.last),
                    "input_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "train_sha256": hashlib.sha256(mixed.numpy().tobytes()).hexdigest(),
                }
            )
        result = primitive_diet(trace, stream, parent)
        saved = memory.state_dict()
        for key in ("ids", "hashes", "seen"):
            assert result[key] == saved[key]
        for key in ("sample_rng", "admission_rng"):
            assert torch.equal(result[key], saved[key])
        np.testing.assert_array_equal(result["images"], saved["images"])
        assert result["diet"]["replay_exposures"] == (1024 if arm == "base" else 1792)
        records[arm], memories[arm] = trace, memory
    assert fingerprint(memories["base"].state_dict()) == fingerprint(memories["recovery"].state_dict())
    for a, b in zip(records["base"][64:], records["recovery"][64:], strict=True):
        assert a == b
    assert torch.equal(before, torch.get_rng_state())
