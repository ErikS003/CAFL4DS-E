"""Controls must distinguish allocation from within-clip image selection."""

import numpy as np
import pytest

from cafl4ds.single_image_filter import matched_random
from cafl4ds.single_image_long_gate import global_schedule, independent_schedule, policy_specs


def test_global_control_does_not_copy_concentrated_clip_allocation():
    accepted = np.r_[np.ones(96, bool), np.zeros(96 * 7, bool)]
    result = global_schedule(accepted, 44)
    assert result.sum() == accepted.sum()
    assert result.reshape(8, 96).sum(1)[1:].sum() > 0
    clip = matched_random(accepted, np.arange(len(accepted)), 96, 44)
    np.testing.assert_array_equal(clip.reshape(8, 96).sum(1), accepted.reshape(8, 96).sum(1))


def test_global_control_depends_only_on_total_count():
    first = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    second = first[::-1]
    np.testing.assert_array_equal(global_schedule(first, 17), global_schedule(second, 17))


def test_independent_random_is_reproducible_and_prefix_stable():
    short = independent_schedule(96, 0.25, 31)
    longer = independent_schedule(2304, 0.25, 31)
    np.testing.assert_array_equal(short, longer[:96])
    assert 0.20 < longer.mean() < 0.30


def test_relaxation_retains_improvement_requirement_and_labels_controls():
    specs = policy_specs(dict(threshold_factors=[1, 0.5, 0], independent_probabilities=[0.1, 0.25, 0.5]), -0.02)
    assert specs["gate_original"]["threshold"] == -0.02
    assert specs["gate_half"]["threshold"] == -0.01
    assert specs["gate_zero"]["threshold"] == 0
    for gate in ("gate_original", "gate_half", "gate_zero"):
        assert specs[f"clip_{gate}"]["retrospective"]
        assert specs[f"global_{gate}"]["retrospective"]
    assert len(specs) == 14
    with pytest.raises(ValueError):
        policy_specs(dict(threshold_factors=[1, 0.5, 0], independent_probabilities=[0.1]), 0.1)
