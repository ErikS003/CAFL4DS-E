"""Validate causal memories, exact replay dose and isolated context scoring."""

import numpy as np
import pytest

from cafl4ds.adaptive_mixture import Mixture, context_distance, memory_plan


def test_memory_snapshots_precede_each_quota():
    """Precomputed indices never expose future image content."""
    plans = memory_plan("test", 1)
    for block, values in enumerate(plans):
        assert len(values) == len(set(values)) == 64
        assert max(values) < (8 + block * 8) * 64


def test_all_mixtures_share_old_identity_dose_and_order():
    """Adaptive timing cannot claim a gain by using more replay images."""
    plans = memory_plan("test", 1)
    features = np.ones((12288, 4))
    for arm in ("fixed", "random", "adaptive"):
        selector = Mixture(arm, plans, 1)
        records = [selector.choose(step, features) for step in range(192)]
        assert sum(len(r["old"]) for r in records) == 1472
        assert sum(len(r["current"]) for r in records) == 4672
        for block, expected in enumerate(plans):
            actual = [index for r in records[8 + block * 8 : 16 + block * 8] for index in r["old"]]
            assert actual == expected


def test_context_detects_change_without_labels():
    """Identical context stays quiet while a rotated context differs."""
    z = np.tile([1.0, 0], (128, 1))
    assert context_distance(z, list(range(64, 128)), list(range(32))) == 0
    z[64:] = [0, 1]
    assert context_distance(z, list(range(64, 128)), list(range(32))) == 1


def test_context_rejects_future_training_identity():
    """Recent training content must precede the current arrivals."""
    with pytest.raises(ValueError, match="strictly past"):
        context_distance(np.ones((128, 2)), list(range(64, 128)), [80])


def test_future_feature_changes_do_not_change_present_choice():
    """An offline reference cache cannot leak later arrivals into scoring."""
    plans = memory_plan("test", 2)
    z = np.random.default_rng(3).normal(size=(12288, 4))
    altered = z.copy()
    altered[640:] = 1000
    a, b = Mixture("adaptive", plans, 2), Mixture("adaptive", plans, 2)
    assert [a.choose(j, z) for j in range(10)] == [b.choose(j, altered) for j in range(10)]
