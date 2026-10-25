"""Verify historical retention, matched current samples and independent replay state."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.historical_reservoir import HistoricalReservoir
from cafl4ds.filters.window_replay import WindowReplay
from scripts.run_paired_update_response import fingerprint


def arrival(step):
    """Encode identity in each test image so history is directly observable."""
    return torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1)


def test_matched_current_samples_and_global_rng_with_different_retention():
    """Replacement never perturbs current sampling or learner RNG, even across long histories."""
    memory, fifo = HistoricalReservoir(), WindowReplay(capacity=256)
    other = HistoricalReservoir(admission_seed=90103)
    rng = torch.get_rng_state().clone()
    for step in range(100):
        images, ctx = arrival(step), SimpleNamespace(step=step)
        mixed = memory.observe(images, ctx)
        recent = fifo.observe(images, ctx)
        other.observe(images, ctx)
        assert mixed.shape == recent.shape == images.shape
        assert memory.last["current_ids"] == fifo.last["current_ids"] == other.last["current_ids"]
        assert all(i < step * 32 for i in memory.last["replay_ids"])
        if step < 9:
            assert torch.equal(mixed, recent)
    assert memory.marker()["ids"] != other.marker()["ids"]
    assert min(memory.marker()["ids"]) < min(fifo.marker()["ids"])
    assert torch.equal(rng, torch.get_rng_state())


def test_exact_restore_and_independent_memory():
    """Saved generators and pixels reproduce future samples; restoration does not alias state."""
    original = HistoricalReservoir()
    for step in range(12):
        original.observe(arrival(step), SimpleNamespace(step=step))
    saved = original.state_dict()
    restored = HistoricalReservoir()
    restored.load_state_dict(saved)
    assert fingerprint(original.marker()) == fingerprint(restored.marker())
    for step in range(12, 16):
        a = original.observe(arrival(step), SimpleNamespace(step=step))
        b = restored.observe(arrival(step), SimpleNamespace(step=step))
        assert torch.equal(a, b) and original.last == restored.last
    assert saved["seen"] == 384


def test_restore_rejects_corrupted_pixels_and_duplicate_identities():
    """Corrupt saved memory cannot silently change the scientific image diet."""
    memory = HistoricalReservoir()
    memory.observe(arrival(0), SimpleNamespace(step=0))
    state = memory.state_dict()
    bad = copy.deepcopy(state)
    bad["images"][0] += 1
    with pytest.raises(ValueError, match="pixels"):
        HistoricalReservoir().load_state_dict(bad)
    bad = copy.deepcopy(state)
    bad["ids"][1] = bad["ids"][0]
    with pytest.raises(ValueError, match="identities"):
        HistoricalReservoir().load_state_dict(bad)


def test_algorithm_r_has_uniform_marginal_inclusion_without_recent_bias():
    """Across independent admission seeds, early and late frames meet the same inclusion law."""
    counts = np.zeros(96)
    for seed in range(512):
        memory = HistoricalReservoir(capacity=32, admission_seed=seed)
        for step in range(3):
            memory.observe(arrival(step), SimpleNamespace(step=step))
        counts[memory.marker()["ids"]] += 1
    expected = 512 / 3
    sigma = np.sqrt(512 * (1 / 3) * (2 / 3))
    assert np.max(np.abs(counts - expected)) < 5 * sigma
    assert counts.sum() == 512 * 32


def test_bad_capacity_or_nonconsecutive_arrival_rejected_before_mutation():
    """Identity clocks and optimizer batch shape remain explicit invariants."""
    with pytest.raises(ValueError, match="Capacity"):
        HistoricalReservoir(capacity=255)
    memory = HistoricalReservoir()
    before = fingerprint(memory.marker())
    with pytest.raises(ValueError, match="consecutive"):
        memory.observe(arrival(1), SimpleNamespace(step=1))
    assert before == fingerprint(memory.marker())


def test_screen_accepts_bounded_tradeoff_and_rejects_each_exceeded_budget():
    """Protection may trade a bounded endpoint loss, with each domain protected separately."""
    from scripts.verify_memory_age_replay import screen  # noqa: PLC0415

    current, fifo, reservoir = np.zeros(13), np.zeros(13), np.zeros(13)
    current[:3], fifo[:3], reservoir[:3] = [55, 60.8, 49.2], [54, 58, 50], [55, 60, 50]
    current[-1], fifo[-1], reservoir[-1] = 11, 8, 6.5
    assert screen(reservoir, fifo, current)["passed"]
    for domain in (1, 2):
        bad = reservoir.copy()
        bad[domain] = current[domain] - 1.01
        assert not screen(bad, fifo, current)["passed"]
    bad = reservoir.copy()
    bad[-1] = 7.01
    assert not screen(bad, fifo, current)["passed"]
