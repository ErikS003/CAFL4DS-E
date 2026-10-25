"""Verify scientific aggregation and causal paired diets before new outcomes."""

import numpy as np
import pytest
import torch

from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.historical_reservoir import HistoricalReservoir
from cafl4ds.filters.window_replay import WindowReplay
from cafl4ds.long_memory_replication import conditional_interval, endpoint_screen, path_summary, reader_assignments


def test_warm_shortfall_does_not_reward_a_lower_peak():
    points = [0, 1, 2]
    a = path_summary(np.array([[50], [60], [55]]), points)
    b = path_summary(np.array([[50], [51], [49]]), points)
    assert a["max_drawdown"].item() > b["max_drawdown"].item()
    assert a["mean_warm_shortfall"].item() == 0
    assert b["mean_warm_shortfall"].item() == 0.25


def test_source_weighting_and_domain_harm_gate():
    base = np.full((3, 3, 4, 3), 50.0)
    improved = base + 2
    result = endpoint_screen(base, base + 0.5, improved, np.full((4, 3), 49))
    assert result["point_screen_passed"]
    improved[1, :, :, 1] = 48.9
    assert not endpoint_screen(base, base, improved, base[0, 0])["gates"]["each_source_domain_budget"]


def test_consistent_seed_requirement_is_not_pooled_away():
    base = np.full((3, 3, 4, 3), 50.0)
    effects = np.full_like(base, 2)
    effects[0, :2] = -0.1
    effects[0, 2] = 9
    result = endpoint_screen(base, base - 1, base + effects, base[0, 0] - 1)
    assert result["overall_vs_current"][0] > 1
    assert not result["point_screen_passed"]


def test_paired_replay_current_subsets_and_causality():
    fifo, reservoir = WindowReplay(256, 16, 73919), HistoricalReservoir()
    global_rng = torch.get_rng_state().clone()
    for step in range(40):
        images = torch.arange(step * 32, (step + 1) * 32).view(32, 1, 1, 1).expand(-1, 2, 2, 3)
        a, b = fifo.observe(images, FilterContext(None, step)), reservoir.observe(images, FilterContext(None, step))
        assert fifo.last["current_ids"] == reservoir.last["current_ids"]
        assert max(fifo.last["replay_ids"] + reservoir.last["replay_ids"], default=-1) < 32 * step
        assert a.shape == b.shape == images.shape
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert set(fifo.marker()["ids"]) != set(reservoir.marker()["ids"])


def test_identical_actions_have_exact_zero_conditional_range():
    counts = np.array([[2, 0], [1, 1], [0, 2]])
    queries = np.ones((3, 4), dtype=int)
    result = conditional_interval(np.zeros((2, 4)), [0, 1, 0, 1], [0, 0, 1, 1], counts, queries)
    np.testing.assert_array_equal(result["lower"], 0)
    np.testing.assert_array_equal(result["upper"], 0)


def test_conditional_resampling_preserves_constant_effect_and_class_weights():
    counts = np.array([[2, 0], [1, 1], [0, 2]])
    queries = np.ones((3, 6), dtype=int)
    # Unequal class sizes must not change the equal-class/day/night target.
    difference = np.tile([0.02, 0.02, 0, -0.02, -0.02, 0], (2, 1))
    result = conditional_interval(difference, [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1], counts, queries)
    np.testing.assert_allclose(result["lower"], [0, 1, -1], atol=1e-12)
    np.testing.assert_allclose(result["upper"], [0, 1, -1], atol=1e-12)


def test_path_support_axis_and_trapezoid_are_preserved():
    values = np.array([[[50.0, 60.0], [48.0, 60.0], [46.0, 58.0]], [[55.0, 55.0], [55.0, 57.0], [54.0, 56.0]]])
    measured = path_summary(values, [0, 1, 3])
    np.testing.assert_allclose(measured["mean_warm_shortfall"][0], [7 / 3, 2 / 3])
    assert measured["max_drawdown"].shape == (2, 2)


def test_endpoint_reader_work_is_balanced_across_processes():
    jobs = [{"id": str(i), "draws": list(range(256 if i % 32 == 1 else 3))} for i in range(864)]
    jobs.insert(0, {"id": "warm", "draws": list(range(256))})
    assigned = reader_assignments(jobs, 4)
    loads = [sum(len(j["draws"]) for j in worker) for worker in assigned]
    assert max(loads) - min(loads) <= 3
    assert [sum(len(j["draws"]) == 256 for j in worker) for worker in assigned] == [7, 7, 7, 7]
    assert sorted(j["id"] for worker in assigned for j in worker) == sorted(j["id"] for j in jobs)


@pytest.mark.parametrize("points", [[0, 0, 1], [0, 2, 1]])
def test_invalid_path_grid_rejected(points):
    with pytest.raises(ValueError):
        path_summary(np.zeros((3, 3)), points)
