"""Planning and interim resource decisions, distinct from new scientific evidence."""

import numpy as np
import pytest

from cafl4ds.target_precision_budget import PrecisionBudgetPlanner, planned_seeds, query_precision_gate


def test_constant_effect_projection_is_exact_and_selects_smallest_budget():
    labels = np.tile([0, 1], 20)
    domains = np.repeat([0, 1], 20)
    response = np.full((3, 12, 40), 2.0)
    result = PrecisionBudgetPlanner([24, 48], repetitions=80).project(
        response, labels, domains, {f"{d}/{c}": 100 for d in (0, 1) for c in (0, 1)}
    )
    assert result["selected_panels"] == 24
    for r in result["candidates"]:
        np.testing.assert_allclose(r["median_crossed_half_width_pp"], 0, atol=1e-12)


def test_more_panels_reduce_support_monte_carlo_uncertainty():
    labels = np.tile([0, 1], 20)
    domains = np.repeat([0, 1], 20)
    response = np.repeat(np.linspace(-5, 5, 12)[None, :, None], 40, axis=2)
    result = PrecisionBudgetPlanner([24, 384], repetitions=500).project(
        response, labels, domains, {f"{d}/{c}": 100 for d in (0, 1) for c in (0, 1)}
    )
    a, b = result["candidates"]
    assert max(b["median_support_half_width_pp"]) < min(a["median_support_half_width_pp"]) / 2


def test_seed_prefix_is_stable_and_unique():
    original = [61001, 61007]
    result = planned_seeds(original, 256)
    assert result[:2] == original and len(set(result)) == 256
    with pytest.raises(ValueError):
        planned_seeds([62001], 3)


def test_query_only_interim_gate_does_not_use_effect_direction():
    rows = [{"fresh": {"query": {"half_width": [0, 0.8, 0.9]}, "mean_gain_pp": [-50, -50, -50]}}] * 18
    assert query_precision_gate(rows)["continue_to_final_budget"]
    rows[0] = {"fresh": {"query": {"half_width": [0, 5, 5]}}}
    assert query_precision_gate(rows)["continue_to_final_budget"]
    assert not query_precision_gate(rows[:1])["continue_to_final_budget"]
