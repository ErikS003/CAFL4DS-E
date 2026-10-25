"""Verify replay-selection isolation and honest cross-draw action diagnostics."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.replay_selection import (
    FixedReplaySelection,
    SustainedReplayCoverage,
    coverage_selection,
    cross_draw_opportunities,
    ranked_blocks,
)
from scripts.run_replay_selection import freeze_health_gates


def test_replacement_keeps_rng_memory_current_and_future_draws() -> None:
    """A different first replay set cannot alter ingestion or later uniform selection."""
    baseline = ReplayRatio(16, seed=19)
    for i in range(300):
        baseline.reservoir._ingest(torch.full((3, 2, 2), float(i)))
    memory: dict[str, Any] = {
        "buffer": copy.deepcopy(baseline.reservoir._buffer),
        "seen": baseline.reservoir._seen,
        "rng": baseline.reservoir._gen.get_state().clone(),
    }
    targeted = FixedReplaySelection(list(range(240, 256)), 10, 19)
    targeted.restore_memory(memory)
    replacement = torch.stack(memory["buffer"][240:256])
    rng = torch.get_rng_state().clone()
    for step in (10, 11):
        batch = StreamBatch(torch.full((32, 3, 2, 2), float(step)), era=0, step=step)
        context = cast(FilterContext, SimpleNamespace(step=step))
        a, b = baseline.select(batch, context), targeted.select(batch, context)
        assert torch.equal(a[:16], b[:16])
        assert torch.equal(torch.stack(baseline.reservoir._buffer), torch.stack(targeted.reservoir._buffer))
        assert torch.equal(baseline.reservoir._gen.get_state(), targeted.reservoir._gen.get_state())
        if step == 10:
            assert torch.equal(b[16:], replacement)
        else:
            assert torch.equal(a, b)
    assert torch.equal(rng, torch.get_rng_state())


def test_coverage_avoids_current_duplicates_and_returns_unique_indices() -> None:
    """The farthest feature is picked first; tied/redundant features remain bounded."""
    z = torch.tensor([[1.0, 0], [0, 1], [-1, 0], [1, 0]])
    chosen = coverage_selection(z, z[:1], 3)
    assert chosen[0] == 2
    assert len(set(chosen)) == 3
    assert coverage_selection(z, z[:1], 4) == [2, 1, 0, 3]


def test_sustained_coverage_preserves_memory_and_recomputes_each_step() -> None:
    """Repeated geometry selection cannot bias ingestion, current exposure or RNG."""
    baseline = ReplayRatio(16, seed=19)
    for i in range(300):
        baseline.reservoir._ingest(torch.full((3, 2, 2), float(i)))
    memory: dict[str, Any] = {
        "buffer": copy.deepcopy(baseline.reservoir._buffer),
        "seen": baseline.reservoir._seen,
        "rng": baseline.reservoir._gen.get_state().clone(),
    }
    calls = []

    def observe(images: torch.Tensor) -> torch.Tensor:
        calls.append(images.clone())
        value = images[:, 0, 0, 0]
        return torch.stack((torch.cos(value / 20), torch.sin(value / 20)), dim=1)

    coverage = SustainedReplayCoverage(19, observe)
    coverage.restore_memory(memory)
    rng = torch.get_rng_state().clone()
    for step in range(10, 18):
        batch = StreamBatch(torch.full((32, 3, 2, 2), float(step)), era=0, step=step)
        ctx = cast(FilterContext, SimpleNamespace(step=step))
        a, b = baseline.select(batch, ctx), coverage.select(batch, ctx)
        assert torch.equal(a[:16], b[:16])
        assert coverage.last["candidate_indices"][:16] == baseline.last["replay_indices"]
        assert len(set(coverage.last["replay_indices"])) == 16
        assert torch.equal(torch.stack(baseline.reservoir._buffer), torch.stack(coverage.reservoir._buffer))
        assert torch.equal(baseline.reservoir._gen.get_state(), coverage.reservoir._gen.get_state())
        assert coverage.last["current_indices"] == baseline.last["current_indices"]
    assert len(calls) == 8
    assert not torch.equal(calls[0], calls[-1])
    assert torch.equal(rng, torch.get_rng_state())


def test_ranked_mixture_retains_uniform_exposure_without_duplicates() -> None:
    """A highly ranked baseline block cannot duplicate the protected uniform half."""
    selected = ranked_blocks(torch.tensor([9.0, 1, 8, 2, 3, 4, 5, 6]), list(range(16)))
    assert selected["task_gradient"] == list(range(8)) + list(range(16, 24))
    assert selected["mixed"] == list(range(8)) + list(range(16, 24))


def test_opportunity_does_not_choose_on_the_scored_draw() -> None:
    """A hindsight advantage that reverses across draws must not look beneficial."""
    values = np.array([[0.5, 0.52, 0.5, 0.5], [0.5, 0.48, 0.5, 0.5]])
    result = cross_draw_opportunities(values)
    assert result["folds"][0]["selected_rule"] == "uniform"
    assert result["folds"][1]["selected_rule"] == "coverage"
    assert result["mean_gain"] == pytest.approx(-0.01)
    with pytest.raises(ValueError):
        cross_draw_opportunities(values[:1])


def test_health_gate_excludes_heldout_pool_and_evaluation_labels() -> None:
    """Poisoning heldout-pool targets cannot change that pool's frozen gate."""
    features: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    names = (
        "selected_reconstruction",
        "virtual_ssl_loss",
        "selected_coverage",
        "selected_variance",
        "ssl_gradient_norm",
        "current_ssl_cosine",
        "health_rankme",
        "health_variance",
        "health_cka_drift",
        "health_reconstruction",
    )
    for pool in range(3):
        state = f"state{pool}"
        for rule in ("uniform", "coverage"):
            features.append(
                {
                    "state": state,
                    "pool_group": f"pool{pool}",
                    "rule": rule,
                    "features": {k: float(pool + (rule == "coverage")) for k in names},
                }
            )
            for role, draw in (("development", 0), ("holdout", 0), ("development", 1)):
                scores.append(
                    {
                        "state": state,
                        "pool_group": f"pool{pool}",
                        "rule": rule,
                        "draw": draw,
                        "horizon": 1,
                        "panel": "pooled",
                        "role": role,
                        "tolerance": 1e-10,
                        "accuracy_balanced": 0.5 + 0.01 * (rule == "coverage"),
                    }
                )
    original = freeze_health_gates(features, scores)
    poisoned = copy.deepcopy(scores)
    for row in poisoned:
        if row["pool_group"] == "pool0" or row["role"] == "holdout" or row["draw"] > 0:
            row["accuracy_balanced"] = 0.01
    changed = freeze_health_gates(features, poisoned)
    assert [c for c in original["choices"] if c["heldout_pool"] == "pool0"] == [
        c for c in changed["choices"] if c["heldout_pool"] == "pool0"
    ]
    for model in original["models"]:
        assert f"state{model['heldout_pool'][-1]}" not in model["training_states"]
    # Opposite delayed outcomes must reverse the gate when horizon4 is requested.
    delayed = copy.deepcopy(scores)
    for row in delayed:
        row["horizon"] = 4
        row["accuracy_balanced"] = 0.5 - 0.01 * (row["rule"] == "coverage")
    reversed_gate = freeze_health_gates(features, scores + delayed, target_horizon=4)
    assert reversed_gate["target_horizon"] == 4
    assert all(c["rule"] == "uniform" for c in reversed_gate["choices"])
    assert all(c["rule"] == "coverage" for c in original["choices"])
