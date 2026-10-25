"""Check matched group interventions and strict cross-pool target isolation."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.replay_selection import CoveragePulse
from cafl4ds.training_choice import CHOICES, FIELDS, FixedTrainingChoice, choices, freeze_predictors


def test_group_replacement_keeps24_slots_and_future_memory_draws() -> None:
    """Both admission and replay substitutions retain equal exposure and future RNG."""
    baseline = ReplayRatio(16, seed=19)
    for i in range(300):
        baseline.reservoir._ingest(torch.full((3, 2, 2), float(i)))
    memory: dict[str, Any] = {
        "buffer": copy.deepcopy(baseline.reservoir._buffer),
        "seen": baseline.reservoir._seen,
        "rng": baseline.reservoir._gen.get_state().clone(),
    }
    gen = torch.Generator()
    gen.set_state(memory["rng"])
    candidates = torch.randperm(256, generator=gen)[:64].tolist()
    current = torch.randperm(32, generator=torch.Generator().manual_seed(19 + 104729 * 11))[:16].sort().values.tolist()
    plans = choices(current, candidates)
    for rule in CHOICES[1:]:
        ordinary = ReplayRatio(16, seed=19)
        ordinary.restore_memory(memory)
        targeted = FixedTrainingChoice(plans[rule], 10, 19)
        targeted.restore_memory(memory)
        for step in (10, 11, 12):
            batch = StreamBatch(
                torch.arange(32.0).reshape(32, 1, 1, 1).expand(32, 3, 2, 2) + 1000 * step, era=0, step=step
            )
            context = cast(FilterContext, SimpleNamespace(step=step))
            a, b = ordinary.select(batch, context), targeted.select(batch, context)
            if step == 10:
                assert int((a != b).flatten(1).any(1).sum()) == 8
            else:
                assert torch.equal(a, b)
            assert torch.equal(torch.stack(ordinary.reservoir._buffer), torch.stack(targeted.reservoir._buffer))
            assert torch.equal(ordinary.reservoir._gen.get_state(), targeted.reservoir._gen.get_state())


def test_frozen_predictors_ignore_held_pool_and_evaluation_outcomes() -> None:
    """Poisoning excluded pools/roles/draws cannot change the held pool's decisions."""
    features, scores = [], []
    names = sorted({k for fields in FIELDS.values() for k in fields})
    for pool in range(3):
        for rule in CHOICES:
            features.append(
                {
                    "state": f"state{pool}",
                    "pool_group": f"pool{pool}",
                    "rule": rule,
                    "features": {k: float(pool + CHOICES.index(rule)) for k in names},
                }
            )
            for role, draw, horizon in (
                ("development", 0, 4),
                ("holdout", 0, 4),
                ("development", 1, 4),
                ("development", 0, 1),
            ):
                scores.append(
                    {
                        "state": f"state{pool}",
                        "pool_group": f"pool{pool}",
                        "rule": rule,
                        "draw": draw,
                        "role": role,
                        "horizon": horizon,
                        "panel": "pooled",
                        "tolerance": 1e-10,
                        "accuracy_balanced": 0.5 + 0.01 * (rule == "replay_a"),
                    }
                )
    original = freeze_predictors(features, scores)
    poisoned = copy.deepcopy(scores)
    for row in poisoned:
        if row["pool_group"] == "pool0" or row["role"] != "development" or row["draw"] != 0 or row["horizon"] != 4:
            row["accuracy_balanced"] = 0.01
    changed = freeze_predictors(features, poisoned)
    assert [c for c in original["choices"] if c["pool_group"] == "pool0"] == [
        c for c in changed["choices"] if c["pool_group"] == "pool0"
    ]
    assert all("state" + model["heldout_pool"][-1] not in model["training_states"] for model in original["models"])
    assert all(c["rule"] == "replay_a" for c in original["choices"] if c["recipe"] == "constant")
    # With no state-dependent utility, residual models reduce to the same action prior.
    assert all(c["rule"] == "replay_a" for c in original["choices"])


def test_pulse_has_four_observations_and_two_common_uniform_recovery_steps() -> None:
    """A fixed pulse cannot keep targeting after its declared interval."""
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
        calls.append(1)
        v = images[:, 0, 0, 0]
        return torch.stack((torch.cos(v / 20), torch.sin(v / 20)), dim=1)

    pulse = CoveragePulse(19, observe, 10, 4)
    pulse.restore_memory(memory)
    for step in range(10, 16):
        batch = StreamBatch(torch.full((32, 3, 2, 2), float(step)), era=0, step=step)
        ctx = cast(FilterContext, SimpleNamespace(step=step))
        a, b = baseline.select(batch, ctx), pulse.select(batch, ctx)
        assert ("candidate_indices" in pulse.last) == (step < 14)
        if step >= 14:
            assert torch.equal(a, b)
        assert torch.equal(baseline.reservoir._gen.get_state(), pulse.reservoir._gen.get_state())
    assert len(calls) == 4
