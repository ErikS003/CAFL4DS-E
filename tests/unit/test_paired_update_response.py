"""Verify exact Adam/RNG restoration, branch isolation and prospective preference rules."""

from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from cafl4ds.paired_update_response import identical, restore, run_window
from scripts import report_paired_update_response as audit
from scripts.run_paired_update_response import fingerprint, state_record


class ToySSL(nn.Module):
    """Stochastic two-dimensional learner with real Adam moments."""

    def __init__(self) -> None:
        """Create the encoder."""
        super().__init__()
        self.encoder = nn.Linear(2, 2)

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """Require both learned state and RNG to reproduce losses."""
        return (self.encoder(images) - torch.rand_like(images)).square().mean()


def test_restored_continuation_and_full_branch_cannot_mutate_parent() -> None:
    """A full branch followed by half restoration must exactly match the uninterrupted half history."""
    torch.manual_seed(123)
    initial = ToySSL()
    model = copy.deepcopy(initial)
    config = OmegaConf.create({"optim": {"_target_": "torch.optim.AdamW", "lr": 0.001}})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    images, order = torch.ones(18432, 2), torch.arange(18432)
    run_window(model, optimizer, images, order, 0, 0, "half")
    parent = copy.deepcopy(state_record(model, optimizer, 36, 18))
    uninterrupted = run_window(model, optimizer, images, order, 36, 18, "half", 72)
    expected = copy.deepcopy(state_record(model, optimizer, 108, 54))
    parent_hash = fingerprint(parent)
    full_model, full_opt = restore(initial, config, parent)
    full = run_window(full_model, full_opt, images, order, 36, 18, "full")
    assert full["updates"] == 36
    audit.verify_window(full, order.tolist(), 36, "full")
    assert fingerprint(parent) == parent_hash
    half_model, half_opt = restore(initial, config, parent)
    half = run_window(half_model, half_opt, images, order, 36, 18, "half", 72)
    actual = state_record(half_model, half_opt, 108, 54)
    assert identical(expected, actual)
    assert half["losses"] == uninterrupted["losses"]
    assert fingerprint(actual) == audit.state_hash(actual)
    assert fingerprint(parent) == parent_hash
    half_model, half_opt = restore(initial, config, parent)
    short = run_window(half_model, half_opt, images, order, 36, 18, "half")
    final = state_record(half_model, half_opt, 72, 36)
    audit.verify_final(final, 36, 18)
    audit.verify_window(short, order.tolist(), 36, "half")
    assert short["arrivals"] == full["arrivals"] == 1152
    assert short["optimizer_images"] == 576
    short["ledger"][0]["current"][0] = 0
    with pytest.raises(ValueError, match="Causal"):
        audit.verify_window(short, order.tolist(), 36, "half")


def test_preference_requires_adjusted_domain_guards_and_cost_savings() -> None:
    """A favorable mean or nominal interval cannot bypass simultaneous/domain/cost gates."""
    safe = {name: audit.bands([0.001] * 5) for name in ("balanced", "A", "B")}
    assert audit.preference(safe, 0.5) == "half_efficiency_preferred"
    assert audit.preference(safe, 0.8) == "unresolved"
    unsafe = {**safe, "A": audit.bands([-0.02] * 5)}
    assert audit.preference(unsafe, 0.5) == "full_quality_required"
    uncertain = audit.bands([-0.005, 0.005, -0.005, 0.005, 0])
    assert uncertain["ci95_pp"][0] > -1
    assert uncertain["simultaneous95_pp"][0] < -1
    assert audit.preference({**safe, "B": uncertain}, 0.5) == "unresolved"
    with pytest.raises(ValueError):
        audit.bands([0] * 4)
