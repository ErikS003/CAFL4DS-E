"""Nested budgets and conditional readouts must preserve their data boundaries."""

import pytest
import torch

from cafl4ds.linear_capacity_domains import linear_outcomes, nested_supports


def test_balanced_nested_samples_preserve_rng() -> None:
    """Budgets are nested within a draw, and full pool is represented once."""
    labels = torch.arange(3).repeat_interleave(192).repeat(2)
    domains = torch.arange(2).repeat_interleave(576)
    rng = torch.get_rng_state().clone()
    samples = nested_supports(labels, domains, torch.arange(1152))
    assert len(samples) == 13 and torch.equal(rng, torch.get_rng_state())
    for seed in (80011, 80017, 80023):
        previous = torch.empty(0, dtype=torch.long)
        for size in (16, 32, 64, 128):
            support = samples[f"{size}_{seed}"]
            assert len(support) == size * 6 and torch.isin(previous, support).all()
            assert all(
                int(((labels[support] == c) & (domains[support] == d)).sum()) == size
                for c in range(3)
                for d in range(2)
            )
            previous = support


def test_conditional_linear_predictions_isolate_domains() -> None:
    """Changing night representations cannot alter day conditional accuracy or CE."""
    generator = torch.Generator().manual_seed(801)
    labels = torch.arange(3).repeat(24)
    domains = torch.arange(2).repeat_interleave(36)
    z = torch.randn(72, 6, generator=generator) + torch.nn.functional.one_hot(labels, 6) * 3
    support = torch.cat([torch.arange(30), torch.arange(36, 66)])
    queries = {"test": torch.cat([torch.arange(30, 36), torch.arange(66, 72)])}
    cs = {"0": 0.1, "1": 0.1}
    first = linear_outcomes(z, labels, domains, support, queries, "conditional", cs)
    altered = z.clone()
    altered[domains == 1] = torch.randn(36, 6, generator=generator) * 10
    second = linear_outcomes(altered, labels, domains, support, queries, "conditional", cs)
    assert first["test"]["linear_error_A"] == second["test"]["linear_error_A"]
    assert first["test"]["refit_ce_A"] == second["test"]["refit_ce_A"]
    with pytest.raises(ValueError, match="overlap"):
        linear_outcomes(z, labels, domains, support, {"test": support[:6]}, "shared", {"shared": 0.1})
