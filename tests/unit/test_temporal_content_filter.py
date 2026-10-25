"""Selection causality, duplicate coverage, frame budgets and private RNG invariants."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.temporal_content_filter import content_descriptor, coverage_indices, select_window


def test_coverage_selects_distinct_structure_before_duplicates() -> None:
    """Repeated frames should not crowd out distinct observed structures."""
    generator = torch.Generator().manual_seed(13)
    unique = torch.rand((4, 3, 64, 64), generator=generator)
    images = unique.repeat_interleave(16, dim=0)
    chosen = coverage_indices(images, retain=4)
    assert chosen.tolist() == [0, 16, 32, 48]


def test_flat_frames_have_finite_descriptors_and_stable_ties() -> None:
    """Zero contrast cannot create NaNs or select a duplicate index."""
    frames = torch.zeros(64, 3, 64, 64)
    assert torch.isfinite(content_descriptor(frames)).all()
    assert coverage_indices(frames).tolist() == list(range(32))


@pytest.mark.parametrize("policy", ["full", "fixed_half", "random_half", "coverage_half"])
def test_policy_has_exact_sorted_budget_and_preserves_learner_rng(policy: str) -> None:
    """Selection uses exactly the prescribed work and leaves learner RNG untouched."""
    frames = torch.rand((64, 3, 64, 64), generator=torch.Generator().manual_seed(17))
    before = torch.get_rng_state().clone()
    chosen = select_window(policy, frames, torch.Generator().manual_seed(19))
    assert torch.equal(torch.get_rng_state(), before)
    assert len(chosen) == (64 if policy == "full" else 32)
    assert chosen.tolist() == sorted(set(chosen.tolist()))
    assert int(chosen.min()) >= 0 and int(chosen.max()) < 64


def test_selection_cannot_see_future_window_frames() -> None:
    """Appending or changing future frames has no effect on the observed-window choice."""
    stream = torch.rand((128, 3, 64, 64), generator=torch.Generator().manual_seed(21))
    first = select_window("coverage_half", stream[:64], torch.Generator().manual_seed(23))
    stream[64:] = 1
    second = select_window("coverage_half", stream[:64], torch.Generator().manual_seed(23))
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="exactly 64"):
        select_window("coverage_half", stream, torch.Generator())
