"""Check causal, brightness-sensitive selection and paired-control compatibility."""

import pytest
import torch

from cafl4ds.temporal_anchored_filter import change_indices, select_window
from cafl4ds.temporal_content_filter import select_window as old_select


def test_flat_ties_keep_earliest_candidate_and_latest_anchor() -> None:
    """A constant window has deterministic, bounded temporal gaps."""
    selected = change_indices(torch.zeros(64, 3, 64, 64))
    expected = [index for start in range(0, 64, 4) for index in (start, start + 3)]
    assert selected.tolist() == expected
    assert 63 in selected
    assert int(torch.diff(selected).max()) <= 4


def test_brightness_change_survives_pooling_without_normalization() -> None:
    """A flat exposure step is informative to the raw score despite zero texture."""
    images = torch.zeros(64, 3, 64, 64)
    images[2:] = .1
    selected = change_indices(images)
    assert selected[:2].tolist() == [2, 3]


def test_previous_frame_at_block_boundary_is_observed() -> None:
    """First candidate in later blocks scores against the preceding observed frame."""
    images = torch.zeros(64, 3, 64, 64)
    images[4:] = .8
    assert change_indices(images)[2:4].tolist() == [4, 7]


@pytest.mark.parametrize("policy", ["full", "fixed_half", "uniform_half", "random_half", "anchored_half"])
def test_budgets_order_and_private_rng(policy: str) -> None:
    """Every policy respects the declared acceptance and leaves learner RNG intact."""
    images = torch.rand(64, 3, 64, 64)
    before = torch.get_rng_state().clone()
    selected = select_window(policy, images, torch.Generator().manual_seed(42))
    assert torch.equal(before, torch.get_rng_state())
    assert len(selected) == (64 if policy == "full" else 32)
    assert bool((torch.diff(selected) > 0).all())
    if policy in {"full", "fixed_half", "random_half"}:
        assert torch.equal(selected, old_select(policy, images, torch.Generator().manual_seed(42)))
    if policy in {"uniform_half", "anchored_half"}:
        assert 63 in selected


def test_future_suffix_cannot_change_observed_window_decision() -> None:
    """No later arrivals are read or accepted as part of this window."""
    stream = torch.rand(128, 3, 64, 64)
    before = change_indices(stream[:64])
    stream[64:] = 1
    assert torch.equal(before, change_indices(stream[:64]))
    with pytest.raises(ValueError, match="expected"):
        change_indices(stream)


def test_invalid_pixels_rejected() -> None:
    """Nonfinite observations cannot silently alter rank decisions."""
    images = torch.zeros(64, 3, 64, 64)
    images[10, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        change_indices(images)
