"""Tests for causal anchor membership and observer isolation."""

import pytest
import torch

from cafl4ds.retention_anchor import private_loss, reservoir_indices
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


class TinyLearner(torch.nn.Module):
    """Small stochastic objective with optimizer and mode state."""

    def __init__(self) -> None:
        """Build a tiny parameterized objective."""
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """Use randomness to expose accidental observer consumption."""
        prediction = self.layer(images.mean((2, 3)))
        return (prediction - torch.rand_like(prediction)).square().mean()


def test_reservoir_only_contains_past_and_is_bounded() -> None:
    """Sampling has no future dependency, duplicates or extra capacity."""
    result = reservoir_indices(512, 64, 91)
    assert len(result) == len(set(result)) == 64
    assert min(result) >= 0 and max(result) < 512
    assert result == reservoir_indices(512, 64, 91)
    assert result != reservoir_indices(512, 64, 92)


def test_common_masks_and_complete_state_isolation() -> None:
    """Repeated readings agree without changing Adam, modes or random states."""
    model = TinyLearner()
    optimizer = torch.optim.AdamW(model.parameters())
    images = torch.rand(64, 3, 2, 2)
    optimizer.zero_grad()
    model.training_step(images[:32]).backward()
    optimizer.step()
    model.eval()
    before = fingerprint(capture(model, optimizer))
    first = private_loss(model, images, 17)
    assert first == private_loss(model, images, 17)
    assert fingerprint(capture(model, optimizer)) == before
    assert not any(module.training for module in model.modules())


def test_current_rng_loss_matches_next_training_view() -> None:
    """An unseeded private reading matches the actual next forward pass."""
    model = TinyLearner()
    images = torch.rand(32, 3, 2, 2)
    measured = private_loss(model, images, None)
    assert measured == float(model.training_step(images).detach())


def test_invalid_sizes() -> None:
    """Reject a noncausal reservoir request and incomplete image batches."""
    with pytest.raises(ValueError):
        reservoir_indices(32, 64, 1)
    with pytest.raises(ValueError):
        private_loss(TinyLearner(), torch.zeros(31, 3, 2, 2), 1)
