"""Check support leakage barriers and copied-model observation semantics."""

# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import copy

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.replay_task_alignment import encoder_vector, ssl_directions
from cafl4ds.replay_task_readout import support_folds, virtual_step
from cafl4ds.ssl.factory import build_mae
from scripts.run_image_utility_experiment import identical


def test_cross_fitting_preserves_all_cells_without_leakage() -> None:
    """Each observation fold is disjoint from960 fitting labels and balanced."""
    labels = torch.arange(1152) // 192 % 3
    domains = torch.arange(1152) // 576
    rng = torch.get_rng_state().clone()
    folds = support_folds(labels, domains, torch.arange(1152))
    assert torch.equal(rng, torch.get_rng_state())
    assert len(torch.unique(torch.cat([f["heldout"] for f in folds]))) == 1152
    for f in folds:
        assert len(f["heldout"]) == 192 and len(f["fitting"]) == 960
        assert not torch.isin(f["heldout"], f["fitting"]).any()
        for d in (0, 1):
            for c in (0, 1, 2):
                assert int(((domains[f["heldout"]] == d) & (labels[f["heldout"]] == c)).sum()) == 32
    invalid = labels.clone()
    invalid[0] = 1
    with pytest.raises(ValueError, match="per domain/class"):
        support_folds(invalid, domains, torch.arange(1152))


def test_returned_virtual_model_matches_previous_audited_direction() -> None:
    """Return an updated copy suitable for refitting; preserve original state/RNG."""
    torch.set_num_threads(2)
    torch.manual_seed(73)
    parent = build_mae(TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3))
    optimizer = torch.optim.AdamW(parent.parameters(), lr=1e-4, weight_decay=0.05)
    images = torch.rand(32, 3, 16, 16)
    opt_state = copy.deepcopy(optimizer.state_dict())
    state = copy.deepcopy(parent.state_dict())
    rng = torch.get_rng_state().clone()
    _, expected, _ = ssl_directions(parent, opt_state, images, 101)
    copied, delta, _ = virtual_step(parent, opt_state, images, 101)
    torch.testing.assert_close(delta, expected, rtol=0, atol=0)
    torch.testing.assert_close(encoder_vector(copied) - encoder_vector(parent), delta, rtol=0, atol=0)
    assert identical(state, parent.state_dict())
    assert identical(opt_state, optimizer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in parent.parameters())
