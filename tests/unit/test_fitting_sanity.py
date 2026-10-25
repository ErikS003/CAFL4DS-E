"""Check fitting diagnostics expose gradient flow while preserving the caller's state."""

import torch

from cafl4ds.fitting_sanity import balanced_subset, fitting_run
from cafl4ds.models.vit import TinyViTEncoder


def test_private_balanced_subset() -> None:
    """The sampler uses exactly six per cell and leaves training RNG untouched."""
    labels = torch.tensor([c for d in (0, 1) for c in (0, 1, 2) for _ in range(10)])
    domains = torch.tensor([d for d in (0, 1) for c in (0, 1, 2) for _ in range(10)])
    rng = torch.get_rng_state().clone()
    selected = balanced_subset(labels, domains)
    assert torch.equal(rng, torch.get_rng_state()) and len(selected.unique()) == 36
    for d in (0, 1):
        for c in (0, 1, 2):
            assert ((labels[selected] == c) & (domains[selected] == d)).sum() == 6


def test_gradient_flow_freezing_and_private_training() -> None:
    """Joint fitting moves encoder parameters; head-only fitting cannot move them."""
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3)
    images = torch.rand(9, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    labels = torch.arange(9) % 3
    state = {k: v.clone() for k, v in encoder.state_dict().items()}
    rng = torch.get_rng_state().clone()
    _, _, _, _, joint = fitting_run(encoder, images, labels, "joint_higher_lr", 75021, steps=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in state.items())
    assert joint[-1]["encoder_gradient_norm"] > 0 and joint[-1]["encoder_parameter_delta"] > 0
    _, _, mean, scale, frozen = fitting_run(encoder, images, labels, "head_standardized", 75021, steps=1)
    assert frozen[-1]["encoder_gradient_norm"] == frozen[-1]["encoder_parameter_delta"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        features = encoder.embed(images)
    assert torch.allclose(mean, features.mean(0)) and torch.all(scale > 0)
