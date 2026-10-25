"""Check supervised controls do not consume evaluation data, caller models or RNG."""

import torch

from cafl4ds.measurement_controls import shuffle_within_domain, train_control
from cafl4ds.models.vit import TinyViTEncoder


def test_training_is_private_and_query_independent() -> None:
    """Changing images outside the supplied fit slice cannot change the fitted control."""
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3)
    state = {k: v.clone() for k, v in encoder.state_dict().items()}
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(8))
    labels = torch.arange(12) % 3
    rng = torch.get_rng_state().clone()
    a, head_a, log_a = train_control(encoder, images[:9], labels[:9], 74021, epochs=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in state.items())
    images[9:] = 999
    labels[9:] = 2
    b, head_b, log_b = train_control(encoder, images[:9], labels[:9], 74021, epochs=1)
    assert log_a == log_b
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert all(torch.equal(v, head_b.state_dict()[k]) for k, v in head_a.state_dict().items())


def test_shuffle_preserves_domain_counts_and_rng() -> None:
    """Permutation changes correspondence while preserving per-domain class totals."""
    labels = torch.arange(60) % 3
    domains = torch.arange(60) // 30
    rng = torch.get_rng_state().clone()
    shuffled = shuffle_within_domain(labels, domains, 74017)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.equal(labels, shuffled)
    for domain in (0, 1):
        assert torch.equal(labels[domains == domain].sort().values, shuffled[domains == domain].sort().values)
