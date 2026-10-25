"""Tests for the augmentation-strength knob added for the P0.2.4 heavy-augmentation stressor.

Confirms :func:`make_ssl_augment` preserves output shape and value range at both default and
over-strong settings, that the strength knobs thread through the SSL factories, and that the
ColorJitter hue stays inside torchvision's valid ``<= 0.5`` bound at large strengths.
"""

import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.augment import make_ssl_augment
from cafl4ds.ssl.factory import build_barlow, build_simsiam


def _batch(n: int = 4, size: int = 16) -> torch.Tensor:
    """A small random image batch in ``[0, 1]``."""
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


def test_default_and_heavy_augment_preserve_shape_and_range() -> None:
    """Both the default and an over-strong pipeline map a [0,1] batch to the same shape in [0,1]."""
    x = _batch(size=16)
    for aug in (make_ssl_augment(16), make_ssl_augment(16, min_scale=0.08, jitter_strength=2.0)):
        out = aug(x)
        assert out.shape == x.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_hue_is_capped_at_half_for_large_strength() -> None:
    """A large jitter strength must not push ColorJitter's hue past torchvision's 0.5 cap."""
    # jitter_strength=10 would give hue 1.0 uncapped; construction must still succeed (hue clamped).
    aug = make_ssl_augment(16, jitter_strength=10.0)
    assert aug(_batch(size=16)).shape == (4, 3, 16, 16)


def test_factories_thread_augment_strength() -> None:
    """The strength knobs reach the SSL methods and produce runnable positive pairs."""
    enc = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    for build in (build_simsiam, build_barlow):
        method = build(enc, aug_min_scale=0.08, aug_jitter_strength=2.0)
        view_1, view_2 = method.two_view(_batch(size=16))
        assert view_1.shape == view_2.shape == (4, 3, 16, 16)
