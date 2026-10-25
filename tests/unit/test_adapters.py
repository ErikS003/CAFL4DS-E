"""Tests for constrained continual-learning encoders."""

from __future__ import annotations

import pytest
import torch

from cafl4ds.models.adapters import ResidualAdapterEncoder
from cafl4ds.models.vit import TinyViTEncoder


def _backbone() -> TinyViTEncoder:
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=2, num_heads=3)


def test_adapter_starts_as_exact_identity_and_freezes_backbone() -> None:
    """The new arm is representation-matched at t0 and trains only its residual."""
    backbone = _backbone()
    images = torch.rand(3, 3, 16, 16)
    expected = backbone.embed(images)
    wrapped = ResidualAdapterEncoder(backbone, bottleneck_dim=4)
    assert torch.equal(wrapped.embed(images), expected)
    assert not any(parameter.requires_grad for parameter in wrapped.backbone.parameters())
    assert all(parameter.requires_grad for parameter in wrapped.adapter.parameters())


def test_adapter_receives_gradient_while_backbone_stays_frozen() -> None:
    """A training signal changes the residual path without touching stable weights."""
    wrapped = ResidualAdapterEncoder(_backbone(), bottleneck_dim=4)
    wrapped.embed(torch.rand(3, 3, 16, 16)).square().mean().backward()
    assert wrapped.adapter[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in wrapped.backbone.parameters())


def test_adapter_rejects_non_positive_bottleneck() -> None:
    """A residual bottleneck must contain at least one unit."""
    with pytest.raises(ValueError, match="must be positive"):
        ResidualAdapterEncoder(_backbone(), bottleneck_dim=0)
