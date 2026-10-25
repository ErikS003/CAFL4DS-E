"""Numerical validity and read-only invariants for labeled replay diagnostics."""

# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cafl4ds.image_utility import balanced_losses, embed
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.reader_calibration import fit_probe
from cafl4ds.replay_task_alignment import encoder_vector, head_logits, ssl_directions, task_gradient
from cafl4ds.ssl.factory import build_mae
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import optimizer_for, update


def test_task_gradient_matches_balanced_ce_finite_difference() -> None:
    """Check true CE derivative and sklearn head equivalence, not just code shape."""
    torch.set_num_threads(2)
    torch.manual_seed(41)
    method = build_mae(TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3))
    images = torch.rand(24, 3, 16, 16)
    labels = torch.arange(24) % 3
    domains = (torch.arange(24) // 3) % 2
    z = embed(method, images)
    probe = fit_probe(z, labels, torch.arange(12), 0.1)
    probabilities = head_logits(probe, z).softmax(1).numpy()
    expected = probe.classifier.predict_proba((z.numpy() - probe.mean) / probe.scale)
    np.testing.assert_allclose(probabilities, expected, atol=1e-12)
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in method.modules()]
    state = copy.deepcopy(method.state_dict())
    gradient, ce = task_gradient(method, probe, images[12:], labels[12:], domains[12:])
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in method.modules()]
    assert identical(state, method.state_dict())
    assert all(p.grad is None for p in method.parameters())
    direction = gradient / gradient.norm()
    values = []
    for sign in (-1, 1):
        offset = 0
        with torch.no_grad():
            for name, parameter in method.encoder.named_parameters():
                count = parameter.numel()
                perturbation = direction[offset:offset + count].reshape_as(parameter)
                parameter.copy_(state[f"encoder.{name}"] + sign * 1e-3 * perturbation)
                offset += count
        features = embed(method, images[12:])
        loss = F.cross_entropy(head_logits(probe, features), labels[12:], reduction="none")
        values.append(balanced_losses(loss, labels[12:], domains[12:])["balanced"])
    derivative = (values[1] - values[0]) / 2e-3
    assert derivative == pytest.approx(float(gradient.norm()), rel=0.01, abs=1e-4)
    assert ce > 0


def test_virtual_adamw_direction_matches_real_stored_moment_step() -> None:
    """Virtual steps preserve parent/RNG and reproduce actual AdamW displacement."""
    torch.manual_seed(53)
    method = build_mae(TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3))
    images = torch.rand(32, 3, 16, 16)
    optimizer = optimizer_for(method)
    update(method, optimizer, images, 0, None)
    before = encoder_vector(method)
    state, opt_state = copy.deepcopy(method.state_dict()), copy.deepcopy(optimizer.state_dict())
    grads = [p.grad.clone() if p.grad is not None else None for p in method.parameters()]
    rng = torch.get_rng_state().clone()
    sgd, delta, _ = ssl_directions(method, opt_state, images, 79)
    assert identical(state, method.state_dict())
    assert identical(opt_state, optimizer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(identical(g, p.grad) for g, p in zip(grads, method.parameters(), strict=True))
    torch.manual_seed(79)
    update(method, optimizer, images, 1, None)
    torch.testing.assert_close(delta, encoder_vector(method) - before, rtol=0, atol=0)
    assert float(sgd.norm()) > 0
