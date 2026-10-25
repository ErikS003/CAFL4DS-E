"""Physical numerical checks for all five health derivatives."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cafl4ds.health_calibration import embeddings
from cafl4ds.health_metric_derivatives import METRICS, differentiable_health, health_gradients
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import capture
from cafl4ds.ssl.factory import build_mae
from scripts.run_paired_update_response import fingerprint


@pytest.mark.parametrize("metric", METRICS)
def test_metric_value_and_feature_derivative(metric: str) -> None:
    """Compare values and directional derivatives against independent NumPy calculations."""
    rng = np.random.default_rng(104)
    z, warm = rng.normal(size=(48, 9)), rng.normal(size=(48, 9))
    direction = rng.normal(size=z.shape)
    direction /= np.linalg.norm(direction)
    features = torch.tensor(z, requires_grad=True)
    value = differentiable_health(features, torch.tensor(warm))[metric]
    assert float(value.detach()) == pytest.approx(health_values(z, warm)[metric], abs=1e-12)
    derivative = torch.autograd.grad(value, features)[0]
    projected = float((derivative * torch.tensor(direction)).sum())
    epsilon = 1e-5
    actual = (
        health_values(z + epsilon * direction, warm)[metric] - health_values(z - epsilon * direction, warm)[metric]
    ) / (2 * epsilon)
    assert projected == pytest.approx(actual, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("metric", ("cka", "cosine"))
def test_reference_is_stationary(metric: str) -> None:
    """A zero derivative at identical reference geometry must not be called broken."""
    z = torch.randn(40, 8, dtype=torch.float64, requires_grad=True)
    value = differentiable_health(z, z.detach().clone())[metric]
    gradient = torch.autograd.grad(value, z)[0]
    assert float(value.detach()) == pytest.approx(0, abs=1e-12)
    assert float(gradient.norm()) < 1e-12


def test_parameter_derivatives_and_complete_observer_invariance() -> None:
    """Check the encoder chain rule and preserve preexisting gradients, modes and RNG."""
    torch.set_num_threads(2)
    torch.manual_seed(219)
    model = build_mae(TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.001
    old_grads = [p.grad.clone() for p in model.parameters()]
    bank = torch.rand(32, 3, 16, 16)
    warm = torch.tensor(embeddings(model, torch.rand_like(bank)))
    marker = fingerprint(capture(model, optimizer))
    gradients, parent = health_gradients(model, bank, warm)
    assert fingerprint(capture(model, optimizer)) == marker
    assert all(torch.equal(g, p.grad) for g, p in zip(old_grads, model.parameters(), strict=True))
    state = copy.deepcopy(model.state_dict())
    for metric in METRICS:
        vector = gradients[metric]
        norm = float(torch.sqrt(sum(g.double().square().sum() for g in vector)))
        endpoints = []
        for sign in (-1, 1):
            with torch.no_grad():
                for (name, p), g in zip(model.encoder.named_parameters(), vector, strict=True):
                    p.copy_(state["encoder." + name] + sign * 1e-4 * g / norm)
            endpoints.append(health_values(embeddings(model, bank), warm.numpy())[metric])
        assert (endpoints[1] - endpoints[0]) / 2e-4 == pytest.approx(norm, rel=0.03, abs=1e-4)
        model.load_state_dict(state)
        assert parent[metric] == pytest.approx(health_values(embeddings(model, bank), warm.numpy())[metric], abs=1e-11)
