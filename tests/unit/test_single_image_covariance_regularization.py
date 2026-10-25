"""Check differentiable bank geometry, directional pressure and exact retry state."""

import copy

import numpy as np
import pytest
import torch

from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import capture
from cafl4ds.single_image_covariance_regularization import (
    apply_gradients,
    geometry,
    next_factor,
    prepare_gradients,
    regularizer,
)
from cafl4ds.single_image_gate_protection import rollback
from scripts.run_paired_update_response import fingerprint

SETTINGS = {
    "covariance_target_ratio": 1.1,
    "covariance_warm_multiplier": 1.3,
    "covariance_weight": 10.0,
    "pressure_gain": 9.0,
    "variance_floor_ratio": 1.0,
    "variance_weight": 0.25,
    "regularizer_batch_size": 7,
}


def test_geometry_matches_health_and_preserves_gradients() -> None:
    """Test geometry matches health and preserves gradients."""
    torch.manual_seed(77)
    z = torch.randn(40, 5, dtype=torch.float64, requires_grad=True)
    c, v = geometry(z)
    expected = health_values(z.detach().numpy(), z.detach().numpy())
    if not float(c.detach()) == pytest.approx(expected["covariance"], rel=1e-12):
        raise AssertionError("Scientific state verification failed")
    if not float(v.detach()) == pytest.approx(expected["variance"], rel=1e-12):
        raise AssertionError("Scientific state verification failed")
    if not torch.autograd.gradcheck(lambda x: geometry(x)[0], (z,)):
        raise AssertionError("Scientific state verification failed")


def test_covariance_pressure_points_inward_without_rewarding_low_spread() -> None:
    """Test covariance pressure points inward without rewarding low spread."""
    torch.manual_seed(78)
    z = torch.randn(40, 5, dtype=torch.float64, requires_grad=True)
    c, v = geometry(z)
    loss, values = regularizer(z, float(c.detach()) / 1.29, float(v.detach()) / 1.2, SETTINGS)
    direction = torch.autograd.grad(loss, z)[0]
    if not float(geometry(z - 0.0001 * direction)[0].detach()) < float(c.detach()):
        raise AssertionError("Scientific state verification failed")
    if not values["covariance_weight"] > SETTINGS["covariance_weight"]:
        raise AssertionError("Scientific state verification failed")
    if not values["variance_penalty"] == 0:
        raise AssertionError("Scientific state verification failed")
    low = z.detach().clone().requires_grad_(True)
    loss, values = regularizer(low, float(c.detach()) * 10, float(v.detach()) * 2, SETTINGS)
    if not values["variance_penalty"] > 0:
        raise AssertionError("Scientific state verification failed")
    if not torch.isfinite(torch.autograd.grad(loss, low)[0]).all():
        raise AssertionError("Scientific state verification failed")


class Encoder(torch.nn.Module):
    """Small differentiable learner for bank-gradient checks."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.layer = torch.nn.Linear(4, 3)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed."""
        return self.layer(x).tanh()


class Learner(torch.nn.Module):
    """Small differentiable learner for bank-gradient checks."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = torch.nn.Linear(3, 4)

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        """Training step."""
        return (self.decoder(self.encoder.embed(x)) - x).square().mean()


def test_chunked_bank_gradient_equals_full_autograd_and_retry_is_exact() -> None:
    """Test chunked bank gradient equals full autograd and retry is exact."""
    torch.manual_seed(79)
    model = Learner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    bank, image = (torch.randn(40, 4), torch.randn(1, 4))
    features = model.encoder.embed(bank).detach().numpy()
    reference = np.array([1.0, 0.08, 0.004])
    parent = capture(model, optimizer)
    gradients, rng, _ = prepare_gradients(model, optimizer, image, bank, features, reference, 80, SETTINGS)
    full = copy.deepcopy(model)
    full.zero_grad()
    loss = (
        full.training_step(image)
        + regularizer(full.encoder.embed(bank).double(), reference[2], reference[1], SETTINGS)[0]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
    for actual, parameter in zip(gradients, full.parameters(), strict=True):
        torch.testing.assert_close(actual, parameter.grad, rtol=2e-05, atol=2e-07)
    rollback(model, optimizer, parent)
    apply_gradients(model, optimizer, gradients, rng)
    expected = capture(model, optimizer)
    rollback(model, optimizer, parent)
    if not fingerprint(capture(model, optimizer)) == fingerprint(parent):
        raise AssertionError("Scientific state verification failed")
    apply_gradients(model, optimizer, gradients, rng)
    if not fingerprint(capture(model, optimizer)) == fingerprint(expected):
        raise AssertionError("Scientific state verification failed")
    if not all(float(s["step"]) == 1 for s in optimizer.state.values()):
        raise AssertionError("Scientific state verification failed")


def test_secant_search_reserves_headroom_and_handles_zero_margin() -> None:
    """Test secant search reserves headroom and handles zero margin."""
    factor = next_factor(1 / 64, 1.298, 1.307, 1.3)
    if not factor == pytest.approx(1 / 64 * 0.8 * 0.002 / 0.009):
        raise AssertionError("Scientific state verification failed")
    if not next_factor(1.0, 1.3, 1.4, 1.3) == 0.01:
        raise AssertionError("Scientific state verification failed")
    with pytest.raises(ValueError):
        next_factor(1, 1.31, 1.4, 1.3)
