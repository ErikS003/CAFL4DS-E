"""Meaningful finite-difference, matched-gradient and optimizer-state controls."""

import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.health_calibration import embeddings
from cafl4ds.image_health_prediction import image_step, variance_gradient
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import capture, restore
from cafl4ds.ssl.factory import build_mae
from scripts.run_image_health_prediction_pilot import direction_forecast
from scripts.run_paired_update_response import fingerprint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device-conversion regression")
def test_forecast_accepts_cpu_parent_and_cuda_displacement():
    g = [torch.tensor([2.0], device="cuda")]
    full = {"encoder.p": torch.tensor([3.0], device="cuda")}
    parent = {"encoder.p": torch.tensor([1.0])}
    assert direction_forecast(g, ["p"], full, parent, 2.0) == 2.0


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(83)
    model = build_mae(TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3))
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    images, bank = torch.rand(4, 3, 16, 16), torch.rand(32, 3, 16, 16)
    image_step(model, optimizer, images)
    return model, config, capture(model, optimizer), images, bank


def test_omission_matches_subtracting_candidate_gradient():
    initial, config, state, images, _ = fixture()
    a, ao = restore(initial, config, state)
    b, bo = restore(initial, config, state)
    torch.set_rng_state(state["cpu_rng"])
    first = image_step(a, ao, images, omit=1)
    torch.set_rng_state(state["cpu_rng"])
    second = image_step(b, bo, images, omit=1, reconstruct=True)
    assert first["per_image_losses"] == second["per_image_losses"]
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=2e-7)
    assert fingerprint(state) == fingerprint(capture(*restore(initial, config, state)))


def test_health_gradient_matches_finite_difference_and_preserves_observer_state():
    model, config, state, _, bank = fixture()
    model, optimizer = restore(model, config, state)
    marker = fingerprint(capture(model, optimizer))
    old_grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    gradients, value = variance_gradient(model, bank)
    assert fingerprint(capture(model, optimizer)) == marker
    assert all(
        (a is None and p.grad is None) or torch.equal(a, p.grad)
        for a, p in zip(old_grads, model.parameters(), strict=True)
    )
    length = float(torch.sqrt(sum((g**2).sum() for g in gradients)))
    before = copy.deepcopy(model.state_dict())
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            for (name, p), g in zip(model.encoder.named_parameters(), gradients, strict=True):
                p.copy_(before["encoder." + name] + sign * 1e-3 * g / length)
        z = embeddings(model, bank)
        values.append(health_values(z, z)["variance"])
    assert (values[1] - values[0]) / 2e-3 == pytest.approx(length, rel=0.02)
    assert value > 0


def test_full_step_matches_standard_objective_and_all_replay_state():
    initial, config, state, images, bank = fixture()
    model, optimizer = restore(initial, config, state)
    result = image_step(model, optimizer, images)
    end = capture(model, optimizer)
    z = embeddings(model, bank)
    replay, replay_optimizer = restore(initial, config, state)
    assert result == image_step(replay, replay_optimizer, images)
    assert fingerprint(end) == fingerprint(capture(replay, replay_optimizer))
    np.testing.assert_array_equal(z, embeddings(replay, bank))
    standard, standard_optimizer = restore(initial, config, state)
    standard.train()
    standard_optimizer.zero_grad()
    loss = standard.training_step(images)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(standard.parameters(), 1)
    standard_optimizer.step()
    assert float(loss.detach()) == pytest.approx(result["loss"], abs=1e-7)
    for a, b in zip(model.parameters(), standard.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=2e-7)
