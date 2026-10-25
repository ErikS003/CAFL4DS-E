"""Check mature-label causality and matched singleton optimizer controls."""

import copy

import numpy as np
import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.single_image_signal import (
    centered_repeatability,
    gradient_features,
    optimizer_features,
    performance64,
    residual_forecasts,
    seeded_step,
    training_seed,
    zero_step,
)
from cafl4ds.ssl.factory import build_mae


def arrays():
    rng = np.random.default_rng(24)
    origins = np.tile(np.array([8, 16, 24, 32, 40, 48]), 4)
    held = np.repeat([False, False, True, True], 6)
    x = rng.normal(size=(24, 3))
    y = rng.normal(size=(24, 2))
    base = rng.normal(size=(24, 2)) * 0.1
    return x, y, base, origins, origins + 8, held


def test_unmatured_targets_do_not_enter_any_current_forecast():
    x, y, base, origins, ends, held = arrays()
    before, meta = residual_forecasts(x, y, base, origins, ends, held, prefix=24, test_start=32)
    changed = y.copy()
    changed[ends > 40] += 100
    after, _ = residual_forecasts(x, changed, base, origins, ends, held, prefix=24, test_start=32)
    test = held & (origins >= 32)
    for mode in before:
        np.testing.assert_array_equal(before[mode][origins[test] <= 40], after[mode][origins[test] <= 40])
    # Eight prefix labels plus two held labels that have matured at32.
    assert meta["online_fit_rows_by_origin"]["32"] == 10


def test_transfer_never_fits_held_history_labels():
    x, y, base, origins, ends, held = arrays()
    first, _ = residual_forecasts(x, y, base, origins, ends, held, prefix=24, test_start=32)
    y[held] += 50
    changed, _ = residual_forecasts(x, y, base, origins, ends, held, prefix=24, test_start=32)
    np.testing.assert_array_equal(first["transfer"], changed["transfer"])
    assert not np.array_equal(first["frozen"], changed["frozen"])


def test_seeded_future_is_independent_of_skipped_past_draws():
    torch.manual_seed(training_seed("held_1", 32))
    first = torch.rand(8)
    torch.rand(1000)
    torch.manual_seed(training_seed("held_1", 32))
    second = torch.rand(8)
    assert torch.equal(first, second)
    assert training_seed("held_1", 32) != training_seed("held_2", 32)


def tiny_mae():
    encoder = TinyViTEncoder(img_size=16, patch_size=4, embed_dim=12, depth=1, num_heads=3, mlp_ratio=2.0)
    return build_mae(encoder, decoder_dim=16, decoder_depth=1, decoder_heads=4)


def test_paid_gradient_preserves_weights_rng_grads_and_matches_actual_loss():
    torch.set_num_threads(2)
    model = tiny_mae()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.rand(1, 3, 16, 16)
    seeded_step(model, optimizer, image, 17)
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    observed = optimizer_features(model, optimizer)
    features = gradient_features(model, optimizer, image, 19)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    for p, old in zip(model.parameters(), grads, strict=True):
        assert (old is None and p.grad is None) or torch.equal(old, p.grad)
    np.testing.assert_array_equal(observed, optimizer_features(model, optimizer))
    actual = seeded_step(model, optimizer, image, 19)
    assert actual[0] == features[0]
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_zero_tensor_step_moves_momentum_while_none_skips():
    model = tiny_mae()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seeded_step(model, optimizer, torch.rand(1, 3, 16, 16), 17)
    state = copy.deepcopy(model.state_dict())
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    old_steps = [float(s["step"]) for s in optimizer.state.values()]
    zero_step(model, optimizer)
    assert not all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert [float(s["step"]) for s in optimizer.state.values()] == [s + 1 for s in old_steps]


def test_repeatability_removes_parent_only_offsets():
    rng = np.random.default_rng(11)
    effects = rng.normal(size=(6, 3, 1, 1)) + rng.normal(size=(6, 3, 3, 3)) * 0.001
    first = centered_repeatability(effects)
    shifted = centered_repeatability(effects + np.arange(6)[:, None, None, None] * 100)
    np.testing.assert_allclose(
        first["leave_one_draw_out_correlations"], shifted["leave_one_draw_out_correlations"], atol=1e-12
    )
    assert min(first["leave_one_draw_out_correlations"]) > 0.99


def test_float64_ce_retains_low_amplitude_changes():
    p = np.array([[0.7, 0.3], [0.2, 0.8]], np.float32)
    result = performance64({"reader": p}, np.array([0, 1]), np.array([0, 1]))
    assert result["reader_accuracy_balanced"] == 100
    assert result["reader_ce_balanced"] == pytest.approx(-np.log(p.astype(float)[[0, 1], [0, 1]]).mean(), abs=1e-15)
