"""Check private loss observation, shadow hysteresis and truly matched continuation diets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.health_replay_mix import VariableWindowReplay
from cafl4ds.historical_loss_screen import ShadowLossAlarm, action_effects, align_continuation, bank_losses
from cafl4ds.retention_anchor import private_loss
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


class ToyLoss(torch.nn.Module):
    """Expose a random, differentiable per-image objective with independently restored modes."""

    def __init__(self):
        """Keep the unit measurement fast and independent of the scientific MAE."""
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)

    def per_sample_loss(self, images):
        """Use randomized views to exercise RNG isolation."""
        views = images + torch.rand_like(images)
        return self.encoder(views).square().mean(1)

    def training_step(self, images):
        """Match the existing scalar-loss measurement helper."""
        return self.per_sample_loss(images).mean()


def test_private_bank_loss_is_repeatable_and_preserves_complete_observer_state():
    """Common masks reproduce readings while learner/mode/gradient/RNG state is untouched."""
    model = ToyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    images = torch.rand(64, 3)
    model.training_step(images).backward()
    optimizer.step()
    model.eval()
    model.encoder.train()
    before = fingerprint(capture(model, optimizer))
    grads = fingerprint([p.grad for p in model.parameters()])
    first = bank_losses(model, images)
    second = bank_losses(model, images)
    np.testing.assert_array_equal(first, second)
    assert fingerprint(capture(model, optimizer)) == before
    assert fingerprint([p.grad for p in model.parameters()]) == grads
    assert not np.array_equal(first[0], first[1])
    for row, seed in zip(first, (84211, 84223), strict=True):
        assert abs(row.mean() - private_loss(model, images, seed)) < 1e-7


def test_shadow_hysteresis_and_unscheduled_observations_do_not_actuate():
    """Persistent high loss emits once; two low readings rearm without changing the reference."""
    control = ShadowLossAlarm()
    assert not control.observe(132, 1.0)
    series = [1.02, 1.02, 1.08, 1.01, 1.015, 1.01, 1.01, 1.02, 1.02]
    alarms = [control.observe(140 + 8 * i, loss) for i, loss in enumerate(series)]
    assert [i for i, a in enumerate(alarms) if a] == [1, 8]
    assert control.reference == 1.0
    assert not control.observe(205, float("nan"))
    assert not hasattr(control, "split")
    with pytest.raises(ValueError, match="advance"):
        control.observe(204, 1.0)


def test_continuation_alignment_matches_pixels_after_different_sampling_histories():
    """Both branches ingest all images, then a common sampler yields identical future batches."""
    current, replay = VariableWindowReplay(), VariableWindowReplay()
    for step in range(40):
        pixels = torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1).float()
        current.observe(pixels, SimpleNamespace(step=step), 0)
        replay.observe(pixels, SimpleNamespace(step=step), 24 if step else 0)
    assert not torch.equal(current.marker()["rng"], replay.marker()["rng"])
    before = torch.get_rng_state().clone()
    align_continuation(current, 192)
    align_continuation(replay, 192)
    assert torch.equal(before, torch.get_rng_state())
    for step in range(40, 44):
        pixels = torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1).float()
        a = current.observe(pixels, SimpleNamespace(step=step), 16)
        b = replay.observe(pixels, SimpleNamespace(step=step), 16)
        assert torch.equal(a, b) and current.last == replay.last


def test_domain_conservative_action_utility_is_not_balanced_improvement():
    """A balanced gain does not erase damage to one domain, and draw pooling stays explicit."""
    current = np.full((3, 3), 50.0)
    replay = current + np.array([[1, 3, -1], [1, -1, 3], [0, 0, 0]])
    result = action_effects(current, replay)
    np.testing.assert_array_equal(result["per_draw_utility"], [-1, -1, 0])
    assert result["pooled_utility"] == pytest.approx(2 / 3)


def test_shadow_rejects_missing_reference_and_nonfinite_scheduled_losses():
    """Bad readings never silently become healthy observations."""
    control = ShadowLossAlarm()
    with pytest.raises(ValueError, match="Missing"):
        control.observe(140, 1.0)
    with pytest.raises(ValueError, match="Positive finite"):
        control.observe(132, float("nan"))
