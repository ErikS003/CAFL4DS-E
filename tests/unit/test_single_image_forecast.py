"""Check causal forecasts, independent RLS arithmetic and genuine singleton updates."""

import copy

import numpy as np
import pytest
import torch

from cafl4ds.single_image_forecast import PrefixRidge, designs, error_summary, forecasts, observer, single_step


def test_recursive_ridge_matches_direct_refit():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(48, 5))
    y = x @ rng.normal(size=(5, 3)) + rng.normal(size=(48, 3)) * 0.1
    fit = PrefixRidge(x[:20], y[:20])
    a = fit.design(x)
    penalty = np.diag(np.r_[1e-8, np.full(5, 10.0)])
    for i in range(20, 48):
        beta = np.linalg.solve(a[:i].T @ a[:i] + penalty, a[:i].T @ ((y[:i] - fit.ycenter) / fit.yscale))
        np.testing.assert_allclose(fit.predict(x[i])[0], fit.ycenter + a[i] @ beta * fit.yscale, atol=1e-10)
        fit.update(x[i], y[i])


def synthetic_series():
    rng = np.random.default_rng(11)
    metrics = ["rank", "variance", "covariance", "cka", "cosine", "feature_norm", "bank_reconstruction"]
    state = rng.normal(size=(41, 7)).cumsum(0)
    return state[:-1], state[1:], rng.normal(size=(40, 4)), metrics, rng.normal(size=(40, 2))


def test_future_targets_cannot_change_current_forecast():
    before, after, image, metrics, training = synthetic_series()
    _, original = forecasts(before, after, image, metrics, training, 16)
    changed = after.copy()
    changed[27:] += 1000
    _, future = forecasts(before, changed, image, metrics, training, 16)
    for name in original:
        np.testing.assert_array_equal(original[name][:12], future[name][:12])
    # The online predictor may learn the changed target only for the NEXT step.
    assert not np.array_equal(original["online_history_image"][12:], future["online_history_image"][12:])


def test_no_future_rows_enter_design_or_prefix_scaling():
    before, after, image, metrics, training = synthetic_series()
    a = designs(before, after, image, metrics, training)
    before[30:] += 100
    after[30:] -= 100
    image[30:] += 100
    training[30:] -= 100
    b = designs(before, after, image, metrics, training)
    for name in a:
        np.testing.assert_array_equal(a[name][:30], b[name][:30])


def test_observer_restores_rng_modes_and_weights():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout()).train()
    model[0].eval()
    rng = torch.get_rng_state()
    state = copy.deepcopy(model.state_dict())
    with observer(model, 99):
        assert not model.training
        torch.rand(5)
    assert model.training and not model[0].training and model[1].training
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


class ToyLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, image):
        return (self.weight * image).square().mean()


def test_singleton_really_advances_optimizer_once():
    model = ToyLearner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss, gradient = single_step(model, optimizer, torch.ones(1, 1, 2, 2))
    assert loss == 1 and gradient > 0 and model.weight.item() < 1
    assert optimizer.state[model.weight]["step"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        single_step(model, optimizer, torch.ones(2, 1, 2, 2))


def test_accuracy_ties_are_a_separate_direction():
    score = error_summary(np.array([-1.0, 0.0, 1.0, 0.0]), np.zeros(4))
    assert score["direction_accuracy"] == 0.5
    assert score["changed_direction_accuracy"] == 0
    assert score["balanced_direction_recall"] == pytest.approx(1 / 3)
    assert score["mae"] == 0.5
