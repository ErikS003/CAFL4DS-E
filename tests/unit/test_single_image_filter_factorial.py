"""Scientific invariants for the objective-by-feature singleton follow-up."""

import copy

import numpy as np
import pytest
import torch

from cafl4ds.single_image_filter_factorial import (
    adamw_displacement,
    admission_threshold,
    fit_factorial,
    linear_readout,
    reference_gradient,
)
from cafl4ds.single_image_filter_factorial_analysis import clustered_interval, forecast_scores


@pytest.mark.parametrize("history_steps", [0, 3])
def test_analytic_adamw_matches_clipped_optimizer(history_steps):
    torch.manual_seed(77)
    model = torch.nn.Linear(5, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.07)
    x, target = torch.randn(4, 5), torch.randn(4, 2)
    for _ in range(history_steps):
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    parameters = list(model.parameters())
    grads = torch.autograd.grad(20 * (model(x) - target).square().mean(), parameters)
    before = torch.cat([p.detach().flatten().clone() for p in parameters])
    state = copy.deepcopy(optimizer.state_dict())
    predicted, norm = adamw_displacement(optimizer, parameters, grads)
    assert norm > 1
    assert all(torch.equal(p, q) for p, q in zip(before, torch.cat([p.detach().flatten() for p in parameters])))
    for group, saved in zip(optimizer.state_dict()["param_groups"], state["param_groups"]):
        assert group == saved
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(parameters, grads):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(parameters, 1)
    optimizer.step()
    actual = torch.cat([p.detach().flatten() for p in parameters]) - before
    torch.testing.assert_close(predicted, actual, rtol=1e-5, atol=6e-8)


def test_threshold_only_uses_its_supplied_development_predictions():
    p = np.array([-0.3, -0.2, -0.1, 0.1, 0.2])
    threshold = admission_threshold(p, 0.6)
    assert threshold == pytest.approx(-0.02)
    assert not np.any(p[p < threshold] >= 0)
    assert admission_threshold([1, 2, 3], 0.6) == 0
    with pytest.raises(ValueError):
        admission_threshold([np.nan], 0.6)


def test_unused_parameter_is_not_given_a_decay_or_momentum_step():
    active = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    unused = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = torch.optim.AdamW([active, unused], lr=0.02, weight_decay=0.5)
    gradients = (torch.tensor([0.4, -0.7]), None)
    proposed, _ = adamw_displacement(optimizer, [active, unused], gradients)
    assert proposed[-1] == 0
    active.grad = gradients[0].clone()
    before = torch.cat([active.detach().clone(), unused.detach().clone()])
    torch.nn.utils.clip_grad_norm_([active, unused], 1)
    optimizer.step()
    actual = torch.cat([active.detach(), unused.detach()]) - before
    torch.testing.assert_close(proposed, actual, atol=1e-7, rtol=1e-5)


def test_whole_history_validation_and_refit_roster():
    rng = np.random.default_rng(6)
    groups = np.repeat(["train0", "train1", "valid"], 20)
    seeds = np.tile(np.repeat([11, 13], 10), 3)
    x = rng.normal(size=(60, 3))
    a = dict(
        histories=groups,
        seeds=seeds,
        state=x,
        base=x,
        rich=x,
        targets=np.column_stack([x[:, 0] * 0.01, x[:, 1] * 0.005]),
    )
    settings = dict(
        validation_histories=["valid"],
        trajectory_seeds=[11, 13],
        max_validation_admission=0.6,
        regressors={"linear": {"_target_": "sklearn.linear_model.Ridge", "alpha": 1}},
    )
    models, selection = fit_factorial(a, settings)
    assert len(models) == 6
    for key, record in selection.items():
        assert record["validation_rows"] == 20
        assert record["rows"] == 60
        assert record["threshold"] <= 0
        assert set(models[key]) == {11, 13}
        np.testing.assert_allclose(record["threshold"], min(0, np.quantile(record["validation_predictions"], 0.6)))


def test_bootstrap_keeps_seed_pairs_inside_source_histories():
    values = np.array([[0, 2], [2, 4], [10, 20], [14, 24]])
    groups = np.array(["a", "a", "b", "b"])
    draws = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    result = clustered_interval(values, groups, draws)
    np.testing.assert_allclose(result["mean"], [6.5, 12.5])
    expected = np.array([[1, 3], [12, 22], [6.5, 12.5], [6.5, 12.5]])
    np.testing.assert_allclose(result["ci95"], np.quantile(expected, [0.025, 0.975], axis=0))


def test_forecast_ties_are_not_fabricated_direction_success():
    result = forecast_scores([0, 0, -0.1, 0.1], [0, 0, -0.2, 0.2])
    assert result["nonzero"] == 2
    assert result["balanced_sign_accuracy"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt(0.005))
    assert result["mae"] == pytest.approx(0.05)
    assert forecast_scores([0, 0], [0, 0])["balanced_sign_accuracy"] is None


def test_task_gradient_matches_frozen_probe_finite_difference():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(12, 4)

        def embed(self, x):
            return self.projection(x.flatten(1))

    torch.manual_seed(93)
    model = torch.nn.Module()
    model.encoder = Encoder()
    model.decoder = torch.nn.Linear(4, 12)  # Unused by the task reader.
    model.double()
    panel = dict(
        support_images={i: torch.randn(50, 3, 2, 2, dtype=torch.float64) for i in [1, 2, 3]},
        support_labels={i: torch.tensor([0, 1] * 25) for i in [1, 2, 3]},
        query_images=torch.randn(32, 3, 2, 2, dtype=torch.float64),
        query_labels=np.repeat([0, 1], 16),
        query_domains=np.tile(np.repeat([0, 1], 8), 2),
    )
    readout = linear_readout(model, panel, [1, 2, 3], 5)
    heads = readout["heads"]

    def loss():
        z = model.encoder.embed(panel["query_images"]).detach().numpy()
        logit = z @ heads[:, :-1].T + heads[:, -1]
        return np.mean(np.logaddexp(0, logit) - panel["query_labels"][:, None] * logit)

    z = model.encoder.embed(panel["query_images"]).detach().numpy()
    probabilities = 1 / (1 + np.exp(-(z @ heads[:, :-1].T + heads[:, -1])))
    np.testing.assert_allclose(probabilities.T, readout["probabilities"][:, 0, :, 1], rtol=1e-10, atol=1e-12)
    gradient = reference_gradient(model, panel, heads, 5)
    direction = torch.randn_like(gradient)
    direction /= direction.norm()
    original = [p.detach().clone() for p in model.parameters()]

    def perturb(amount):
        offset = 0
        with torch.no_grad():
            for p, saved in zip(model.parameters(), original):
                p.copy_(saved + amount * direction[offset : offset + p.numel()].reshape_as(p))
                offset += p.numel()

    perturb(1e-5)
    positive = loss()
    perturb(-1e-5)
    negative = loss()
    perturb(0)
    assert float(gradient @ direction) == pytest.approx((positive - negative) / 2e-5, abs=1e-8, rel=1e-6)
    decoder_size = sum(p.numel() for p in model.decoder.parameters())
    assert torch.count_nonzero(gradient[-decoder_size:]) == 0
