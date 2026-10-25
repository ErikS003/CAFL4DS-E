"""Check source/seed aggregation, repeated opposite preferences and complete restore."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.reservoir_action_replication import aggregate, capture, guard_learning, protection, restore


def test_unanimous_benefit_does_not_create_switching_opportunity():
    """A repeatable constant-action benefit is distinct from action-selection value."""
    result = aggregate(np.ones((3, 3, 8, 3, 3)))
    assert result["benefit_passed"] and not result["opportunity_passed"]
    assert np.array_equal(result["overall_effect"], [1, 1, 1])


def test_work_cap_rejects_repeated_jobs_and_insufficient_remaining_budget(tmp_path):
    """A manual stage retry cannot silently spend the fixed matrix budget twice."""
    import json  # noqa: PLC0415

    out = tmp_path / "block1-seed5003"
    attempts = out / "attempts"
    attempts.mkdir(parents=True)
    guard_learning(tmp_path, out, replay=False)
    (attempts / "collect_trunk.json").write_text(json.dumps({"ssl_started": 1024}))
    with pytest.raises(RuntimeError, match="already attempted"):
        guard_learning(tmp_path, out, replay=False)
    (attempts / "collect_trunk.json").write_text(json.dumps({"ssl_started": 55296}))
    with pytest.raises(RuntimeError, match="budget"):
        guard_learning(tmp_path, out, replay=True)


def test_positive_overall_cannot_hide_failed_block_or_support_domain():
    """Each source and support/domain budget remains independently visible."""
    values = np.ones((3, 3, 8, 3, 3))
    values[0] = -0.1
    assert not aggregate(values)["benefit_passed"]
    values[:] = 1
    values[0, :, :, 0, 2] = -1.01
    assert not aggregate(values)["benefit_gates"]["block_draw_domain_average_budget"]


def test_opportunity_requires_both_directions_repeated_across_sources_and_seeds():
    """A contrary seed vetoes stability and a single source cannot qualify."""
    values = np.zeros((3, 3, 8, 3, 3))
    values[:2, :2, :2] = 1
    values[:2, :2, 2:4] = -1
    assert aggregate(values)["opportunity_passed"]
    values[0, 2, 0] = -1
    assert not aggregate(values)["opportunity_passed"]
    values[:] = 0
    values[0, :, :2] = 1
    values[0, :, 2:4] = -1
    assert not aggregate(values)["opportunity_passed"]


def test_incomplete_matrix_fails_instead_of_reweighting_available_runs():
    """Missing runs cannot be dropped from the prespecified population."""
    with pytest.raises(ValueError, match="3 blocks"):
        aggregate(np.ones((2, 3, 8, 3, 3)))


def test_complete_restore_keeps_adam_gradients_modes_and_rng():
    """A branch starts independently with every saved learner component restored."""
    model = torch.nn.Linear(2, 1)
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 0.001})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    model.eval()
    state = capture(model, optimizer)
    expected_random = torch.rand(4)
    recovered, adam = restore(model, config, state)
    assert not recovered.training and len(adam.state) == len(optimizer.state)
    assert torch.equal(torch.rand(4), expected_random)
    for name, parameter in recovered.named_parameters():
        assert torch.equal(parameter.grad, state["gradients"][name])
        assert parameter.data_ptr() != dict(model.named_parameters())[name].data_ptr()


def test_drawdown_uses_parent_and_declared_irregular_time_grid():
    """Protection summaries retain the parent peak and weight real arrival gaps."""
    curve = np.asarray([60, 58, 59, 61], dtype=float)[:, None, None] * np.ones((4, 3, 3))
    result = protection(curve, np.full((3, 3), 59))
    assert np.all(result["maximum_drawdown"] == 2)
    assert np.all(result["maximum_loss_below_M0"] == 1)
    assert np.all(result["arrival_weighted_drawdown"] == 0.875)
