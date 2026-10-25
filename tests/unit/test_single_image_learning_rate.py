"""Learning-rate search must change displacement without leaking rejected state."""

import numpy as np
import pytest
import torch

from cafl4ds.short_buffer import capture
from cafl4ds.single_image_gate_protection import rollback
from cafl4ds.single_image_learning_rate import choose_first_passing, set_rates, validate_factors


def test_ladder_is_ordered_positive_and_starts_at_nominal():
    np.testing.assert_array_equal(validate_factors([1, 0.25, 0.0625]), [1, 0.25, 0.0625])
    for invalid in [[], [0.5, 0.25], [1, 0], [1, 0.5, 0.75], [1, 1], [1, np.nan]]:
        with pytest.raises(ValueError):
            validate_factors(invalid)


def test_first_safe_rate_is_selected_and_incomplete_search_rejected():
    factors = [1, 0.25, 0.0625, 0.015625]
    assert choose_first_passing([1, 1, 0, 0], [0.4, 0.1, 0, 0], factors, 0.2) == 0.25
    assert choose_first_passing([1, 1, 1, 1], [0.4, 0.3, 0.25, 0.21], factors, 0.2) == 0
    assert choose_first_passing([0, 0, 0, 0], [0, 0, 0, 0], factors, 0.2) == 0
    with pytest.raises(ValueError, match="first passing"):
        choose_first_passing([1, 1, 0, 0], [0.1, 0.1, 0, 0], factors, 0.2)
    with pytest.raises(ValueError, match="exhaust"):
        choose_first_passing([1, 0, 0, 0], [0.4, 0, 0, 0], factors, 0.2)


def test_group_rates_are_scaled_consistently():
    p = torch.nn.Parameter(torch.ones(1))
    q = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([dict(params=[p], lr=0.01), dict(params=[q], lr=0.001)])
    set_rates(optimizer, [0.01, 0.001], 0.25)
    np.testing.assert_allclose([g["lr"] for g in optimizer.param_groups], [0.0025, 0.00025])


def test_retry_matches_independent_adamw_step_and_advances_momentum_once():
    from scripts.run_paired_update_response import fingerprint

    torch.manual_seed(73)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    x = torch.ones(1, 3)
    for _ in range(3):
        optimizer.zero_grad()
        model(x).square().sum().backward()
        optimizer.step()
    before = capture(model, optimizer)
    original_parameters = [p.detach().clone() for p in model.parameters()]
    trial_states = []
    for factor in [1, 0.25]:
        set_rates(optimizer, [0.01], factor)
        optimizer.zero_grad()
        model(x).square().sum().backward()
        optimizer.step()
        set_rates(optimizer, [0.01], 1)
        trial_states.append(capture(model, optimizer))
        rollback(model, optimizer, before)
        assert fingerprint(capture(model, optimizer)) == fingerprint(before)
    full, small = trial_states
    for initial, full_value, small_value in zip(
        original_parameters, full["model"].values(), small["model"].values(), strict=True
    ):
        torch.testing.assert_close(small_value - initial, 0.25 * (full_value - initial), rtol=2e-5, atol=2e-8)
    # Two rejected full/quarter attempts followed by an independently chosen small step.
    for _ in range(2):
        set_rates(optimizer, [0.01], 1)
        optimizer.zero_grad()
        model(x).square().sum().backward()
        optimizer.step()
        rollback(model, optimizer, before)
    set_rates(optimizer, [0.01], 0.25)
    optimizer.zero_grad()
    model(x).square().sum().backward()
    optimizer.step()
    set_rates(optimizer, [0.01], 1)
    assert fingerprint(capture(model, optimizer)) == fingerprint(small)
    for state in optimizer.state.values():
        assert float(state["step"]) == 4
