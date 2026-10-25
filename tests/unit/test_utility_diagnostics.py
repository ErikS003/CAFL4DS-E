"""Check diagnostic isolation, stable alpha ties and legacy CE replay verification."""

import copy

import pytest
import torch

from cafl4ds.image_utility import FixedProbe, split_query_roles
from cafl4ds.utility_diagnostics import alpha_geometry, diagnostic_outcomes, implemented_keep
from scripts.run_image_utility_experiment import identical
from scripts.run_saved_utility_diagnostics import verify_fixed
from scripts.run_slow_edge_pilot import load_inputs


def test_alpha_geometry_includes_crossing_ties_and_dense_interior_sets() -> None:
    """A crossing tie can create a diet that neither adjacent interval selects."""
    coverage, novelty = torch.tensor([0.0, 2.0, 1.0, 3.0]), torch.tensor([3.0, 1.0, 2.0, 0.0])
    result = alpha_geometry(coverage, novelty, 2)
    ideals = {tuple(row) for row in result["ideal_reachable_sets"]}
    assert ideals == {(0, 2), (0, 1), (1, 3)}
    assert implemented_keep(coverage, novelty, 0.5, 2) == [0, 1]
    checked = {tuple(row["keep"]) for row in result["implemented_checked_sets"]}
    for alpha in torch.linspace(0, 1, 101).tolist():
        assert tuple(implemented_keep(coverage, novelty, alpha, 2)) in checked
    assert len(alpha_geometry(coverage, novelty, 4)["ideal_reachable_sets"]) == 1


def test_refitted_readers_preserve_model_modes_gradients_and_rng() -> None:
    """Supervised outcome fitting cannot alter the SSL model or its next training draw."""
    torch.set_num_threads(2)
    method, _, _, evaluation, _ = load_inputs("synthetic", train_per_cell=64, evaluation_per_cell=36)
    roles = split_query_roles(evaluation)
    fixed = FixedProbe.fit(method, evaluation)
    method.train()
    method.encoder.blocks[-1].eval()
    for parameter in method.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = copy.deepcopy(method.state_dict())
    gradients = [p.grad.clone() for p in method.parameters()]
    modes = [m.training for m in method.modules()]
    rng = torch.get_rng_state().clone()
    outcomes = diagnostic_outcomes(method, evaluation, roles, fixed, draws=1)
    assert identical(state, method.state_dict())
    assert modes == [m.training for m in method.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(g, p.grad) for g, p in zip(gradients, method.parameters(), strict=True))
    for role in ("development", "holdout"):
        assert outcomes[role]["refit_ce_balanced"] >= 0
        assert 0 <= outcomes[role]["linear_error_balanced"] <= 1
        assert 0 <= outcomes[role]["knn_error_balanced"] <= 1


def test_replay_check_normalizes_json_horizon_keys_and_rejects_changed_ce() -> None:
    """New outcome metrics do not obscure a change to the original fixed-head outcome."""
    expected = {"4": {"holdout": {"fixed_ce_balanced": 1.0}}}
    actual = {4: {"holdout": {"fixed_ce_balanced": 1.0, "refit_ce_balanced": 0.4}}}
    assert verify_fixed(actual, expected) == 0
    actual[4]["holdout"]["fixed_ce_balanced"] = 1.01
    with pytest.raises(ValueError, match="Replayed fixed CE changed"):
        verify_fixed(actual, expected)
