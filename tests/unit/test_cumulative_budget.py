"""Cumulative reference discipline, path feasibility and matched-work stopping."""

import copy

import numpy as np
import torch

from cafl4ds.cumulative_budget import CumulativeBudgetStudy, path_status, status
from cafl4ds.update_filtering import optimize_current


def measured(gain=(0.0, 0.0, 0.0), width=0.1):
    gain = np.asarray(gain, float)
    return {
        "mean_gain_pp": gain.tolist(),
        **{
            k: {"lower": (gain - width).tolist(), "upper": (gain + width).tolist(), "half_width": [width] * 3}
            for k in ("support", "query", "crossed")
        },
    }


def episode(name):
    return {
        "source": name,
        "points": [
            {"acquired": measured(), "warm": measured(), **({"local": measured()} if b else {})} for b in range(5)
        ],
    }


def test_cumulative_loss_does_not_reset_with_each_local_block():
    p = episode("a")["points"]
    p[3]["acquired"] = measured((-0.3, 0.8, -1.4))
    p[3]["local"] = measured((-0.2, 0.0, -0.4))
    assert status(p[3]["local"]) == ["within", "within"]
    assert status(p[3]["acquired"])[1] == "violation"
    assert path_status(p, 3) == "violation"


def test_skip_preserves_parameters_optimizer_and_random_stream():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    parameters = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    for _ in range(32):
        assert optimize_current(model, optimizer, torch.zeros(32, 2), False) is None
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in parameters.items())
    assert state == optimizer.state_dict()
    assert torch.equal(rng, torch.get_rng_state())


def test_final_recovery_cannot_hide_intermediate_budget_violation():
    p = episode("a")["points"]
    p[2]["warm"] = measured((0, 2, -2))
    assert path_status(p, 4) == "violation"


def test_original_reference_uncertainty_is_not_certified_by_holding():
    rows = [episode(str(i)) for i in range(6)]
    rows[0]["points"][0]["warm"] = measured((0, 1.1, -1.1), width=0.3)
    result = CumulativeBudgetStudy().summarize(rows, 64, 384)
    assert result["initial_reference_unresolved_histories"] == ["0"]
    assert result["policies"]["informed_stop"]["path_status"][0] == "unknown"
    assert not result["measurement_and_opportunity_ready"]


def test_random_stopping_matches_total_work_and_does_not_certify_a_policy():
    rows = [episode(str(i)) for i in range(6)]
    for i, e in enumerate(rows):
        for b in range(1, 5):
            gain = (0.5, 0.5, 0.5) if b == i % 4 + 1 else (0, 0, 0)
            e["points"][b]["acquired"] = measured(gain)
    result = CumulativeBudgetStudy().summarize(rows, 384, 384)
    assert (
        result["policies"]["informed_stop"]["ssl_updates"] == result["policies"]["matched_random_stop"]["ssl_updates"]
    )
    assert result["policies"]["matched_random_stop"]["assignment_permutations"] == 720
    assert not result["policy_qualified"] and not result["predictor_fitted"]


def test_balanced_gain_cannot_purchase_a_domain_violation():
    rows = [episode(str(i)) for i in range(6)]
    rows[0]["points"][1]["acquired"] = measured((4, 10, -2))
    result = CumulativeBudgetStudy().summarize(rows, 384, 384)
    assert result["policies"]["informed_stop"]["blocks"][0] == 0


def test_a_failed_precision_population_is_not_hidden_by_pooling():
    rows = [episode(str(i)) for i in range(6)]
    for e in rows:
        e["points"][2]["acquired"] = measured(width=0.8)
    result = CumulativeBudgetStudy().summarize(rows, 384, 384)
    assert not result["precision"]["acquired_16"]["support"]["passed"]
    assert not result["precision_pass"]
    unchanged = copy.deepcopy(rows)
    CumulativeBudgetStudy().summarize(rows, 384, 384)
    assert rows == unchanged
