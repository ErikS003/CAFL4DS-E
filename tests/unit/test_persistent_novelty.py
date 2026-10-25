"""Persistence, exploration, causal windows and stateful selection regression checks."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.persistent_novelty import ANCHORS, persistent_indices, persistent_scores, run_stream, select_window
from cafl4ds.short_buffer import capture, restore
from scripts.run_paired_update_response import fingerprint
from tests.unit.test_short_buffer import Toy


def test_sustained_change_gets_priority_and_an_isolated_spike_does_not():
    history = torch.tensor([[1.0, 0.0]]).repeat(16, 1)
    features = history[:1].repeat(64, 1)
    features[8] = torch.tensor([0.0, 1.0])
    features[24:28] = torch.tensor([0.0, 1.0])
    scores, _ = persistent_scores(features, history)
    assert float(scores[8]) == 0
    assert float(scores[24]) > 1
    assert float(scores[25]) > 1
    assert torch.equal(scores[-2:], torch.zeros(2))


def test_directional_disagreement_does_not_count_as_persistent_novelty():
    history = torch.tensor([[1.0, 0.0, 0.0]]).repeat(16, 1)
    features = history[:1].repeat(64, 1)
    features[8:11] = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
    scores, _ = persistent_scores(features, history)
    assert float(scores[8]) == 0


def test_anchors_exact_budget_stable_ties_and_private_rng():
    features = torch.zeros(64, 3)
    rng = torch.get_rng_state().clone()
    chosen, detail = persistent_indices(features, None)
    assert len(chosen) == len(chosen.unique()) == 32
    assert set(ANCHORS.tolist()).issubset(chosen.tolist())
    assert chosen.tolist() == sorted(chosen.tolist())
    assert 63 in chosen and 62 not in detail["content_positions"]
    assert detail["content_positions"] == [i for i in range(62) if i not in ANCHORS.tolist()][:16]
    assert torch.equal(torch.get_rng_state(), rng)
    for policy in ("full", "uniform_half", "random_half"):
        select_window(policy, None, None, torch.Generator().manual_seed(7))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("policy", ("full", "uniform_half", "random_half", "persistent_half"))
def test_future_clip_changes_do_not_change_completed_diet_and_history_resets(policy):
    initial = Toy()
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 0.01})
    torch.manual_seed(19)
    parent = capture(initial, torch.optim.AdamW(initial.parameters(), lr=0.01))
    arrivals = torch.linspace(0, 1, 256).reshape(256, 1, 1, 1)
    rows = []
    for data in (arrivals, torch.cat((arrivals[:128], torch.ones_like(arrivals[128:])))):
        model, optim = restore(initial, config, parent)
        rows.append(run_stream(model, optim, data, policy, copy.deepcopy(initial).eval(), 17))
    expected_updates = 8 if policy == "full" else 4
    assert rows[0]["updates"] == expected_updates
    for index in (0, 1):
        assert rows[0]["blocks"][index]["selected_indices"] == rows[1]["blocks"][index]["selected_indices"]
        assert rows[0]["blocks"][index]["loss_gradients"] == rows[1]["blocks"][index]["loss_gradients"]
    for index, block in enumerate(rows[0]["blocks"]):
        assert all(block["start"] <= i < block["end"] for i in block["selected_indices"])
        if index % 2 == 0:
            assert block["clip_reference_reset"] and block["reference_source_indices"] == []
        elif policy == "persistent_half":
            assert block["reference_source_indices"] == rows[0]["blocks"][index - 1]["selected_indices"][-16:]
    assert fingerprint(parent) == fingerprint(capture(*restore(initial, config, parent)))


def test_invalid_shapes_and_nonfinite_history_fail():
    with pytest.raises(ValueError):
        persistent_scores(torch.ones(63, 3), None)
    with pytest.raises(ValueError):
        persistent_scores(torch.ones(64, 3), torch.full((16, 3), float("nan")))


def test_exact_count_contrasts_preserve_ties_instead_of_float_wins():
    from scripts.run_persistent_novelty_experiment import DRAWS, SEEDS, exact_contrast

    by = {(policy, seed): {"directory": f"{policy}_{seed}"} for policy in ("a", "b") for seed in SEEDS}
    counts = {}
    for seed in SEEDS:
        # Distribution across support draws changes, but total correct is equal.
        for draw, first, second in zip(DRAWS, (201, 202, 203), (200, 204, 202), strict=True):
            counts[f"a_{seed}", draw, "finetune"] = {"balanced": first}
            counts[f"b_{seed}", draw, "finetune"] = {"balanced": second}
    result = exact_contrast(by, counts, "a", "b", "finetune", "balanced", DRAWS)
    assert result["wins"] == 0 and result["ties"] == 5
    assert result["mean_pp"] == 0 and result["ci95_pp"] == [0, 0]
