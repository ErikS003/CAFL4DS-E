"""Protect causal selection, paired RNG, and branch restoration on CPU and CUDA."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.short_buffer import capture, diverse_indices, restore, run_episode
from scripts.run_paired_update_response import fingerprint


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def training_step(self, images):
        return (self.weight * images.mean() - torch.rand((), device=images.device)).square()

    def embed(self, images):
        return images.flatten(1)

    @property
    def encoder(self):
        return self


def test_diversity_ties_and_normalization():
    assert diverse_indices(torch.zeros(8, 3), 4).tolist() == [0, 1, 2, 3]
    values = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert diverse_indices(values, 2).tolist() == [0, 3]


@pytest.mark.parametrize("arm", ["full", "uniform_half", "buffered_random_half", "buffered_diverse_half"])
def test_exact_work_causality_and_prefix_invariance(arm):
    initial = Toy()
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 0.01})
    optimizer = torch.optim.AdamW(initial.parameters(), lr=0.01)
    torch.manual_seed(7)
    parent = capture(initial, optimizer)
    arrivals = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
    records = []
    for data in (arrivals, torch.cat((arrivals[:128], torch.ones_like(arrivals[128:])))):
        model, optim = restore(initial, config, parent)
        records.append(run_episode(model, optim, data, arm, copy.deepcopy(initial).eval(), 17))
    assert records[0]["updates"] == (8 if arm == "full" else 4)
    assert records[0]["accepted_frames"] == (256 if arm == "full" else 128)
    first = [r["selected_indices"] for r in records[0]["blocks"] if r["end"] <= 128]
    second = [r["selected_indices"] for r in records[1]["blocks"] if r["end"] <= 128]
    assert first == second
    for block in records[0]["blocks"]:
        assert len(set(block["selected_indices"])) == block["accepted_frames"]
        assert all(block["start"] <= i < block["end"] for i in block["selected_indices"])
    assert fingerprint(parent) == fingerprint(capture(*restore(initial, config, parent)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_restoration_isolates_adam_modes_and_stochastic_continuation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = Toy().to(device)
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 0.01})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    arrivals = torch.ones(128, 1, 1, 1)
    run_episode(model, optimizer, arrivals, "full", model, 17)
    model.eval()
    parent = capture(model, optimizer)
    marker = fingerprint(parent)
    terminals = []
    for _ in range(2):
        branch, optim = restore(model, config, parent)
        assert fingerprint(capture(branch, optim)) == marker
        run_episode(branch, optim, arrivals, "full", model, 17)
        terminals.append(fingerprint(capture(branch, optim)))
    assert terminals[0] == terminals[1]
    assert terminals[0] != marker
    assert fingerprint(parent) == marker


def test_saved_json_loss_pairs_pass_replay_and_corrupt_values_fail(tmp_path):
    """JSON list conversion must pass while changed saved losses still fail auditing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import json

    from scripts.run_short_buffer_experiment import audit_new_trajectories

    initial = Toy().cuda()
    config = OmegaConf.create({"optim": {"_target_": "torch.optim.AdamW", "lr": 0.01}})
    torch.manual_seed(5003)
    optimizer = torch.optim.AdamW(initial.parameters(), lr=0.01)
    parent = capture(initial, optimizer)
    model, optimizer = restore(initial, config.optim, parent)
    arrivals = torch.ones(1536, 1, 1, 1)
    training = run_episode(model, optimizer, arrivals, "full", copy.deepcopy(initial).eval(), 5003)
    state = {
        "learner": capture(model, optimizer),
        "selector_rng": training.pop("selector_rng"),
        "buffer": [],
        "arrivals": 1536,
        "updates": 48,
        "accepted_frames": 1536,
    }
    directory = tmp_path / "run"
    directory.mkdir()
    torch.save(state, directory / "terminal.pt")
    (directory / "training.json").write_text(json.dumps(training))
    record = {
        "bundle": 0,
        "seed": 5003,
        "arm": "full",
        "relative_directory": "run",
        "initial_fingerprint": fingerprint(parent),
        "terminal_fingerprint": fingerprint(state),
    }
    result = audit_new_trajectories(initial, config, [arrivals], [record], tmp_path, lambda **_kwargs: None)
    assert result["passed"] and result["audit_ssl_updates"] == 48
    training["blocks"][0]["loss_gradients"] = [[-99.0, 0.0]]
    (directory / "training.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="loss_gradients"):
        audit_new_trajectories(initial, config, [arrivals], [record], tmp_path, lambda **_kwargs: None)
