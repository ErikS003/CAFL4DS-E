import pytest
import torch

from cafl4ds.regime_windows import POLICIES, WIDTHS, live_scores, run_stream, select_indices


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + torch.rand(()) * 0.01).square()

    def per_sample_loss(self, images):
        return images.flatten(1).mean(1) + torch.rand(len(images)) * 0.01


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("policy", POLICIES)
def test_budget_queue_causality_and_partial_tail(policy, width):
    frames = torch.arange(1088).float().reshape(1088, 1, 1, 1) / 1088
    changed = frames.clone()
    changed[544:] = 0
    states = []
    for arrivals in (frames, changed):
        torch.manual_seed(3)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        saved = {}
        result = run_stream(
            model,
            optimizer,
            arrivals,
            policy,
            width,
            17,
            (544, 1088),
            lambda clock, state: saved.setdefault(clock, state),
        )
        assert result["updates"] == (34 if policy == "full" else 17)
        assert len(saved[1088]["pending_indices"]) == len(saved[1088]["queued_indices"]) == 0
        states.append(saved)
    assert torch.equal(states[0][544]["learner"]["model"]["weight"], states[1][544]["learner"]["model"]["weight"])
    assert states[0][544]["window_start"] == states[1][544]["window_start"]
    if width == 512:
        assert states[0][544]["pending_indices"] == list(range(512, 544))
    if width == 32 and policy != "full":
        assert len(states[0][544]["queued_indices"]) == 16


def test_live_scoring_rng_and_modes():
    model = Toy().train()
    before = torch.get_rng_state().clone()
    scores = live_scores(model, torch.zeros(128, 1, 1, 1), 111)
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert torch.equal(scores, live_scores(model, torch.zeros(128, 1, 1, 1), 111))


def test_loss_ties_are_earliest_and_random_private():
    generator = torch.Generator().manual_seed(1)
    before = torch.get_rng_state().clone()
    assert select_indices("live_loss_half", 32, generator, torch.ones(32)).tolist() == list(range(16))
    assert len(select_indices("random_half", 128, generator)) == 64
    assert torch.equal(before, torch.get_rng_state())


def test_exact_count_lattice_keeps_subtraction_ties():
    from scripts.run_regime_windows_experiment import interval

    result = interval([1e-14, -1e-14, 0])
    assert result["ties"] == 3 and result["wins"] == 0
    assert result["mean_pp"] == 0 and result["ci95_pp"] == [0, 0]


def test_source_gap_is_preserved_without_duplicate_frames():
    from scripts.decode_regime_video import timestamps

    log = "\n".join(f"n: {n} pts: {n * 100} pts_time:{seconds} s:1280x720" for n, seconds in enumerate((0, 0.3, 0.36)))
    rows = timestamps(log, count=3)
    assert [r["source_frame_index"] for r in rows] == [0, 1, 2]
    assert rows[1]["slot_delay_seconds"] > 0.1
    assert [r["pts_seconds"] for r in rows] == [0, 0.3, 0.36]


def test_json_replay_normalizes_loss_gradient_tuples(tmp_path):
    import json
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from cafl4ds.artifacts import atomic_json
    from cafl4ds.short_buffer import capture, restore
    from scripts.run_paired_update_response import fingerprint
    from scripts.run_regime_windows_experiment import audit

    config = OmegaConf.create({"optim": {"_target_": "torch.optim.AdamW", "lr": 0.001}})
    initial = Toy()
    torch.manual_seed(5003)
    optimizer = instantiate(config.optim, params=list(initial.parameters()))
    parent = capture(initial, optimizer)
    model, optim = restore(initial, config.optim, parent)
    frames = torch.ones(12672, 1, 1, 1)
    directory = tmp_path / "unit-run"
    directory.mkdir()

    def observe(clock, state):
        checkpoint = directory / f"at_{clock}"
        checkpoint.mkdir()
        torch.save(state, checkpoint / "state.pt")
        return {
            "state_fingerprint": fingerprint(state),
            "updates": state["updates"],
            "pending_frames": len(state["pending_indices"]),
            "queued_frames": len(state["queued_indices"]),
        }

    training = run_stream(model, optim, frames, "full", 32, 5003, observe=observe)
    assert isinstance(training["blocks"][0]["loss_gradients"][0], tuple)
    atomic_json(directory / "training.json", training)
    row = {
        "seed": 5003,
        "policy": "full",
        "width": 32,
        "directory": "unit-run",
        "updates": 396,
        "initial_fingerprint": fingerprint(parent),
    }
    audit(initial, config, frames, [row], tmp_path, lambda **kwargs: None)
    assert json.loads((tmp_path / "training-replay-audit.json").read_text())["passed"]
