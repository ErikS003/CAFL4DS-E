import pytest
import torch

from cafl4ds.return_actions import ACTIONS, run_branch
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + torch.rand(()) * 0.01).square()


@pytest.mark.parametrize("action", ACTIONS)
def test_action_budgets_and_prefix_causality(action):
    frames = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
    changed = frames.clone()
    changed[128:] = 0
    snapshots = []
    for arrivals in (frames, changed):
        torch.manual_seed(11)
        model = Toy()
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = fingerprint(capture(model, optim))
        states = {}
        result = run_branch(
            model, optim, arrivals, action, 17, (128, 256), lambda clock, state: states.setdefault(clock, state)
        )
        expected = {"hold": 0, "uniform_half": 4, "full": 8}[action]
        assert result["updates"] == expected
        assert result["accepted_frames"] == 32 * expected
        assert not states[256]["pending_indices"] and not states[256]["queued_indices"]
        if action == "hold":
            assert fingerprint(capture(model, optim)) == before
            assert all(fingerprint(s["learner"]) == before for s in states.values())
        snapshots.append(states)
    assert fingerprint(snapshots[0][128]["learner"]) == fingerprint(snapshots[1][128]["learner"])


def test_hold_preserves_nonempty_adam_rng_and_modes():
    torch.manual_seed(3)
    model = Toy()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(3):
        optim.zero_grad()
        model.training_step(torch.ones(32, 1, 1, 1)).backward()
        optim.step()
    model.eval()
    before = fingerprint(capture(model, optim))
    run_branch(model, optim, torch.ones(256, 1, 1, 1), "hold", 17, (128, 256))
    assert fingerprint(capture(model, optim)) == before


def test_unknown_action():
    with pytest.raises(ValueError):
        run_branch(Toy(), None, torch.ones(256, 1, 1, 1), "bad", 1, (128, 256))
