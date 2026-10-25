import numpy as np
import pytest
import torch

from cafl4ds.coverage_admission import ARMS, run_admission, select_coverage
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


def test_coverage_rejects_redundancy_and_commits_chronological_history():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    selected, rounds, memory = select_coverage(z, [np.array([1.0, 0.0])])
    assert selected == [1, 2] and rounds[0][0] == 0 and rounds[0][2] == 2
    assert np.allclose(memory[-2:], z[selected])
    _, _, bounded = select_coverage(z, [np.array([1.0, 0.0])] * 32)
    assert len(bounded) == 32
    with pytest.raises(ValueError):
        select_coverage(np.zeros((4, 2)), [])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + torch.rand(()) * 0.01).square()

    def per_sample_loss(self, images):
        return (self.weight * images.flatten(1).mean(1) + torch.rand(len(images)) * 0.01).square()

    def embed(self, images):
        value = images.flatten(1).mean(1)
        return torch.stack((value, torch.ones_like(value)), 1)


@pytest.mark.parametrize("arm", ARMS)
def test_window_causality_and_budget(arm):
    snapshots = []
    for changed in (False, True):
        torch.manual_seed(17)
        model, ref = Toy(), Toy().eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        frames = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
        if changed:
            frames[128:] = 1
        states = {}
        result = run_admission(
            model, optimizer, frames, arm, ref, 3, (128, 256), lambda clock, state: states.setdefault(clock, state)
        )
        assert result["updates"] == (8 if arm == "early_stop" else 4)
        snapshots.append(states[128])
    assert fingerprint(snapshots[0]) == fingerprint(snapshots[1])


def test_diagnostics_do_not_change_training_rng_or_updates():
    states = []
    for diagnostics in (True, False):
        torch.manual_seed(9)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        run_admission(model, optimizer, torch.ones(128, 1, 1, 1), "uniform", None, 3, (128,), diagnostics=diagnostics)
        states.append(capture(model, optimizer))
    assert fingerprint(states[0]) == fingerprint(states[1])
