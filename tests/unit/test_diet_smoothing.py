import pytest
import torch

from cafl4ds.diet_smoothing import historical_count, run_branch, set_history
from scripts.run_paired_update_response import fingerprint
from tests.unit.test_return_actions import Toy


def test_schedules_match_historical_dose_and_taper():
    smooth = [historical_count(i, "smooth") for i in range(132)]
    abrupt = [historical_count(i, "abrupt") for i in range(132)]
    assert sum(smooth) == sum(abrupt) == 272
    assert all(a >= b for a, b in zip(smooth, smooth[1:]))
    assert smooth[32:] == [0] * 100


@pytest.mark.parametrize("action", ["smooth", "abrupt"])
def test_budget_replay_and_causality(action):
    set_history(torch.ones(512, 1, 1, 1))
    frames = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
    signatures = []
    for changed in (False, True):
        torch.manual_seed(11)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        arrivals = frames.clone()
        if changed:
            arrivals[128:] = 0
        states = {}
        result = run_branch(
            model, optimizer, arrivals, action, 17, (128, 256), lambda clock, state: states.setdefault(clock, state)
        )
        assert result["updates"] == 8
        for record in result["blocks"]:
            assert len(record["exposure_indices"][0]) == 32
            assert all(record["start"] <= i < record["end"] for i in record["selected_indices"])
        signatures.append(fingerprint(states[128]))
    assert signatures[0] == signatures[1]


def test_complete_mixing_exposure_and_exact_reproduction():
    set_history(torch.arange(512).float().reshape(512, 1, 1, 1) / 512)
    frames = torch.ones(4224, 1, 1, 1)
    old_sequences = []
    for action in ("smooth", "abrupt"):
        signatures = []
        for _ in range(2):
            torch.manual_seed(7)
            model = Toy()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            states = {}
            result = run_branch(
                model, optimizer, frames, action, 17, (4224,), lambda clock, state: states.setdefault(clock, state)
            )
            assert result["updates"] == 132 and result["accepted_frames"] == 3952
            old = [i for b in result["blocks"] for i in b["history_indices"]]
            current = [i for b in result["blocks"] for i in b["selected_indices"]]
            assert len(old) == len(set(old)) == 272
            assert len(current) == len(set(current)) == 3952
            signatures.append(fingerprint(states[4224]))
        assert signatures[0] == signatures[1]
        old_sequences.append(old)
    assert old_sequences[0] == old_sequences[1]
