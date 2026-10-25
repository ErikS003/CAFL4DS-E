import pytest
import torch

from cafl4ds.return_actions import run_branch
from cafl4ds.scheduling import HALF, PATTERNS, allocation, choose, interval, qualifies
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


def test_allocation_preserves_disjointness_and_balances_development_and_test():
    clips = [
        {"clip_id": f"{light}_{scene}_{k}", "keyframe_attributes": {"timeofday": light, "scene": scene}}
        for light in ("daytime", "night")
        for scene in ("city street", "highway")
        for k in range(60)
    ]
    streams = allocation(clips)
    assert tuple(s["pattern"] for s in streams) == PATTERNS
    ids = [k for s in streams for period in s["periods"] for k in period]
    assert len(ids) == len(set(ids)) == 240
    assert sum(s["role"] == "development" for s in streams) == 4
    for subset in (streams[:4], streams[4:]):
        assert sum(s["pattern"].count("D") for s in subset) == sum(s["pattern"].count("N") for s in subset)
    with pytest.raises(ValueError):
        allocation(clips[:-1])


def test_schedule_choice_ties_and_family_uncertainty():
    assert len(HALF) == 6 and all(s.count("1") == 2 for s in HALF)
    assert choose(dict.fromkeys(HALF, 1)) == "0011"
    ordinary = interval([0, 1, 2, 3, 4, 5])
    adjusted = interval([0, 1, 2, 3, 4, 5], 15)
    assert adjusted["ci95_pp"][0] < ordinary["ci95_pp"][0]
    effects = {m: {"mean_pp": 1, "ci95_pp": [0.2, 1.8]} for m in ("balanced", "A", "B")}
    assert qualifies(effects)
    effects["B"] = {"mean_pp": -2, "ci95_pp": [-3, -0.5]}
    assert not qualifies(effects)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + torch.rand(()) * 0.01).square()


def test_pause_resume_keeps_adam_rng_and_matches_omitting_paused_arrivals():
    fingerprints = []
    for pause in (False, True):
        torch.manual_seed(17)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        run_branch(model, optimizer, torch.ones(64, 1, 1, 1), "full", 3, (64,))
        before = fingerprint(capture(model, optimizer))
        if pause:
            run_branch(model, optimizer, torch.zeros(64, 1, 1, 1), "hold", 3, (64,))
            assert fingerprint(capture(model, optimizer)) == before
        run_branch(model, optimizer, torch.full((64, 1, 1, 1), 0.5), "full", 3, (64,))
        assert all(int(s["step"]) == 4 for s in optimizer.state.values())
        fingerprints.append(fingerprint(capture(model, optimizer)))
    assert fingerprints[0] == fingerprints[1]
