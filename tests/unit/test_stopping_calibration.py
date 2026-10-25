import pytest
import torch

from cafl4ds.return_actions import run_branch
from cafl4ds.short_buffer import capture
from cafl4ds.stopping_calibration import BUDGETS, PATTERNS, allocation, choose, gates
from scripts.run_paired_update_response import fingerprint


def test_allocation_balanced_global_sources_and_dwell_scales():
    clips = [
        {"clip_id": f"{light}/{scene}/{k}", "keyframe_attributes": {"timeofday": light, "scene": scene}}
        for light in ("daytime", "night")
        for scene in ("city street", "highway")
        for k in range(36)
    ]
    streams = allocation(clips)
    ids = [c for s in streams for pair in s["periods"] for c in pair]
    assert len(ids) == len(set(ids)) == 144
    assert tuple(s["pattern"] for s in streams) == PATTERNS
    assert PATTERNS[0][:2] == "DN" and PATTERNS[2][:4] == "DDDN" and PATTERNS[4][:7] == "DDDDDDN"
    assert sum(p.count("D") for p in PATTERNS) == sum(p.count("N") for p in PATTERNS)
    with pytest.raises(ValueError):
        allocation(clips[:-1])


def test_budget_choice_prefers_less_work_on_equal_quality():
    assert choose(dict.fromkeys(BUDGETS, 1)) == 0
    values = dict.fromkeys(BUDGETS, 0)
    values[48] = values[192] = 1
    assert choose(values) == 48


def test_transfer_requires_positive_gain_retention_and_measured_work_saving():
    effects = {ref: {m: {"mean_pp": 1, "ci95_pp": [0.1, 2]} for m in ("balanced", "A", "B")} for ref in (0, 192, 384)}
    assert gates(96, effects, 0.5)["combined"]
    assert not gates(96, effects, 0.1)["combined"]
    effects[384]["B"]["ci95_pp"][0] = -1.1
    assert not gates(96, effects, 0.5)["combined"]


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + 0.01 * torch.rand(())).square()


def test_shared_prefix_is_exact_stop_then_hold_counterfactual():
    frames = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
    torch.manual_seed(7)
    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    prefix = {}
    run_branch(model, optimizer, frames, "full", 11, (128, 256), lambda clock, state: prefix.setdefault(clock, state))
    torch.manual_seed(7)
    stopped = Toy()
    opt = torch.optim.AdamW(stopped.parameters(), lr=0.001)
    run_branch(stopped, opt, frames[:128], "full", 11, (128,))
    run_branch(stopped, opt, frames[128:], "hold", 11, (128,))
    assert fingerprint(capture(stopped, opt)) == fingerprint(prefix[128]["learner"])
