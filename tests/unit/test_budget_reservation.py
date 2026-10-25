import pytest
import torch

from cafl4ds.budget_reservation import accepted, checkpoints, run
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint


def test_equal_budget_and_disjoint_opportunities():
    sets = {arm: {k for k in range(0, 1024, 32) if accepted(k, arm)} for arm in ("spend", "reserve", "hold")}
    assert len(sets["spend"]) == len(sets["reserve"]) == 16
    assert not sets["hold"] and not sets["spend"] & sets["reserve"]
    assert sets["spend"] | sets["reserve"] == set(range(0, 1024, 32))


def test_identical_continuation():
    for k in range(1024, 12288, 32):
        assert len({accepted(k, arm) for arm in ("spend", "reserve", "hold")}) == 1


def test_checkpoints_independent_of_outcomes():
    assert checkpoints("DDDNNNDDDNNN") == {"before_shift": 2560, "stable_second": 3072, "before_return": 5632}
    assert checkpoints("DNDNDNDNDNDN") == {"before_shift": 512, "stable_second": 1024, "before_return": 1536}


def test_invalid_alignment():
    with pytest.raises(ValueError):
        accepted(1, "spend")


class TinyLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1)

    def training_step(self, images):
        prediction = self.layer(images.mean((2, 3)))
        return (prediction - torch.rand_like(prediction)).square().mean()


def test_hold_preserves_trained_adam_modes_and_rng():
    model = TinyLearner()
    optimizer = torch.optim.AdamW(model.parameters())
    images = torch.rand(1024, 3, 2, 2)
    run(model, optimizer, images, "spend")
    model.eval()
    before = fingerprint(capture(model, optimizer))
    observed = []
    trace = run(model, optimizer, images, "hold", lambda clock, state: observed.append((clock, fingerprint(state))))
    assert fingerprint(capture(model, optimizer)) == before
    assert observed == [(1024, before)]
    assert all(r["loss_gradient"] is None for r in trace)
