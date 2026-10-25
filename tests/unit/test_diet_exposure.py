import pytest
import torch

from cafl4ds.diet_exposure import POLICIES, run_stream, select_window


@pytest.mark.parametrize("kind", ["uniform", "random"])
def test_once_twice_same_diet_private_rng(kind):
    before = torch.get_rng_state().clone()
    once = torch.Generator().manual_seed(123)
    twice = torch.Generator().manual_seed(123)
    for _ in range(96):
        a = select_window(kind + "_half", once)
        b = select_window(kind + "_twice", twice)
        assert torch.equal(a, b)
        assert len(a) == 32 and a.tolist() == sorted(set(a.tolist()))
    assert torch.equal(before, torch.get_rng_state())


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, images):
        return (self.weight * images.mean() + torch.rand(()) * 0.01).square()


@pytest.mark.parametrize("policy", POLICIES)
def test_exposures_budget_and_causal_prefix(policy):
    frames = torch.arange(256).float().reshape(256, 1, 1, 1) / 256
    changed = frames.clone()
    changed[128:] = 0
    records = []
    for arrivals in (frames, changed):
        torch.manual_seed(9)
        model = Toy()
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        records.append(run_stream(model, optim, arrivals, policy, None, 123))
    expected = 8 if policy == "full" or policy.endswith("twice") else 4
    assert records[0]["updates"] == expected
    assert records[0]["training_exposures"] == expected * 32
    for a, b in zip(records[0]["blocks"][:2], records[1]["blocks"][:2], strict=True):
        assert a["selected_indices"] == b["selected_indices"]
        assert a["loss_gradients"] == b["loss_gradients"]
        assert all(a["fresh_stochastic_draws"])
        if policy.endswith("twice"):
            assert a["exposure_indices"][0] == a["exposure_indices"][1]


def test_unknown_policy():
    with pytest.raises(ValueError):
        select_window("bad", torch.Generator())
