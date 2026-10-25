import pytest
import torch

from cafl4ds.live_loss import loss_indices, score_window


def test_earliest_ties_and_chronological_training():
    assert loss_indices(torch.ones(64)).tolist() == list(range(32))
    assert loss_indices(torch.arange(64).float()).tolist() == list(range(32, 64))


@pytest.mark.parametrize("scores", [torch.ones(63), torch.full((64,), float("nan"))])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        loss_indices(scores)


class RandomScorer(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.child = torch.nn.Linear(1, 1)
        self.child.eval()
        self.fail = fail

    def per_sample_loss(self, images):
        assert not self.training
        if self.fail:
            raise RuntimeError("deliberate")
        return torch.rand(len(images))


def test_scoring_rng_modes_and_repeatability():
    model = RandomScorer()
    before = torch.get_rng_state().clone()
    scores = score_window(model, torch.zeros(64, 1), 123)
    assert torch.equal(before, torch.get_rng_state())
    assert model.training and not model.child.training
    assert torch.equal(scores, score_window(model, torch.zeros(64, 1), 123))
    assert not torch.equal(scores, score_window(model, torch.zeros(64, 1), 124))


def test_exception_restores_modes_and_rng():
    model = RandomScorer(fail=True)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        score_window(model, torch.zeros(64, 1), 123)
    assert model.training and not model.child.training
    assert torch.equal(before, torch.get_rng_state())
