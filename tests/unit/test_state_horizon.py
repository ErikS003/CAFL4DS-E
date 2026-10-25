import pytest
import torch

from cafl4ds.short_buffer import capture
from cafl4ds.state_horizon import observe_before_action
from scripts.run_paired_update_response import fingerprint
from tests.unit.test_return_actions import Toy


class Encoder(torch.nn.Module):
    def embed(self, images):
        return torch.cat((images.flatten(1), torch.ones(len(images), 1)), dim=1)


class ReaderToy(Toy):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def per_sample_loss(self, images):
        return images.flatten(1).mean(1) * self.weight + torch.rand(len(images))


def test_observer_preserves_modes_rng_and_adam_and_reproduces():
    model = ReaderToy()
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.training_step(torch.ones(32, 1, 1, 1)).backward()
    optim.step()
    model.eval()
    model.encoder.train()
    before = fingerprint(capture(model, optim))
    results = [
        observe_before_action(model, torch.ones(128, 1, 1, 1), torch.zeros(32, 1, 1, 1), 0.5, 17) for _ in range(2)
    ]
    assert fingerprint(capture(model, optim)) == before
    assert {k: v for k, v in results[0].items() if k != "observer_seconds"} == {
        k: v for k, v in results[1].items() if k != "observer_seconds"
    }
    assert results[0]["feature_centroid_distance_l2"] > 0


def test_observer_requires_only_fixed_arrived_batch():
    with pytest.raises(ValueError):
        observe_before_action(ReaderToy(), torch.ones(128, 1, 1, 1), torch.ones(64, 1, 1, 1), 0.5, 17)
