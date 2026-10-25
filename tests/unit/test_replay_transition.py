"""Ordering changes preserve exposure and implement the intended transition factor."""

import pytest
import torch

from cafl4ds.replay_transition import schedules


def test_same_images_batches_and_work_across_schedules() -> None:
    """Four interventions differ only in their batch order."""
    domains = torch.tensor([0] * 192 + [1] * 192)
    original = torch.arange(384).reshape(12, 32)[torch.tensor([0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5])].flatten()
    result = schedules(original, domains)
    expected = {
        "day_then_night": (1, 1),
        "night_then_day": (1, 0),
        "alternating_day_first": (11, 1),
        "alternating_night_first": (11, 0),
    }
    for name, order in result.items():
        assert torch.equal(order.sort().values, original.sort().values)
        assert {tuple(b.tolist()) for b in order.reshape(12, 32)} == {
            tuple(b.tolist()) for b in original.reshape(12, 32)
        }
        batch_domains = domains[order.reshape(12, 32)[:, 0]]
        switches = int((batch_domains[1:] != batch_domains[:-1]).sum())
        assert (switches, int(batch_domains[-1])) == expected[name]
    with pytest.raises(ValueError):
        schedules(torch.zeros(384, dtype=torch.long), domains)
