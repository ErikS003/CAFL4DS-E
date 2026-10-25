"""Behavioral checks for age/cluster selection and exact counterfactual control."""

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.structured_replay import structured_choices
from cafl4ds.training_choice import FixedTrainingChoice


def test_rules_preserve_budget_and_balance_age_and_clusters() -> None:
    """Age quartiles and distinct clusters contribute equal replay quotas."""
    generator = torch.Generator().manual_seed(7)
    centers = torch.eye(8, dtype=torch.float64)
    z = centers.repeat_interleave(8, dim=0) + torch.randn(64, 8, generator=generator) * 0.001
    arrivals = torch.randperm(64, generator=generator).tolist()
    selections, labels = structured_choices(arrivals, z)
    for indices in selections.values():
        assert len(indices) == len(set(indices)) == 16
        assert indices == sorted(indices)
    assert {arrivals[i] for i in selections["recent"]} == set(range(48, 64))
    assert [sum(arrivals[i] // 16 == q for i in selections["age_balanced"]) for q in range(4)] == [4] * 4
    assert [sum(labels[i] == c for i in selections["cluster_balanced"]) for c in range(8)] == [2] * 8
    assert structured_choices(arrivals, z) == (selections, labels)


def test_first_full_group_change_preserves_future_memory_and_rng() -> None:
    """Changing replay membership leaves current exposure and future draws unchanged."""
    ordinary = ReplayRatio(16, seed=5)
    ordinary.reservoir._buffer = [torch.full((1, 2, 2), float(i)) for i in range(256)]
    ordinary.reservoir._seen = 320
    saved = {"buffer": ordinary.reservoir._buffer, "seen": 320, "rng": ordinary.reservoir._gen.get_state()}
    current = torch.randperm(32, generator=torch.Generator().manual_seed(5 + 104729 * 11))[:16].sort().values.tolist()
    changed = FixedTrainingChoice({"current": current, "replay": list(range(32, 48))}, 10, 5)
    changed.restore_memory(saved)
    for offset in range(6):
        batch = StreamBatch(
            torch.arange(128).float().reshape(32, 1, 2, 2) + 1000 + offset * 128, era=0, step=10 + offset
        )
        context = FilterContext(None, 10 + offset)  # type: ignore[arg-type]
        original = ordinary.select(batch, context)
        replacement = changed.select(batch, context)
        assert torch.equal(torch.stack(ordinary.reservoir._buffer), torch.stack(changed.reservoir._buffer))
        assert torch.equal(ordinary.reservoir._gen.get_state(), changed.reservoir._gen.get_state())
        if offset:
            assert torch.equal(original, replacement)
        else:
            assert torch.equal(original[:16], replacement[:16])
            assert not torch.equal(original[16:], replacement[16:])


def test_cluster_shortages_redistribute_without_repeating_images() -> None:
    """Singleton groups remain represented and the populated cluster fills the budget."""
    centers = torch.eye(8, dtype=torch.float64)
    z = torch.cat((centers[:1].repeat(57, 1), centers[1:]))
    selections, labels = structured_choices(list(range(64)), z)
    selected = selections["cluster_balanced"]
    assert len(selected) == len(set(selected)) == 16
    assert all(i in selected for i in range(57, 64))
    assert sum(labels[i] == labels[0] for i in selected) == 9
