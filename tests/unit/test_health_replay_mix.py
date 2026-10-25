"""Check causal recovery stages and variable replay allocation with every-arrival ingestion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cafl4ds.filters.health_replay_mix import HealthReplayMix, VariableWindowReplay


def frames(step):
    """Encode frame identity directly in small image pixels."""
    return torch.arange(step * 32, (step + 1) * 32).reshape(32, 1, 1, 1).float()


def test_alarm_recovery_boundaries_and_retrigger(monkeypatch):
    """A confirmed alarm affects the next update; another alarm restarts the stages."""
    control = HealthReplayMix()
    assert control.split(1) == (32, 0)
    control.observe(132, np.ones((4, 3)))
    monkeypatch.setattr("cafl4ds.natural_health_rate.reading", lambda *_: np.array([0.2, 0.0, 0.0]))
    assert not control.observe(140, np.ones((4, 3)))
    assert control.split(141) == (32, 0)
    assert control.observe(148, np.ones((4, 3)))
    expected = {149: (8, 24), 180: (8, 24), 181: (16, 16), 212: (16, 16), 213: (24, 8), 244: (24, 8), 245: (32, 0)}
    for step, mix in expected.items():
        assert control.split(step) == mix
    assert not control.observe(156, np.ones((4, 3)))
    assert control.observe(164, np.ones((4, 3)))
    assert control.split(165) == (8, 24)
    assert control.credit == 0


def test_all_current_ingests_without_sampling_and_alarm_replay_precedes_eviction():
    """Retain all eight preceding batches; heavy replay cannot see the current arrival."""
    memory = VariableWindowReplay()
    global_before, private_before = torch.get_rng_state().clone(), memory.marker()["rng"]
    for step in range(8):
        assert torch.equal(memory.observe(frames(step), SimpleNamespace(step=step), 0), frames(step))
    assert torch.equal(private_before, memory.marker()["rng"])
    mixed = memory.observe(frames(8), SimpleNamespace(step=8), 24)
    assert mixed.shape[0] == 32
    assert len(memory.last["current_ids"]) == 8 and len(set(memory.last["replay_ids"])) == 24
    assert (mixed[:8] >= 256).all() and (mixed[8:] < 256).all()
    assert memory.marker()["ids"] == list(range(32, 288))
    assert memory.seen == 288
    assert torch.equal(global_before, torch.get_rng_state())


def test_variable_allocation_restores_exactly_and_owns_all_arrivals():
    """Save/restore preserves future mixtures after changing the replay share."""
    first, second = VariableWindowReplay(), VariableWindowReplay()
    source = frames(0)
    first.observe(source, SimpleNamespace(step=0), 0)
    source.fill_(-999)
    first.observe(frames(1), SimpleNamespace(step=1), 24)
    second.load_state_dict(first.state_dict())
    for step, replay in enumerate((0, 8, 16, 24, 0), 2):
        a = first.observe(frames(step), SimpleNamespace(step=step), replay)
        b = second.observe(frames(step), SimpleNamespace(step=step), replay)
        assert torch.equal(a, b) and first.last == second.last
        assert (a >= 0).all()


@pytest.mark.parametrize("replay", [-1, 1, 32])
def test_invalid_allocation(replay):
    """Only the four declared fixed-budget mixtures are available."""
    with pytest.raises(ValueError, match="allocation"):
        VariableWindowReplay().observe(frames(0), SimpleNamespace(step=0), replay)
