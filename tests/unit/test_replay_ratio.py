"""Replay allocation must change exposure without changing history or its RNG."""

import copy

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.replay_ratio import ReplayRatio


def test_nested_draws_and_action_independent_history() -> None:
    """All actions receive equal work and share nested samples and reservoir updates."""
    initial = ReplayRatio(seed=13, capacity=32)
    for i in range(40):
        initial.reservoir._ingest(torch.full((3, 2, 2), float(i)))
    arms = [copy.deepcopy(initial) for _ in range(3)]
    for arm, replay in zip(arms, (8, 16, 24), strict=True):
        arm.set_action(replay)
    for step in range(3):
        images = torch.arange(32.0).reshape(32, 1, 1, 1).expand(32, 3, 2, 2) + 100 * (step + 1)
        for arm in arms:
            # Context only supplies step; method is deliberately unavailable.
            arm.select(StreamBatch(images, era=0, step=step), type("Context", (), {"step": step})())
            assert arm.last["current_count"] + arm.last["replay_count"] == 32
        assert (
            set(arms[2].last["current_indices"])
            <= set(arms[1].last["current_indices"])
            <= set(arms[0].last["current_indices"])
        )
        assert arms[0].last["replay_indices"] == arms[2].last["replay_indices"][:8]
        assert arms[1].last["replay_indices"] == arms[2].last["replay_indices"][:16]
        for arm in arms[1:]:
            assert torch.equal(torch.stack(arm.reservoir._buffer), torch.stack(arms[0].reservoir._buffer))
            assert torch.equal(arm.reservoir._gen.get_state(), arms[0].reservoir._gen.get_state())


def test_selection_does_not_consume_training_rng() -> None:
    """Private current/replay random choices leave augmentation RNG untouched."""
    arm = ReplayRatio(capacity=32)
    for i in range(32):
        arm.reservoir._ingest(torch.full((3, 2, 2), float(i)))
    before = torch.get_rng_state().clone()
    arm.select(StreamBatch(torch.zeros(32, 3, 2, 2), era=0, step=3), type("Context", (), {"step": 3})())
    assert torch.equal(before, torch.get_rng_state())
