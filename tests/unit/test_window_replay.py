"""Check temporal membership, memory ownership and RNG of experience replay."""

from types import SimpleNamespace

import pytest
import torch

from cafl4ds.filters.window_replay import WindowReplay


def frames(start):
    """Use identity-valued images so historical membership is directly observable."""
    return torch.arange(start, start + 32, dtype=torch.float32).reshape(32, 1, 1, 1)


def test_fifo_history_before_ingest_and_fixed_batch_size():
    """Memory holds only the most recent admitted window; replay precedes current ingestion."""
    buffer = WindowReplay(capacity=32, seed=3)
    first = buffer.observe(frames(0), SimpleNamespace(step=0))
    assert torch.equal(first, frames(0))
    second = buffer.observe(frames(64), SimpleNamespace(step=2))
    assert second.shape == (32, 1, 1, 1)
    assert (second[:16] >= 64).all() and (second[16:] < 32).all()
    assert len(set(buffer.last["current_ids"])) == len(set(buffer.last["replay_ids"])) == 16
    assert buffer.marker()["ids"] == list(range(64, 96))
    assert buffer.last["replay_arrival_ages"] == [2] * 16
    third = buffer.observe(frames(96), SimpleNamespace(step=3))
    assert (third[16:] >= 64).all() and (third[16:] < 96).all()
    assert buffer.seen == 96


def test_private_rng_and_owned_pixels():
    """Sampling never perturbs learner RNG or aliases a mutable arrival tensor."""
    buffer = WindowReplay(capacity=64, seed=7)
    source = frames(0)
    before = torch.get_rng_state().clone()
    buffer.observe(source, SimpleNamespace(step=0))
    source.fill_(-999)
    selected = buffer.observe(frames(32), SimpleNamespace(step=1))
    assert torch.equal(before, torch.get_rng_state())
    assert (selected[16:] >= 0).all()


def test_save_restore_exact_continuation_after_rejected_arrivals():
    """Unobserved rejected arrivals cannot enter memory; restored sampling is exact."""
    a, b = WindowReplay(64, seed=9), WindowReplay(64, seed=9)
    a.observe(frames(0), SimpleNamespace(step=0))
    # Arrivals one through three are rejected; only the next admitted batch is observed.
    b.load_state_dict(a.state_dict())
    x, y = a.observe(frames(128), SimpleNamespace(step=4)), b.observe(frames(128), SimpleNamespace(step=4))
    assert torch.equal(x, y) and a.last == b.last
    assert a.marker()["ids"] == [*range(32), *range(128, 160)]
    assert a.seen == 64
    bad = a.state_dict()
    bad["images"][0].fill_(999)
    with pytest.raises(ValueError, match="pixels"):
        b.load_state_dict(bad)


@pytest.mark.parametrize("capacity,replay", [(0, 16), (33, 16), (32, 0), (32, 32)])
def test_invalid_window_configuration(capacity, replay):
    """Reject configurations that violate the fixed-batch experiment contract."""
    with pytest.raises(ValueError):
        WindowReplay(capacity, replay)
