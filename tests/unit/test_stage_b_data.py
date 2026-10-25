"""Ensure Stage B's replay is causal and optimizer work is held fixed."""

import pytest
import torch

from cafl4ds.stage_b_data import IndexReservoir, online_lr, repeated_batches


def test_reservoir_is_causal_and_does_not_consume_global_rng() -> None:
    """Every replay image predates the decision; admission remains bounded."""
    rng = torch.random.get_rng_state().clone()
    memory = IndexReservoir(41)
    history = set()
    for step in range(24):
        incoming = list(range(step * 32, (step + 1) * 32))
        record = memory.batch(incoming)
        assert set(record["replay"]) <= history
        assert not set(record["replay"]) & set(incoming)
        assert len(record["current"]) + len(record["replay"]) == 32
        assert len(record["buffer_after"]) <= 256
        assert set(record["buffer_after"]) <= history | set(incoming)
        history.update(incoming)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="single-pass"):
        memory.batch(list(range(32)))


def test_schedule_has_fixed_endpoints_and_diagnostic_recurrence_is_explicit() -> None:
    """Warm-up recurrence must not silently change the online schedule."""
    assert online_lr(0) == 2.5e-6
    assert online_lr(3) == 1e-5
    assert online_lr(23) == 1e-6
    batches = repeated_batches(576, 64, 17)
    assert len(batches) == 64 and all(len(b) == 32 for b in batches)
    first_epoch = [i for batch in batches[:18] for i in batch]
    assert len(set(first_epoch)) == 576
    assert batches == repeated_batches(576, 64, 17)
