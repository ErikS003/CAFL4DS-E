"""Check independent scoring and fingerprint compatibility with research artifacts."""

import numpy as np
import pytest
import torch

from scripts.audit_recurrence_retention import close, interval, order_hash, scores
from scripts.confirm_last_block import tensor_hash


def test_independent_balancing_and_reject_corrupt_predictions() -> None:
    """Unequal class/domain sizes must not become micro accuracy."""
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    domains = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = scores(np.zeros(8, dtype=int), labels, domains)
    assert result == {"A": 0.5, "B": 0.5, "balanced": 0.5, "micro": 0.5}
    with pytest.raises(ValueError, match="shape/classes"):
        scores(np.full(8, 3), labels, domains)
    with pytest.raises(ValueError, match="Independent mismatch"):
        close(result["balanced"], 0.75, "tampered readout")


def test_seed_interval_and_locked_tensor_fingerprint() -> None:
    """Support draws cannot increase n; shape and dtype are part of original hashes."""
    assert interval([0.01] * 5)["ci95_pp"] == [1.0, 1.0]
    with pytest.raises(ValueError, match="five seeds"):
        interval([0.01] * 15)
    order = list(range(18432))
    assert order_hash(order) == tensor_hash(torch.tensor(order))
