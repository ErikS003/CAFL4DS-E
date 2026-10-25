"""Verify causal memory actuation, control compatibility and independent reconstruction."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.memory_admission import MemoryAdmissionReplay
from cafl4ds.filters.reservoir import ReservoirReplay
from cafl4ds.models.vit import TinyViTEncoder
from scripts.report_memory_admission import membership


def reference() -> TinyViTEncoder:
    """Make a real small encoder with the same descriptor dimension as the locked study."""
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=96, depth=1, num_heads=3)


def test_traced_reservoir_exactly_matches_original_rng_and_images() -> None:
    """Instrumentation must not alter the reused control policy after capacity fills."""
    torch.set_num_threads(1)
    torch.manual_seed(23)
    encoder = reference()
    traced = MemoryAdmissionReplay("reservoir", 5003)
    original = ReservoirReplay(256, 16, 5003, max_train_batch=32)
    images = torch.rand(640, 3, 16, 16)
    global_state = torch.get_rng_state().clone()
    for start in range(0, len(images), 32):
        batch = images[start : start + 32]
        observed = traced.observe_with_ids(batch, torch.arange(start, start + 32))
        expected = original.observe(batch, FilterContext(encoder, start // 32))  # type: ignore[arg-type]
        assert torch.equal(observed, expected)
        assert torch.equal(traced.generator.get_state(), original._gen.get_state())
        assert all(identity < start for identity in traced.last_trace["replay"])
        assert torch.equal(observed, images[traced.last_trace["selected_current"] + traced.last_trace["replay"]])
    assert torch.equal(global_state, torch.get_rng_state())
    assert traced.seen == 640 and len(traced.identities) == 256


def test_minred_ties_discard_newest_and_preserve_frozen_reference() -> None:
    """Identical images cannot evict older identical representatives under the frozen tie rule."""
    torch.set_num_threads(1)
    encoder = reference()
    weights = copy.deepcopy(encoder.state_dict())
    buffer = MemoryAdmissionReplay("minred", 3, encoder, capacity=16)
    images = torch.ones(32, 3, 16, 16)
    state = torch.get_rng_state().clone()
    buffer.observe_with_ids(images, torch.arange(32))
    assert buffer.identities == list(range(16))
    buffer.observe_with_ids(images, torch.arange(32, 64))
    assert buffer.identities == list(range(16))
    assert all(event["arrival"] == event["evicted"] for event in buffer.last_trace["events"])
    assert all(torch.equal(value, weights[key]) for key, value in encoder.state_dict().items())
    assert torch.equal(state, torch.get_rng_state())
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    assert buffer.persistent_bytes()["descriptors"] == 16 * 96 * 8


def test_independent_numpy_reconstruction_and_corrupt_ledger_rejection() -> None:
    """Past membership, current RNG scheduling and evictions must independently reproduce."""
    torch.set_num_threads(1)
    torch.manual_seed(31)
    buffer = MemoryAdmissionReplay("minred", 5011, reference())
    reservoir = MemoryAdmissionReplay("reservoir", 5011)
    images = torch.rand(512, 3, 16, 16)
    ledger, raw = [], []
    for start in range(0, len(images), 32):
        ids = torch.arange(start, start + 32)
        selected = buffer.observe_with_ids(images[start : start + 32], ids)
        reservoir.observe_with_ids(images[start : start + 32], ids)
        assert len(selected) == 32
        assert buffer.last_trace["selected_current"] == reservoir.last_trace["selected_current"]
        assert torch.equal(buffer.generator.get_state(), reservoir.generator.get_state())
        ledger.append({"step": start // 32 + 1, **copy.deepcopy(buffer.last_trace)})
        raw.append(buffer.last_features.numpy())
    membership(list(range(512)), np.concatenate(raw), ledger, 5011)
    ledger[-1]["memory"][0] = 99999
    with pytest.raises(ValueError, match="admission differs"):
        membership(list(range(512)), np.concatenate(raw), ledger, 5011)


def test_invalid_policy_and_nonfinite_reference_rejected() -> None:
    """Bad geometry must stop rather than silently switch to a different policy."""
    with pytest.raises(ValueError, match="frozen reference"):
        MemoryAdmissionReplay("minred", 1)
    buffer = MemoryAdmissionReplay("minred", 1, reference())
    with pytest.raises(ValueError, match="Invalid reference"):
        buffer.observe_with_ids(torch.full((32, 3, 16, 16), float("nan")), torch.arange(32))
