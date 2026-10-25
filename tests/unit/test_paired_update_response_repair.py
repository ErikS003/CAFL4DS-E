"""Reject arbitrary restoration mismatches while recognizing the exact recorded alias defect."""

import copy

import pytest
import torch

from scripts.report_paired_update_response_repair import state_hash, verify_restore_marker


def test_exact_mutable_marker_reconstruction_and_rejection() -> None:
    """A different RNG, parent, or arbitrary marker cannot pass the targeted repair."""
    parent = {
        "model": {"w": torch.ones(2)},
        "optimizer": {"step": torch.tensor(18.0)},
        "torch_rng": torch.get_rng_state(),
        "model_training": True,
        "arrival_step": 36,
        "optimizer_steps": 18,
    }
    final = {
        **copy.deepcopy(parent),
        "model": {"w": torch.ones(2) * 2},
        "optimizer": {"step": torch.tensor(54.0)},
        "arrival_step": 72,
        "optimizer_steps": 54,
    }
    alias = {**parent, "model": final["model"], "optimizer": final["optimizer"]}
    assert verify_restore_marker({"restored_fingerprint": state_hash(parent)}, parent, final) == "immutable_parent"
    result = {"restored_fingerprint": state_hash(alias)}
    assert verify_restore_marker(result, parent, final) == "mutable_snapshot_reconstructed"
    changed = copy.deepcopy(parent)
    changed["torch_rng"][0] ^= 1
    with pytest.raises(ValueError, match="exact known"):
        verify_restore_marker(result, changed, final)
    with pytest.raises(ValueError, match="exact known"):
        verify_restore_marker({"restored_fingerprint": "wrong"}, parent, final)
