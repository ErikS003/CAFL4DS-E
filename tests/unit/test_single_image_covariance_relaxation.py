"""Recovery must preserve complete state and reject partial or unrelated writes."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_covariance_relaxation import recover_publication


def pending_generation(root: Path) -> tuple[dict[str, Any], Path, PermissionError]:
    """Create a small complete state that has not yet been published."""
    lock = {"history": "test", "seed": 5003, "settings": {"arm": "relaxed"}}
    pending = root / "trajectories/test/seed_5003/relaxed/resume_states/step_00000016.pending"
    pending.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    state = capture(model, optimizer)
    torch.save(state, pending / "model.pt")
    np.savez_compressed(pending / "trace.npz", accepted=np.array([True] * 16 + [False] * 16))
    atomic_json(
        pending / "commit.json",
        {
            "done": 16,
            "fingerprint": fingerprint(state),
            "file_hashes": {name: digest(pending / name) for name in ("model.pt", "trace.npz")},
        },
    )
    error = PermissionError(13, "Transient Windows checkpoint rename denial", str(pending))
    error.winerror = 5
    return lock, pending, error


def test_windows_recovery_retries_rename_without_modifying_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient sharing violation must not lose or rewrite learning state."""
    lock, pending, error = pending_generation(tmp_path)
    before = {p.name: digest(p) for p in pending.iterdir()}
    rename = Path.rename
    attempts = []

    def deny_once(path: Path, target: Path) -> Path:
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("Transient sharing violation")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", deny_once)
    monkeypatch.setattr("scripts.run_single_image_covariance_relaxation.time.sleep", lambda _: None)
    recover_publication(tmp_path, lock, error)
    destination = pending.with_name("step_00000016")
    assert not pending.exists()
    assert {p.name: digest(p) for p in destination.iterdir()} == before
    assert len(attempts) == 2


def test_corrupt_pending_state_is_never_published(tmp_path: Path) -> None:
    """Integrity failures must leave the generation unpublished."""
    lock, pending, error = pending_generation(tmp_path)
    (pending / "model.pt").write_bytes(b"incomplete checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        recover_publication(tmp_path, lock, error)
    assert pending.exists()
    assert not pending.with_name("step_00000016").exists()


def test_unrelated_permission_errors_are_not_reinterpreted(tmp_path: Path) -> None:
    """Bound recovery to the exact authorized checkpoint directory."""
    lock, _, _ = pending_generation(tmp_path)
    error = PermissionError(13, "Unrelated permission failure", str(tmp_path / "other.pending"))
    error.winerror = 5
    with pytest.raises(PermissionError):
        recover_publication(tmp_path, lock, error)
