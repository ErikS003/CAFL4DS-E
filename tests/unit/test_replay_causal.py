"""Guard the causal contrast against baseline subtraction errors and mismatched arms."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.confirm_replay_causal import _arm_metrics, _save_protocol, validate_pair


def test_causal_contrast_uses_adapted_scores() -> None:
    """Different reference scores must not contaminate the direct online-arm contrast."""
    replay = [
        {
            f"{arm}_{metric}": value
            for arm, value in (("adapted", 0.55), ("frozen", 0.4))
            for metric in ("knn", "linear", "final_conditions")
        }
        for _ in range(5)
    ]
    no_replay = [
        {
            f"{arm}_{metric}": value
            for arm, value in (("adapted", 0.50), ("frozen", 0.3))
            for metric in ("knn", "linear", "final_conditions")
        }
        for _ in range(5)
    ]
    result = _arm_metrics(replay, no_replay, "replay_minus_no_replay")
    assert result["knn"]["mean_pp"] == pytest.approx(5)
    assert result["knn"]["n_seeds"] == 5


def test_pair_rejects_learning_rate_confound(tmp_path: Path) -> None:
    """The historic 1e-3/1e-4 confound cannot pass the new pairing check."""
    replay_dir, no_dir = tmp_path / "replay", tmp_path / "no_replay"
    replay_dir.mkdir()
    no_dir.mkdir()
    replay = {
        "seeds": [2003],
        "optim": {"lr": 1e-4},
        "filter": {"_target_": "cafl4ds.filters.reservoir.ReservoirReplay"},
    }
    no = {"seeds": [2003], "optim": {"lr": 1e-4}, "filter": {"_target_": "cafl4ds.filters.accept_all.AcceptAll"}}
    OmegaConf.save(OmegaConf.create(replay), replay_dir / "config.yaml")
    OmegaConf.save(OmegaConf.create(no), no_dir / "config.yaml")
    validate_pair(replay_dir, no_dir, 2003)
    no["optim"] = {"lr": 1e-3}
    OmegaConf.save(OmegaConf.create(no), no_dir / "config.yaml")
    with pytest.raises(ValueError, match="optim"):
        validate_pair(replay_dir, no_dir, 2003)


def test_protocol_resume_preserves_original_and_rejects_changes(tmp_path: Path) -> None:
    """Resuming must neither replace original provenance nor silently accept another well/config."""
    path = tmp_path / "protocol.json"
    original = {"config": {"well": "a"}, "replay_summary_sha256": "b", "protocol_sha256": "c"}
    _save_protocol(path, original)
    before = path.read_bytes()
    _save_protocol(path, {**original, "created_utc": "later"})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="config"):
        _save_protocol(path, {**original, "config": {"well": "other"}})
    assert path.read_bytes() == before
