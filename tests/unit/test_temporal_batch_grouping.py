"""Tests of the grouping intervention and the boundary of crash-safe reuse."""

from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from cafl4ds.temporal_batch_grouping import ARMS, batches, manipulation
from scripts.run_temporal_batch_grouping import committed


def test_exposures_and_replay_order_match_in_every_block() -> None:
    plans = {a: batches("source", 5003, a) for a in ARMS}
    for start in range(8, 192, 8):
        expected = Counter(i for r in plans["chronological"][start : start + 8] for i in r["current"])
        for a in ("random", "stratified"):
            assert Counter(i for r in plans[a][start : start + 8] for i in r["current"]) == expected
            assert [r["old"] for r in plans[a][start : start + 8]] == [
                r["old"] for r in plans["chronological"][start : start + 8]
            ]
        assert sum(len(r["old"]) for r in plans["stratified"][start : start + 8]) == 64


def test_all_indices_are_available_and_history_is_strictly_past() -> None:
    for a in ARMS:
        for r in batches("source", 5003, a):
            assert len(set(r["current"] + r["old"])) == 32
            assert max(r["current"] + r["old"]) < r["available"]
            if r["old"]:
                assert max(r["old"]) < r["available"] - 512


def test_stratification_has_three_current_images_from_each_stratum() -> None:
    for r in batches("source", 5003, "stratified")[8:]:
        counts = Counter(i // 64 for i in r["current"])
        assert len(counts) == 8 and set(counts.values()) == {3}
        assert manipulation(r)["occupied_strata"] == 8


def test_private_randomness_and_common_calibration() -> None:
    np.random.seed(37)
    before = np.random.get_state()
    first = batches("source", 5003, "random")
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert first == batches("source", 5003, "random")
    assert first != batches("source", 5009, "random")
    assert all(batches("source", 5003, a)[:8] == first[:8] for a in ARMS)


def test_unknown_grouping_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        batches("source", 5003, "oracle")


def test_committed_resume_refuses_changed_evidence(tmp_path) -> None:
    payload = tmp_path / "predictions.npz"
    payload.write_bytes(b"original evidence")
    row = {
        "complete": True,
        "exact_replay": True,
        "artifact_hashes": {str(payload): hashlib.sha256(payload.read_bytes()).hexdigest()},
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(row), encoding="utf-8")
    assert committed(tmp_path) == row
    payload.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="artifact changed"):
        committed(tmp_path)
