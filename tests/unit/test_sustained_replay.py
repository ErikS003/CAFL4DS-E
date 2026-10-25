"""Verify dose/content isolation, causality and conservative reuse boundaries."""

import hashlib
import json

import numpy as np
import pytest

from cafl4ds.sustained_replay import ARMS, SEEDS, batches, exposure_summary, memories, summarize
from scripts.run_sustained_replay import committed
from scripts.verify_sustained_replay import independent_plan


def test_independent_selection_matches_and_history_is_past():
    for arm in ARMS:
        plan = batches("source", 5003, arm)
        assert plan == independent_plan("source", 5003, arm)
        for row in plan:
            assert len(set(row["old"] + row["current"])) == 32
            assert max(row["old"] + row["current"]) < row["available"]
            if row["old"]:
                assert max(row["old"]) < (row["step"] // 8) * 512


def test_dose_is_nested_and_content_preserves_current_identities():
    plans = {a: batches("source", 5003, a) for a in ARMS}
    for step in range(192):
        for prefix in ("uniform", "recent"):
            assert plans[prefix + "25"][step]["old"] == plans[prefix + "50"][step]["old"][:8]
        for dose in (25, 50):
            assert plans[f"uniform{dose}"][step]["current"] == plans[f"recent{dose}"][step]["current"]
    assert all(p[:8] == plans["current"][:8] for p in plans.values())


def test_selection_is_in_shared_memory_and_recent_quarter():
    banks = memories("source", 5003)
    for arm in ARMS[1:]:
        plan = batches("source", 5003, arm)
        for b, memory in enumerate(banks):
            chosen = [i for r in plan[8 + b * 8 : 16 + b * 8] for i in r["old"]]
            assert len(chosen) == len(set(chosen))
            assert set(chosen) <= set(memory)
            if arm.startswith("recent"):
                assert set(chosen) <= set(sorted(memory)[-128:])


def test_exposure_accounting_and_strong_age_contrast():
    metrics = {a: exposure_summary(batches("source", 5003, a)) for a in ARMS}
    assert metrics["current"]["old_exposures"] == 0
    assert metrics["current"]["current_exposures"] == 6144
    for dose, old, current in ((25, 1472, 4672), (50, 2944, 3200)):
        for prefix in ("uniform", "recent"):
            row = metrics[f"{prefix}{dose}"]
            assert row["old_exposures"] == old
            assert row["current_exposures"] == current
            assert sum(row["all_exposures_by_clip"]) == 6144
        assert metrics[f"recent{dose}"]["mean_old_age"] < 0.6 * metrics[f"uniform{dose}"]["mean_old_age"]


def test_private_randomness_determinism_and_invalid_action():
    np.random.seed(17)
    before = np.random.get_state()
    first = batches("source", 5003, "recent25")
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert first == batches("source", 5003, "recent25")
    assert first != batches("source", 5009, "recent25")
    with pytest.raises(ValueError, match="Unknown"):
        batches("source", 5003, "oracle")


def test_corrupted_committed_evidence_is_rejected(tmp_path):
    payload = tmp_path / "predictions.npz"
    payload.write_bytes(b"original")
    row = {
        "complete": True,
        "exact_replay": True,
        "artifact_hashes": {str(payload): hashlib.sha256(payload.read_bytes()).hexdigest()},
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(row))
    assert committed(tmp_path) == row
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        committed(tmp_path)


def test_support_reversal_blocks_lead_and_secondary_cannot_be_promoted():
    rows = []
    values = dict(zip(ARMS, (56, 56.5, 57, 57, 65), strict=True))
    for g in range(6):
        for seed in SEEDS:
            for arm, value in values.items():
                panels = [value] * 3 if arm != "recent25" else [56.4, 57.3, 57.3]
                endpoint = {
                    "scores": {r: {d: value for d in ("balanced", "A", "B")} for r in ("finetune", "linear", "knn")},
                    "panel_balanced_pp": dict(zip(("51011", "51017", "51023"), panels, strict=True)),
                }
                rows.append(
                    {
                        "stream": str(g),
                        "seed": seed,
                        "arm": arm,
                        "readouts": {h: endpoint for h in ("before_return", "terminal")},
                    }
                )
    warm = {str(s): {r: {d: 55 for d in ("balanced", "A", "B")} for r in ("finetune", "linear", "knn")} for s in SEEDS}
    result = summarize(rows, warm)
    assert result["gates"]["uniform50"]["development_lead"]
    assert result["gates"]["recent25"]["practical_primary"]
    assert not result["gates"]["recent25"]["source_seed_support_stability"]
    assert not result["gates"]["recent25"]["development_lead"]
    assert "recent50" not in result["gates"]
