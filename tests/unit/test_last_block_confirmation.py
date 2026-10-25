"""Check count-only selection and fresh-seed confirmation reductions without BDD data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from scripts.confirm_last_block import (
    PANELS,
    SEEDS,
    eligible_conditions,
    lock_json,
    read_arm,
    reduce_rows,
    retention,
    summarize,
    validate_pair,
)


def test_count_rule_requires_support_in_every_panel() -> None:
    """Neither a tiny cell nor a single-class condition qualifies as well-sampled."""
    panels = {
        str(p): {
            "conditions": {
                "0": {"n": 30, "class_counts": {"0": 25, "1": 5}},
                "1": {"n": 60, "class_counts": {"0": 60}},
                "2": {"n": 29 if p == PANELS[-1] else 30, "class_counts": {"0": 15, "1": 15}},
            }
        }
        for p in PANELS
    }
    assert eligible_conditions(panels) == [0]


def test_retention_preserves_all_timepoints_when_filtering_columns() -> None:
    """An excluded final condition must not remove the actual final checkpoint."""
    matrix = {"0": {"0": 0.5}, "1": {"0": 0.7, "1": 0.9}, "2": {"0": 0.4, "1": 0.8, "2": 0.1}}
    result = retention(matrix, [0])
    assert result["forgetting"] == pytest.approx(0.3)
    assert result["bwt"] == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="Insufficient"):
        retention(matrix, [])


def test_panel_readout_and_five_seed_decision(tmp_path: Path) -> None:
    """Average panels within seed and report a known paired effect with n=5, not n=15."""
    rows = []
    initial = {
        key: torch.zeros(2)
        for key in ("encoder.blocks.0.weight", "encoder.blocks.3.weight", "encoder.norm.weight", "decoder.weight")
    }
    well = tmp_path / "well.pt"
    torch.save(initial, well)
    lock_json(
        tmp_path / "protocol.json",
        {"config": {"well": str(well)}, "checkpoint_sha256": hashlib.sha256(well.read_bytes()).hexdigest()},
    )
    lock_json(tmp_path / "evaluation_protocol.json", {"eligible_conditions": [0]})
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for scope, accuracy, forgetting in (("full", 0.5, 0.06), ("last_block", 0.55, 0.02)):
            path = tmp_path / scope / f"seed_{seed}"
            path.mkdir(parents=True)
            (path / "checkpoints").mkdir()
            final = {
                key: value + (0 if scope == "last_block" and "blocks.0." in key else 1)
                for key, value in initial.items()
            }
            torch.save(final, path / "checkpoints" / f"adapted_s{seed}.pt")
            OmegaConf.save(OmegaConf.create({"train_scope": scope, "seeds": [seed], "lr": 1e-4}), path / "config.yaml")
            scores = {key: {"adapted_acc": accuracy, "b5_acc": 0.4} for key in ("knn", "linear")}
            conditions = [
                {"eval_era": 0, "adapted_acc": accuracy, "frozen_acc": 0.4},
                {"eval_era": 1, "adapted_acc": 0.01, "frozen_acc": 0.99},
            ]
            report = {
                "seed": seed,
                "global": scores,
                "conditions": {
                    "final_per_condition": conditions,
                    "adapted_matrix": {"0": {"0": accuracy + forgetting}, "1": {"0": accuracy, "1": 0.01}},
                },
                "probe_panels": [
                    {"panel_seed": p, "global": scores, "final_per_condition": conditions} for p in PANELS[1:]
                ],
            }
            (path / "comparison.json").write_text(json.dumps({"seeds": [report]}), encoding="utf-8")
            row[scope] = read_arm(path, seed, [0])
            assert set(row[scope]["panels"]) == set(map(str, PANELS))
            assert row[scope]["absolute"]["adapted_final_conditions"] == accuracy
        validate_pair(tmp_path, seed)
        rows.append(row)
    result = reduce_rows(rows)
    assert result["metrics"]["last_minus_full"]["linear"]["mean_pp"] == pytest.approx(5)
    assert result["metrics"]["last_minus_full"]["linear"]["n_seeds"] == 5
    assert result["retention_last_minus_full"]["forgetting"]["mean_pp"] == pytest.approx(-4)
    assert result["retention_accuracy_confirmed"]
    assert result["broader_accuracy_superiority"]
    complete = summarize(tmp_path)
    assert complete["primary"]["retention_accuracy_confirmed"]
    assert (tmp_path / "summary.md").is_file()
    assert len(json.loads((tmp_path / "weight_audit.json").read_text())) == 10
    path = tmp_path / "last_block" / f"seed_{SEEDS[0]}" / "config.yaml"
    config = OmegaConf.load(path)
    config.lr = 1e-3
    OmegaConf.save(config, path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_pair(tmp_path, SEEDS[0])


def test_lock_refuses_changed_resume_and_preserves_timestamp(tmp_path: Path) -> None:
    """Resume must not overwrite the original protocol record."""
    path = tmp_path / "protocol.json"
    lock_json(path, {"seed": 5})
    original = path.read_bytes()
    lock_json(path, {"seed": 5})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        lock_json(path, {"seed": 6})
