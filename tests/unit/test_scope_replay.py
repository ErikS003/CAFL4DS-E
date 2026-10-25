"""Exercise matched replay configuration checks and the complete four-arm reduction."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from scripts import run_scope_replay_background as background
from scripts.confirm_last_block import PANELS, SEEDS, lock_json
from scripts.run_scope_replay import PROTOCOL, digest, summarize, validate_config


def config(seed: int, scope: str, replay: bool) -> dict[str, Any]:
    """Build minimal paired settings with the exact registered reservoir policy."""
    policy: dict[str, Any] = {"_target_": "cafl4ds.filters.accept_all.AcceptAll"}
    if replay:
        policy = {
            "_target_": "cafl4ds.filters.composite.CompositeSelector",
            "admission": [],
            "buffer": {
                "_target_": "cafl4ds.filters.reservoir.ReservoirReplay",
                "capacity": 256,
                "replay_batch": 16,
                "max_train_batch": 32,
                "seed": seed,
            },
        }
    return {
        "seed": seed,
        "seeds": [seed],
        "stream": {"seed": seed},
        "train_scope": scope,
        "lr": 1e-4,
        "filter": policy,
        "run_name": "replay" if replay else "control",
    }


@pytest.mark.parametrize("change", ["lr", "capacity", "seed"])
def test_pair_rejects_unregistered_changes(change: str) -> None:
    """Only the fixed policy, not learning rate, buffer size or seed, may differ."""
    replay, control = config(3001, "full", True), config(3001, "full", False)
    validate_config(OmegaConf.create(replay), OmegaConf.create(control), 3001, "full")
    if change == "lr":
        replay["lr"] = 1e-3
    elif change == "capacity":
        replay["filter"]["buffer"]["capacity"] = 512
    else:
        replay["seed"] = 3011
    with pytest.raises(ValueError):
        validate_config(OmegaConf.create(replay), OmegaConf.create(control), 3001, "full")


def test_complete_four_arm_summary_and_control_integrity(tmp_path: Path) -> None:
    """Known replay effects recover within-scope differences and interaction, with twenty audits."""
    control, output = tmp_path / "control", tmp_path / "replay"
    control.mkdir()
    output.mkdir()
    initial = {
        key: torch.zeros(2)
        for key in ("encoder.blocks.0.weight", "encoder.blocks.3.weight", "encoder.norm.weight", "decoder.weight")
    }
    well = tmp_path / "well.pt"
    torch.save(initial, well)
    for root in (control, output):
        lock_json(root / "evaluation_protocol.json", {"eligible_conditions": [0]})
    effects = {
        ("full", False): (0.50, 0.04),
        ("full", True): (0.52, 0.03),
        ("last_block", False): (0.51, 0.05),
        ("last_block", True): (0.55, 0.03),
    }
    for (scope, replay), (accuracy, forgetting) in effects.items():
        for seed in SEEDS:
            directory = (output if replay else control) / scope / f"seed_{seed}"
            (directory / "checkpoints").mkdir(parents=True)
            OmegaConf.save(OmegaConf.create(config(seed, scope, replay)), directory / "config.yaml")
            final = {
                key: value + (0 if scope == "last_block" and "blocks.0." in key else 1)
                for key, value in initial.items()
            }
            torch.save(final, directory / f"checkpoints/adapted_s{seed}.pt")
            scores = {k: {"adapted_acc": accuracy, "b5_acc": 0.4} for k in ("knn", "linear")}
            conditions = [{"eval_era": 0, "adapted_acc": accuracy, "frozen_acc": 0.4}]
            report = {
                "seed": seed,
                "global": scores,
                "conditions": {
                    "final_per_condition": conditions,
                    "adapted_matrix": {"0": {"0": accuracy + forgetting}, "1": {"0": accuracy}},
                },
                "probe_panels": [
                    {"panel_seed": p, "global": scores, "final_per_condition": conditions} for p in PANELS[1:]
                ],
            }
            (directory / "comparison.json").write_text(json.dumps({"seeds": [report]}), encoding="utf-8")
    lock_json(
        output / "protocol.json",
        {
            "config": {"well": str(well)},
            "checkpoint_sha256": digest(well),
            "protocol_sha256": digest(PROTOCOL),
            "source_sha256": {},
            "control_files_sha256": {
                p.relative_to(control).as_posix(): digest(p) for p in control.rglob("*") if p.is_file()
            },
        },
    )
    report = summarize(output, control)
    primary = cast(dict[str, Any], report["primary"])
    assert primary["accuracy"]["full_replay_minus_no_replay"]["knn"]["mean_pp"] == pytest.approx(2)
    assert primary["accuracy"]["last_block_replay_minus_no_replay"]["linear"]["mean_pp"] == pytest.approx(4)
    assert primary["accuracy"]["last_block_replay_minus_no_replay"]["linear"]["n_seeds"] == 5
    assert primary["interaction_replay_effect_last_minus_full"]["knn"]["mean_pp"] == pytest.approx(2)
    assert primary["interaction_replay_effect_last_minus_full"]["forgetting"]["mean_pp"] == pytest.approx(-1)
    assert primary["replay_retention_accuracy_candidate"]["last_block"]
    audit = json.loads((output / "weight_audit.json").read_text())
    assert len(audit["new_replay"]) + len(audit["reused_no_replay"]) == 20
    assert (output / "replay_effects.png").is_file()
    target = control / "full" / f"seed_{SEEDS[0]}" / "comparison.json"
    changed = copy.deepcopy(json.loads(target.read_text()))
    changed["seeds"][0]["global"]["knn"]["b5_acc"] = 0.2
    target.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Reused control changed"):
        summarize(output, control)


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific temporary power request")
@pytest.mark.parametrize("exit_code", [0, 3])
def test_supervisor_releases_keep_awake_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """Use mocked OS/process APIs to check cleanup and duplicate protection without altering power state."""
    calls: list[int] = []

    def power_request(flags: int) -> int:
        calls.append(flags)
        return 1

    monkeypatch.setattr(background, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["supervisor", "--keep-awake"])
    monkeypatch.setattr(
        background,
        "ctypes",
        SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=power_request))),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=456, wait=lambda: exit_code))
    background.main()
    result = json.loads((tmp_path / "background_status.json").read_text())
    assert result["state"] == ("completed" if exit_code == 0 else "failed")
    assert result["keep_awake_release_succeeded"]
    assert not result["keep_awake_active"]
    assert calls == [0x80000001, 0x80000000]
    with pytest.raises(FileExistsError):
        background.main()
