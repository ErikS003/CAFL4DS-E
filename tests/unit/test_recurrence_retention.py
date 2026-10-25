"""Test causal arrivals, fixed evaluation, real online training, retention math, and cleanup."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.recurrence import (
    arrival_orders,
    evaluation_pool,
    observation_grid,
    prediction_scores,
    recovery_latency,
    restrict_scenes,
    retention_readout,
    visit_quotas,
)
from cafl4ds.ssl.factory import build_mae
from scripts import run_recurrence_background as background
from scripts.run_recurrence_retention import ARMS, READOUTS, evaluate, summarize, train_arm


def settings() -> dict[str, Any]:
    """Small deterministic pools with the same domain-balanced construction as BDD."""
    return {
        "pool_seed": 31,
        "support_per_domain_class": 3,
        "query_per_domain_class": 4,
        "min_query_per_domain_class": 3,
        "draw_seeds": [11, 13, 17],
        "shots_per_class": 3,
    }


def test_unique_matched_arrivals_and_quotas() -> None:
    """Every visit has identical scene counts, and shuffled uses exactly the same images."""
    data = SyntheticAttributeSource(
        num_regimes=2, num_canary_classes=2, per_cell=128, img_size=16, long_tail=False
    ).load()
    names = ["regime0", "regime1"]
    quotas = visit_quotas(data, names, maximum=128, minimum=32)
    assert quotas == {0: 64, 1: 64}
    orders = arrival_orders(data, names, quotas, seed=55)
    assert len(set(orders["recurrence"].tolist())) == len(orders["recurrence"])
    assert torch.equal(orders["recurrence"].sort().values, orders["shuffled"].sort().values)
    assert torch.equal(orders["recurrence"], arrival_orders(data, names, quotas, 55)["recurrence"])
    for visit, indices in enumerate(orders["recurrence"].chunk(4)):
        assert set(data.era_key[indices].tolist()) == {visit % 2}
        assert data.canary[indices].bincount().tolist() == [64, 64]
    with pytest.raises(ValueError, match="quota"):
        visit_quotas(data, names, maximum=128, minimum=128)


def test_fixed_eval_and_scene_mapping() -> None:
    """Restrict by names and never allow a support draw to touch a query."""
    data = SyntheticAttributeSource(num_regimes=2, num_canary_classes=3, per_cell=12, long_tail=False).load()
    restricted, indices = restrict_scenes(data, ["canary2", "canary0"])
    assert restricted.canary_names == {0: "canary2", 1: "canary0"}
    assert torch.equal(restricted.images, data.images[indices])
    evaluation = evaluation_pool(restricted, ["regime0", "regime1"], settings())
    for support in evaluation.supports.values():
        assert not set(support.tolist()) & set(evaluation.query.tolist())
        assert evaluation.labels[support].bincount().tolist() == [3, 3]
    assert len(set(evaluation.manifest["original_indices"])) == len(evaluation.images)
    with pytest.raises(ValueError, match="Inadequate"):
        evaluation_pool(restricted, ["regime0", "regime1"], {**settings(), "min_query_per_domain_class": 15})


def make_trace() -> list[dict[str, Any]]:
    """Construct loss during absence and recovery on return with known endpoints."""
    points = [
        (0, 0.4, 0.4),
        (4, 0.8, 0.5),
        (8, 0.6, 0.8),
        (9, 0.7, 0.7),
        (10, 0.8, 0.65),
        (11, 0.8, 0.6),
        (12, 0.85, 0.6),
        (13, 0.8, 0.7),
        (14, 0.8, 0.8),
        (15, 0.8, 0.8),
        (16, 0.8, 0.85),
    ]
    return [
        {
            "step": step,
            "scores": {kind: {"A": a, "B": b, "balanced": (a + b) / 2, "micro": (a + b) / 2} for kind in READOUTS},
        }
        for step, a, b in points
    ]


def test_retention_signs_and_recovery_censoring() -> None:
    """Separate interference from BWT, and never report censored recovery as zero."""
    trace = make_trace()
    result = retention_readout(trace, 4, "finetune")
    assert result["interference"] == pytest.approx(0.2)
    assert result["bwt"] == pytest.approx(0.025)
    assert result["return_gain_A"] == pytest.approx(0.25)
    recovery = recovery_latency(trace, 4, 0.005)
    assert recovery["A"]["updates"] == 2
    assert recovery["B"]["updates"] == 2
    for row in trace:
        if row["step"] > 8:
            row["scores"]["linear"]["A"] = 0.1
    assert recovery_latency(trace, 4, 0.005)["A"]["censored"]
    assert recovery_latency(trace, 4, 0.005)["A"]["updates"] is None
    assert observation_grid(4) == list(range(17))
    with pytest.raises(ValueError, match="Incomplete"):
        retention_readout(trace[:-1], 4, "finetune")


def test_class_balanced_scores() -> None:
    """Domain weights do not silently favor the largest query cell."""
    result = prediction_scores(
        torch.zeros(8), torch.tensor([0, 0, 0, 1, 0, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    )
    assert result["A"] == 0.5 and result["B"] == 0.5 and result["balanced"] == 0.5


def test_real_training_and_readonly_evaluation(tmp_path: Path) -> None:
    """Exercise both optimizer policies, copied fine-tuning, RNG preservation and checkpointing."""
    torch.set_num_threads(1)
    torch.manual_seed(2)
    method = build_mae(
        TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3),
        decoder_dim=12,
        decoder_depth=1,
        decoder_heads=3,
    )
    initial = copy.deepcopy(method.state_dict())
    source = SyntheticAttributeSource(
        num_regimes=2, num_canary_classes=2, per_cell=128, img_size=16, long_tail=False
    ).load()
    validation = SyntheticAttributeSource(
        num_regimes=2, num_canary_classes=2, per_cell=12, img_size=16, long_tail=False, seed=31
    ).load()
    evaluation = evaluation_pool(validation, ["regime0", "regime1"], {**settings(), "draw_seeds": [11]})
    rng = torch.get_rng_state().clone()
    frozen, reference = evaluate(method, evaluation, None, 0, True, 2, tmp_path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(initial[k], v) for k, v in method.state_dict().items())
    config = OmegaConf.create(
        {
            "optim": {"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 0.05},
            "recurrence": {"fine_tune_steps": 2, "recovery_tolerance": 0.005},
        }
    )
    order = arrival_orders(source, ["regime0", "regime1"], {0: 64, 1: 64}, 3)["recurrence"]
    for arm in ARMS[:2]:
        result = train_arm(
            config, method, source, evaluation, order, 3, arm, frozen, reference, tmp_path, lambda **_: None
        )
        assert result["budget"]["arrivals"] == result["budget"]["optimizer_images"] == 512
        assert result["budget"]["updates"] == 16
        assert result["budget"]["reservoir_seen"] == (512 if arm == "recurrence_replay" else 0)
        assert result["encoder_delta_l2"] > 0
        assert len(list((tmp_path / arm / "seed_3/checkpoints").glob("*.pt"))) == 4
        resumed = train_arm(
            config, method, source, evaluation, order, 3, arm, frozen, reference, tmp_path, lambda **_: None
        )
        assert resumed["metrics"] == result["metrics"]
    assert all(torch.equal(initial[k], v) for k, v in method.state_dict().items())
    (tmp_path / ARMS[0] / "seed_7").mkdir()
    with pytest.raises(ValueError, match="Partial arm"):
        train_arm(config, method, source, evaluation, order, 7, ARMS[0], frozen, reference, tmp_path, lambda **_: None)


def test_summary_gate_and_seed_count(tmp_path: Path) -> None:
    """All arms remain in the report; paired inference uses five seeds, not support draws."""
    seeds = [1, 2, 3, 4, 5]
    for arm in ARMS:
        for seed in seeds:
            directory = tmp_path / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            trace = make_trace()
            metrics = {kind: retention_readout(trace, 4, kind) for kind in READOUTS}
            if arm == "recurrence_replay":
                for metric in metrics.values():
                    metric["interference"] -= 0.1
            record = {"metrics": metrics, "trajectory": trace, "recovery_linear": {}, "budget": {"updates": 16}}
            (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
    result = summarize(tmp_path, seeds)
    assert result["retention_supported"]
    assert result["readouts"]["finetune"]["contrasts"]["replay_minus_no_replay"]["interference"]["n_seeds"] == 5
    assert (tmp_path / "trajectories.png").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows power API")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_power_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    """Temporary system-awake requests are released on success and worker failure."""
    calls: list[int] = []

    def power(flags: int) -> int:
        calls.append(flags)
        return 1

    monkeypatch.setattr(background, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["supervisor", "--keep-awake"])
    monkeypatch.setattr(
        background,
        "ctypes",
        SimpleNamespace(windll=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=power))),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=1, wait=lambda: exit_code))
    background.main()
    status = json.loads((tmp_path / "background_status.json").read_text())
    assert status["state"] == ("completed" if exit_code == 0 else "failed")
    assert status["keep_awake_release_succeeded"] and not status["keep_awake_active"]
    assert calls == [0x80000001, 0x80000000]
    with pytest.raises(FileExistsError):
        background.main()


def test_restart_supervisor_orders_preflight_collection_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic collection follows passing preflight, and successful collection is audited."""
    commands: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["supervisor", "--output", str(tmp_path), "--prior-run", str(tmp_path / "prior")])

    def execute(command: list[str], **kwargs: Any) -> None:  # noqa: ANN401 - subprocess mock
        commands.append(command)

    def launch(command: list[str], **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 - subprocess mock
        commands.append(command)
        return SimpleNamespace(pid=1234, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setattr(subprocess, "Popen", launch)
    background.main()
    assert "--preflight-only" in commands[0]
    assert "--preflight-only" not in commands[1]
    assert "scripts.audit_recurrence_retention" in commands[2]
    status = json.loads((tmp_path / "background_status.json").read_text())
    assert status["state"] == "completed" and not status["keep_awake_requested"]


def test_restart_failed_preflight_never_launches_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed setup check must stop before the first research optimizer update."""
    monkeypatch.setattr(sys, "argv", ["supervisor", "--output", str(tmp_path), "--prior-run", str(tmp_path / "prior")])

    def fail(command: list[str], **kwargs: Any) -> None:  # noqa: ANN401 - subprocess mock
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Training launched after failed preflight")
    )
    with pytest.raises(subprocess.CalledProcessError):
        background.main()
    assert json.loads((tmp_path / "background_status.json").read_text())["state"] == "failed"
