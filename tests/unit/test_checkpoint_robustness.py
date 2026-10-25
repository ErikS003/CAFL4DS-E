"""Disjoint evaluation, actual backbone updates, pairing, and immutable source checks."""

import copy
import json
from pathlib import Path

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from scripts import evaluate_checkpoint_robustness as study


def test_disjoint_nested_pools() -> None:
    """Pools are disjoint, deterministic, balanced and nested across budgets."""
    labels = torch.arange(3).repeat(240)
    regimes = torch.arange(4).repeat_interleave(180)
    pool, query = study.split_pools(labels, regimes)
    assert not set(pool.tolist()) & set(query.tolist())
    assert set(pool.tolist()) | set(query.tolist()) == set(range(len(labels)))
    assert torch.equal(query, study.split_pools(labels, regimes)[1])
    for seed in study.DRAWS:
        small = study.support_draw(labels, pool, seed, 5)
        large = study.support_draw(labels, pool, seed, 100)
        assert set(small.tolist()) <= set(large.tolist()) <= set(pool.tolist())
        assert labels[large].bincount().tolist() == [100, 100, 100]
    with pytest.raises(ValueError, match="Insufficient"):
        study.support_draw(labels, pool, 1, 1000)


def test_finetuning_changes_copy_not_source() -> None:
    """Every backbone is trainable downstream, but original weights remain unchanged."""
    torch.set_num_threads(1)
    torch.manual_seed(22)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3)
    encoder.requires_grad_(False)
    initial = copy.deepcopy(encoder.state_dict())
    images, labels = torch.rand(6, 3, 16, 16), torch.tensor([0, 1, 2, 0, 1, 2])
    model, head, audit = study.finetune(encoder, images, labels, seed=1, steps=3)
    assert audit["backbone_delta_l2"] > 0
    assert all(torch.equal(initial[k], v) for k, v in encoder.state_dict().items())
    assert all(p.requires_grad for p in model.parameters())
    repeated, head2, _ = study.finetune(encoder, images, labels, seed=1, steps=3)
    assert all(torch.equal(v, repeated.state_dict()[k]) for k, v in model.state_dict().items())
    assert torch.equal(head.weight, head2.weight)


def test_scores_and_count_selection() -> None:
    """Primary selection is count-only and macro metrics have explicit weights."""
    labels = torch.tensor([0] * 30 + [1] * 10 + [0] * 4)
    regimes = torch.tensor([0] * 40 + [1] * 4)
    eligible = study.qualified_conditions(labels, regimes)
    assert eligible == [0]
    result = study.scores(torch.zeros_like(labels), labels, regimes, eligible)
    assert result["global_balanced"] == 0.5
    assert result["conditions"] == 0.75
    assert result["all_conditions"] == 0.875


def test_probe_predictions() -> None:
    """Both frozen readouts recover a separated synthetic support set."""
    x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    y = torch.tensor([0, 0, 1, 1])
    for kind in ["knn", "linear"]:
        assert torch.equal(study.probe_predictions(kind, x, y, x), y)


def test_lock_canonicalization(tmp_path: Path) -> None:
    """Integer JSON keys round-trip, but real protocol changes are rejected."""
    path = tmp_path / "lock.json"
    study.lock_json(path, {"names": {0: "day"}})
    study.lock_json(path, {"names": {0: "day"}})
    with pytest.raises(ValueError, match="fingerprint"):
        study.lock_json(path, {"names": {0: "night"}})


def test_summary_uses_five_seeds_not_draws(tmp_path: Path) -> None:
    """Shared draws are averaged within seeds and all readouts are reported."""
    (tmp_path / "cells").mkdir()
    for model in ["frozen", *[f"{arm}_{seed}" for arm in study.ARMS for seed in study.SEEDS]]:
        for shots in study.BUDGETS:
            for draw in study.DRAWS:
                score = 0.4 if model == "frozen" else 0.5
                row = {r: dict.fromkeys(study.METRICS, score) for r in study.READOUTS}
                (tmp_path / "cells" / f"{model}_{shots}_{draw}.json").write_text(json.dumps(row))
    study.summarize(tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    result = report["readouts"]["25_finetune"]["contrasts"]["last_block_replay_minus_frozen"]["global_balanced"]
    assert result["n_seeds"] == 5
    assert result["mean_pp"] == pytest.approx(10)
    assert report["primary_joint_positive"]
    assert (tmp_path / "robustness.png").is_file()
