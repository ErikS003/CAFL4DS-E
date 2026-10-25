"""Verify that the restricted MAE arm changes only the intended parameter groups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_mae
from scripts.confirm_replay_advantage import _SEEDS
from scripts.run_adaptation_experiment import _configure_train_scope
from scripts.run_last_block_comparison import summarize


def test_last_block_mae_keeps_early_encoder_unchanged() -> None:
    """A real AdamW update must leave early layers fixed while training last block and decoder."""
    torch.manual_seed(5)
    method = build_mae(
        TinyViTEncoder(img_size=32, embed_dim=24, depth=4, num_heads=3),
        decoder_dim=24,
        decoder_depth=1,
        decoder_heads=3,
    )
    _configure_train_scope(method, "last_block", 16)
    before = {key: value.clone() for key, value in method.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in method.parameters() if p.requires_grad), lr=1e-4)
    method.training_step(torch.rand(4, 3, 32, 32)).backward()
    optimizer.step()
    changed = [key for key, value in method.state_dict().items() if not torch.equal(value, before[key])]
    assert any(key.startswith("encoder.blocks.3.") for key in changed)
    assert any(key.startswith("encoder.norm.") for key in changed)
    assert any(key.startswith("decoder.") for key in changed)
    assert all(key.startswith(("encoder.blocks.3.", "encoder.norm.", "decoder.")) for key in changed)


def test_summary_pairs_panels_and_retention_and_rejects_mismatches(tmp_path: Path) -> None:
    """A known synthetic effect exercises the complete reduction and mismatch gate."""
    for arm, accuracy, forgetting, scope in (("last", 0.55, 0.01, "last_block"), ("full", 0.50, 0.05, "full")):
        for seed in _SEEDS:
            directory = tmp_path / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            config = {"seed": 11, "seeds": [seed], "stream": {"seed": 11}, "train_scope": scope, "lr": 1e-4}
            OmegaConf.save(OmegaConf.create(config), directory / "config.yaml")
            global_scores = {
                key: {"adapted_acc": accuracy, "b5_acc": 0.4, "gain": accuracy - 0.4} for key in ("knn", "linear")
            }
            conditions = [{"adapted_acc": accuracy, "frozen_acc": 0.4, "gain": accuracy - 0.4}]
            report = {
                "seed": seed,
                "global": global_scores,
                "evaluation_sizes": {},
                "conditions": {
                    "final_per_condition": conditions,
                    "adapted_summary": {"forgetting_measure": forgetting, "backward_transfer": -forgetting},
                },
                "probe_panels": [
                    {"panel_seed": panel, "global": global_scores, "final_per_condition": conditions}
                    for panel in (20260927, 20260928)
                ],
            }
            (directory / "comparison.json").write_text(json.dumps({"seeds": [report]}), encoding="utf-8")
    result = summarize(tmp_path / "last", tmp_path / "full")
    assert result["metrics"]["last_minus_full"]["knn"]["mean_pp"] == pytest.approx(5)
    assert result["metrics"]["last_minus_full"]["knn"]["n_seeds"] == 5
    assert result["retention_last_minus_full"]["forgetting_measure"]["mean_pp"] == pytest.approx(-4)
    assert result["retention_accuracy_candidate"]
    assert result["frozen_advantage_verdict"] == "supported"
    path = tmp_path / "last" / f"seed_{_SEEDS[0]}" / "config.yaml"
    changed = OmegaConf.load(path)
    changed.lr = 1e-3
    OmegaConf.save(changed, path)
    with pytest.raises(ValueError, match="config mismatch"):
        summarize(tmp_path / "last", tmp_path / "full")
