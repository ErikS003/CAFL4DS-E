"""Guard the prospective confirmation's statistical unit and decision rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scripts.confirm_replay_advantage import paired_interval, seed_readout, verdict
from scripts.run_adaptation_experiment import run_experiment


def test_paired_interval_does_not_accept_panels_as_fifteen_seeds() -> None:
    """Panels are repeated measurements, so only five training seeds can enter the interval."""
    with pytest.raises(ValueError, match="exactly five"):
        paired_interval([0.03] * 15)
    result = paired_interval([0.03] * 5)
    assert result["mean_pp"] == pytest.approx(3)
    assert result["ci95_pp"] == pytest.approx([3, 3])


def test_confirmation_distinguishes_support_contradiction_and_uncertainty() -> None:
    """A mixed or small effect cannot be silently reported as a positive confirmation."""
    positive = paired_interval([0.03, 0.04, 0.05, 0.04, 0.04])
    negative = paired_interval([-0.03] * 5)
    uncertain = paired_interval([-0.10, 0.05, 0.05, 0.05, 0.05])
    assert verdict({"knn": positive, "final_conditions": positive}) == "supported"
    assert verdict({"knn": positive, "final_conditions": negative}) == "contradicted_in_this_setting"
    assert verdict({"knn": uncertain, "final_conditions": positive}) == "not_established_inconclusive"
    assert verdict({"knn": paired_interval([0.01] * 5), "final_conditions": positive}) == "not_established_inconclusive"


def test_seed_readout_averages_panels_before_training_seeds() -> None:
    """Each seed contributes one paired gain even when three support/query panels are evaluated."""
    panels = [
        {
            "panel_seed": index,
            "global": {"knn": {"gain": gain}, "linear": {"gain": gain}},
            "final_per_condition": [{"gain": gain}],
        }
        for index, gain in enumerate((0.01, 0.02, 0.09))
    ]
    report = {
        "seed": 2003,
        "global": panels[0]["global"],
        "conditions": {"final_per_condition": panels[0]["final_per_condition"], "adapted_summary": {}},
        "evaluation_sizes": {},
        "probe_panels": panels[1:],
    }
    result = seed_readout(report)
    assert result["panel_means"]["knn"] == pytest.approx(0.04)
    assert len(result["panels"]) == 3


def test_repeated_evaluation_and_checkpoint_on_independent_synthetic_data(tmp_path: Path) -> None:
    """Exercise independent-source panels and saved models before using the expensive BDD stream."""
    config_dir = Path(__file__).resolve().parents[2] / "cafl4ds/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="adaptation",
            overrides=["seeds=[2003]", "max_train_per_regime=8", "few_shot_counts=[]"],
        )
    with open_dict(config):
        config.evaluation_data = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
        config.evaluation_data.seed = 9001
        config.probe_panel_seeds = [20260927, 20260928]
        config.save_final_well = True
    paths = run_experiment(config, tmp_path)
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    report = comparison["seeds"][0]
    assert comparison["design"]["independent_evaluation_source"]
    assert (tmp_path / "checkpoints/adapted_s2003.pt").is_file()
    assert report["evaluation_sizes"]["global_query"] == 48
    assert [panel["panel_seed"] for panel in report["probe_panels"]] == [20260927, 20260928]
    assert all(panel["final_per_condition"] for panel in report["probe_panels"])
