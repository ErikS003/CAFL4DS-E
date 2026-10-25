"""End-to-end wiring test for the P1.2 experiment entry point."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_ROOT = Path(__file__).resolve().parents[2]


def test_adapter_frozen_heads_scope_preserves_head_parameters_and_buffers() -> None:
    """The mechanistic arm trains only its adapter and keeps SSL-head BatchNorm fixed."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script_fixed_heads", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation")
    method = script._method(config)
    script._configure_train_scope(method, "adapter_frozen_heads", 4)
    method.train()
    for module in method._frozen_online_modules:
        module.eval()

    assert all(not parameter.requires_grad for parameter in method.projector.parameters())
    assert all(not parameter.requires_grad for parameter in method.predictor.parameters())
    assert all(not module.training for module in method._frozen_online_modules)
    assert sum(parameter.numel() for parameter in method.parameters() if parameter.requires_grad) == 868
    loss = method.training_step(torch.rand(3, 3, 32, 32))
    loss.backward()
    assert method.encoder.adapter[-1].weight.grad is not None


def test_adaptation_script_writes_matched_artifacts(tmp_path: Path) -> None:
    """A tiny run emits live/B5 health and current-versus-past comparisons."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(
            config_name="adaptation",
            overrides=[
                "seeds=[0]",
                "data.num_regimes=3",
                "data.num_canary_classes=3",
                "data.per_cell=12",
                "stream.support_per_canary=2",
                "stream.query_per_canary=2",
                "stream.era_query_per_cell=1",
                "few_shot_counts=[1,2]",
                "max_train_per_regime=4",
                "batch_size=4",
                "monitor.knn_k=3",
            ],
        )
    paths = script.run_experiment(config, tmp_path)
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert comparison["design"]["matched_initial_weights"] is True
    assert comparison["design"]["update_every"] == 1
    assert comparison["design"]["init_mode"] == "from_scratch"
    assert comparison["design"]["selection"] == {
        "name": "B-floor",
        "filter": "accept_all",
        "replay": False,
    }
    assert comparison["aggregate"]["n_seeds"] == 1
    assert paths["condition_accuracy"].is_file()
    health = paths["health_csv"].read_text(encoding="utf-8")
    assert ",live," in health and ",b5," in health


def test_mae_bdd_config_selects_primary_forgetting_vehicle() -> None:
    """The MAE gate changes the SSL family without weakening the matched P1.2 design."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation_bdd_mae")

    assert config.family == "mae"
    assert config.ssl._target_ == "cafl4ds.ssl.factory.build_mae"
    assert config.filter._target_ == "cafl4ds.filters.accept_all.AcceptAll"
    assert list(config.seeds) == [11, 23, 37]
    assert config.well is None
    assert config.run_name == "bdd_mae_adapt_vs_frozen"


def test_full_bdd_config_separates_train_stream_from_validation_evaluation() -> None:
    """The full-data pilot updates on train, evaluates on val, and leaves test unused."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation_bdd_full_mae")

    assert config.data.split == "train"
    assert config.evaluation_data.split == "val"
    assert config.max_train_per_regime is None
    assert list(config.seeds) == [11]
    assert config.seed_split == "full_data_pilot"


def test_clean_warm_bdd_configs_are_disjoint_and_share_partition_definition() -> None:
    """Warm-up gets train-20, the drive gets train-80, and validation remains evaluation-only."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        warm = compose(config_name="warm_well_bdd_train20_mae")
        drive = compose(config_name="adaptation_bdd_clean_warm_mae")

    assert warm.data.partition_role == "warm"
    assert drive.data.partition_role == "stream"
    assert warm.data.warm_fraction == drive.data.warm_fraction == 0.2
    assert warm.data.partition_seed == drive.data.partition_seed == 20260921
    assert warm.data.split == drive.data.split == "train"
    assert drive.evaluation_data.split == "val"
    assert drive.well == warm.well_out


def test_clean_warm_replay_config_is_compute_matched_to_full_backbone_control() -> None:
    """The retention intervention changes replay, not optimizer batch size or model scope."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation_bdd_clean_warm_mae_replay")

    assert config.train_scope == "full"
    assert config.optim.lr == 1e-4
    assert config.filter.buffer.capacity == 256
    assert config.filter.buffer.replay_batch == 16
    assert config.filter.buffer.max_train_batch == config.batch_size == 32
    assert config.data.partition_role == "stream"
    assert config.evaluation_data.split == "val"


def test_replay_filter_design_records_compute_and_storage_budget() -> None:
    """B1 artifacts expose enough metadata to audit replay's extra storage and compute."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script_replay", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation", overrides=["filter=reservoir", "batch_size=8"])
    assert script._filter_design(config) == {
        "name": "B1",
        "filter": "reservoir",
        "replay": True,
        "buffer_capacity": 256,
        "replay_batch": 8,
        "incoming_batch": 8,
        "max_train_batch": None,
        "nominal_full_buffer_train_batch": 16,
        "admission": [],
        "admission_parameters": [],
    }


def test_compute_matched_replay_design_records_fixed_training_batch() -> None:
    """Compute-matched B1 provenance distinguishes it from incoming-plus-replay B1."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script_compute_replay", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation", overrides=["filter=reservoir_compute", "batch_size=8"])
    design = script._filter_design(config)
    assert design["filter"] == "reservoir_compute"
    assert design["max_train_batch"] == 8
    assert design["nominal_full_buffer_train_batch"] == 8


def test_compute_matched_dedup_design_records_calibrated_threshold() -> None:
    """B1.5 provenance records both the admission threshold and compute budget."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script_dedup_replay", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation", overrides=["filter=dedup_reservoir_compute"])
    design = script._filter_design(config)
    assert design["name"] == "B1.5"
    assert design["filter"] == "dedup_reservoir_compute"
    assert design["admission_parameters"] == [{"threshold": 0.998}]


def test_feature_distillation_design_records_target_metric_and_weight() -> None:
    """B1-D artifacts identify the cached feature target and regularization strength."""
    spec = importlib.util.spec_from_file_location(
        "_adaptation_script_feature_distill", _ROOT / "scripts" / "run_adaptation_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds" / "configs")):
        config = compose(config_name="adaptation", overrides=["filter=reservoir_distill_compute"])
    design = script._filter_design(config)
    assert design["name"] == "B1-D"
    assert design["filter"] == "reservoir_distill_compute"
    assert design["distill_weight"] == 1.0
    assert design["distill_target"] == "arrival_time_normalized_backbone_embedding"
    assert design["distill_metric"] == "cosine_distance"
