"""P1.2 matched online-adaptation versus frozen-backbone experiment.

Each seed snapshots an init-matched frozen twin before the first update. The live and frozen arms
share the same held-out global and per-regime evaluation sets and the same checkpoint grid. The live
arm alone receives the stream; B5 is measured only. Outputs include the standard wide health corpus,
per-seed matched reports, a long-form condition-accuracy CSV, and ``comparison.json``.

Example:
    ``uv run python scripts/run_adaptation_experiment.py``
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds import adaptation, deploy, deploy_corpus, harness, warmup
from cafl4ds.data.attributes import AttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.eval import PerEraProbe, adaptation_report, few_shot_probe_curve
from cafl4ds.health_trust import CANARY_SIGNALS, DEFAULT_LABEL_FREE_SIGNALS, Backbone, backbone_family
from cafl4ds.models.adapters import ResidualAdapterEncoder
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init
from cafl4ds.ssl.simsiam import SimSiam
from cafl4ds.ssl.teacher_student import TeacherStudentAdapter

logger.remove()
logger.add(sys.stdout, level="INFO")

# Hydra's local multirun launcher invokes every sweep cell in this process. The attributed source
# is immutable across cells and drive seeds, so retain its decoded tensor cache instead of decoding
# the same BDD images once per cell.
_SOURCE_CACHE: dict[str, AttributeSource] = {}


def _filter_design(config: DictConfig) -> dict[str, Any]:
    """Describe the configured selection/replay arm for artifact provenance."""
    target = str(config.filter.get("_target_", ""))
    buffer = config.filter.get("buffer")
    admission = config.filter.get("admission") or []
    admission_targets = [str(stage.get("_target_", "")).rsplit(".", 1)[-1] for stage in admission]
    admission_parameters = [
        {str(key): value for key, value in stage.items() if not str(key).startswith("_")} for stage in admission
    ]
    if target.endswith("AcceptAll"):
        return {"name": "B-floor", "filter": "accept_all", "replay": False}
    if buffer:
        buffer_target = str(buffer.get("_target_", ""))
        if buffer_target.endswith("ReservoirReplay"):
            distilled = buffer_target.endswith("FeatureDistillationReservoirReplay")
            name = "B1.5" if admission_targets else "B1"
            max_train_batch = buffer.get("max_train_batch")
            label = "dedup_reservoir" if admission_targets else "reservoir"
            if distilled:
                name = "B1-D"
                label = "reservoir_distill"
            if max_train_batch is not None:
                label += "_compute"
            design = {
                "name": name,
                "filter": label,
                "replay": True,
                "buffer_capacity": int(buffer.capacity),
                "replay_batch": int(buffer.replay_batch),
                "incoming_batch": int(config.batch_size),
                "max_train_batch": None if max_train_batch is None else int(max_train_batch),
                "nominal_full_buffer_train_batch": (
                    int(config.batch_size) + int(buffer.replay_batch)
                    if max_train_batch is None
                    else int(max_train_batch)
                ),
                "admission": admission_targets,
                "admission_parameters": admission_parameters,
            }
            if distilled:
                design["distill_weight"] = float(buffer.distill_weight)
                design["distill_target"] = "arrival_time_normalized_backbone_embedding"
                design["distill_metric"] = "cosine_distance"
            return design
    return {
        "name": target.rsplit(".", 1)[-1] or "unknown",
        "filter": target.rsplit(".", 1)[-1] or "unknown",
        "replay": False,
        "admission": admission_targets,
    }


def _selection_runtime(selection_filter: object) -> dict[str, Any]:
    """Collect optional live admission statistics without coupling the loop to filter types."""
    rows = []
    for stage in getattr(selection_filter, "admission", []):
        stats = getattr(stage, "stats", None)
        if callable(stats):
            rows.append({"stage": type(stage).__name__, **stats()})
    runtime: dict[str, Any] = {"admission": rows}
    buffer = getattr(selection_filter, "buffer", None)
    stats = getattr(buffer, "stats", None)
    if callable(stats):
        runtime["buffer"] = {"stage": type(buffer).__name__, **stats()}
    return runtime


def _shared_source(data_config: DictConfig) -> AttributeSource:
    """Return one decoded-source owner for identical resolved data configurations."""
    key = json.dumps(OmegaConf.to_container(data_config, resolve=True), sort_keys=True, default=str)
    source = _SOURCE_CACHE.get(key)
    if source is None:
        source = instantiate(data_config)
        _SOURCE_CACHE[key] = source
    else:
        logger.info("reusing attributed-source decode cache across sweep cells")
    return source


def _method(config: DictConfig) -> SSLMethod:
    """Build one method using the same initialization precedence as the deployment harness."""
    method: SSLMethod = instantiate(config.ssl, encoder=instantiate(config.encoder))
    checkpoint = config.init.checkpoint
    if config.init.mode == "pretrained" and not checkpoint:
        checkpoint = str(Path(to_absolute_path(config.pretrain_dir)) / f"{method.name}.pt")
    apply_encoder_init(method.encoder, config.init.mode, checkpoint)
    if config.get("well"):
        warmup.load_well(method, to_absolute_path(str(config.well)))
    if str(config.get("train_scope", "full")) == "teacher_adapter":
        if not isinstance(method, SimSiam):
            raise TypeError("train_scope='teacher_adapter' requires a fully pretrained SimSiam method.")
        return TeacherStudentAdapter(
            method,
            bottleneck_dim=int(config.get("adapter_bottleneck", 16)),
            retention_weight=float(config.get("teacher_retention_weight", 1.0)),
        )
    _configure_train_scope(method, str(config.get("train_scope", "full")), int(config.get("adapter_bottleneck", 16)))
    return method


def _configure_train_scope(method: SSLMethod, scope: str, adapter_bottleneck: int) -> None:
    """Choose which representation parameters receive online gradients."""
    if scope == "full":
        return
    if scope in {"adapter", "adapter_frozen_heads"}:
        rng_state = torch.random.get_rng_state()
        try:
            method.encoder = ResidualAdapterEncoder(method.encoder, adapter_bottleneck)
        finally:
            # Construction must not change the paired stream/augmentation RNG trajectory.
            torch.random.set_rng_state(rng_state)
        if scope == "adapter_frozen_heads":
            frozen_modules = []
            for name in ("projector", "predictor"):
                module = getattr(method, name, None)
                if isinstance(module, torch.nn.Module):
                    module.requires_grad_(False)
                    module.eval()
                    frozen_modules.append(module)
            # StreamingLoop restores these modules to eval after method.train(), so BatchNorm
            # running statistics remain frozen along with the head parameters.
            method._frozen_online_modules = tuple(frozen_modules)
        return
    if scope == "last_block":
        for parameter in method.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in method.encoder.blocks[-1].parameters():
            parameter.requires_grad_(True)
        for parameter in method.encoder.norm.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(
        f"unknown train_scope {scope!r}; expected 'full', 'adapter', 'adapter_frozen_heads', or 'last_block'."
    )


def _parameter_budget(method: SSLMethod) -> dict[str, int]:
    """Return total and online-trainable parameter counts for provenance."""
    return {
        "total": sum(parameter.numel() for parameter in method.parameters()),
        "trainable": sum(parameter.numel() for parameter in method.parameters() if parameter.requires_grad),
        "encoder_trainable": sum(
            parameter.numel() for parameter in method.encoder.parameters() if parameter.requires_grad
        ),
    }


def _git_sha() -> str | None:
    """Read git provenance from the installed hatch-vcs version without a subprocess."""
    try:
        local = version("cafl4ds").split("+", 1)
    except PackageNotFoundError:
        return None
    return local[1] if len(local) > 1 else None


def _global_scores(
    live_method: SSLMethod,
    frozen_method: SSLMethod,
    stream: RegimeStream,
    knn_k: int,
) -> dict[str, dict[str, float]]:
    """Evaluate both final encoders on identical global support/query examples."""
    live_training, frozen_training = live_method.training, frozen_method.training
    live_method.eval()
    frozen_method.eval()
    try:
        return {
            probe: adaptation_report(
                live_method.encode,
                frozen_method.encode,
                stream.eval_sets,
                probe=probe,
                knn_k=knn_k,
            )
            for probe in ("knn", "linear")
        }
    finally:
        live_method.train(live_training)
        frozen_method.train(frozen_training)


def _few_shot_scores(
    live_method: SSLMethod,
    frozen_method: SSLMethod,
    stream: RegimeStream,
    shots: list[int],
) -> dict[str, Any]:
    """Compare final encoders across nested balanced linear-probe support sizes."""
    live_training, frozen_training = live_method.training, frozen_method.training
    live_method.eval()
    frozen_method.eval()
    try:
        live = few_shot_probe_curve(live_method.encode, stream.eval_sets, shots)
        frozen = few_shot_probe_curve(frozen_method.encode, stream.eval_sets, shots)
        return {
            str(shot): {
                "adapted_acc": live[shot],
                "b5_acc": frozen[shot],
                "gain": live[shot] - frozen[shot],
            }
            for shot in shots
        }
    finally:
        live_method.train(live_training)
        frozen_method.train(frozen_training)


def _run_seed(
    config: DictConfig,
    out_dir: Path,
    seed: int,
    source: AttributeSource | None = None,
    evaluation_source: AttributeSource | None = None,
) -> tuple[dict[str, Any], dict[int, str], dict[int, Any]]:
    """Run one matched live/B5 seed and return its deploy-compatible report."""
    config.seed = seed
    torch.manual_seed(seed)
    source = source or instantiate(config.data)
    stream: RegimeStream = instantiate(config.stream, source=source, evaluation_source=evaluation_source)
    method = _method(config)
    frozen = copy.deepcopy(method)
    parameter_budget = _parameter_budget(method)
    family = backbone_family(method.name)
    filter_design = _filter_design(config)
    if family is not Backbone(str(config.family)):
        raise ValueError(f"configured family {config.family!r} does not match method {method.name!r}")

    live_probe = PerEraProbe(
        stream.eval_sets,
        probe=str(config.condition_probe),
        knn_k=int(config.condition_knn_k),
    )
    name = f"{config.run_name}_s{seed}"
    selection_filter = instantiate(config.filter)
    live = harness.run_stream_arm(
        name=f"{name}_live",
        role="live",
        method=method,
        stream=stream,
        optimizer=instantiate(config.optim, params=(p for p in method.parameters() if p.requires_grad)),
        selection_filter=selection_filter,
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        out_dir=out_dir,
        eval_every=int(config.eval_every),
        update_every=int(config.update_every),
        device=str(config.device),
        era_evaluator=live_probe,
    )
    grid = harness.health_grid(live)
    frozen_arm = harness.run_frozen_arm(
        name=f"{name}_b5",
        frozen_method=frozen,
        monitor=instantiate(config.monitor, eval_sets=stream.eval_sets),
        grid=grid,
        device=str(config.device),
    )

    frozen_probe = PerEraProbe(
        stream.eval_sets,
        probe=str(config.condition_probe),
        knn_k=int(config.condition_knn_k),
    )
    for era in sorted(live_probe.matrix):
        frozen_probe.record(frozen.encode, era)

    expected = [*DEFAULT_LABEL_FREE_SIGNALS[family], *CANARY_SIGNALS]
    header = {
        "backbone": str(config.family),
        "I": str(config.init.mode),
        "warm": bool(config.get("well")),
        "diet": "regime",
        "A": str(filter_design["name"]),
        "seed": seed,
        "img_size": int(config.img_size),
        "device": str(config.device),
        "lr": float(config.optim.lr),
        "update_every": int(config.update_every),
        "train_scope": str(config.get("train_scope", "full")),
        "adapter_bottleneck": int(config.get("adapter_bottleneck", 16)),
        "teacher_retention_weight": float(config.get("teacher_retention_weight", 1.0)),
        "parameter_budget": parameter_budget,
        "matched_frozen": True,
    }
    report = deploy.build_deploy_report(
        config_header=header,
        family=family,
        live=live,
        b5=frozen_arm,
        expected_signals=expected,
        canary_chance=1.0 / stream.eval_num_canary_classes,
    )
    report["study"] = "P1.2"
    report["selection"] = filter_design
    report["selection_runtime"] = _selection_runtime(selection_filter)
    report["adaptation"] = {
        "seed": seed,
        "parameter_budget": parameter_budget,
        "global": _global_scores(method, frozen, stream, int(config.condition_knn_k)),
        "few_shot": _few_shot_scores(
            method,
            frozen,
            stream,
            [int(shot) for shot in config.get("few_shot_counts", [1, 2, 5, 10, 20])],
        ),
        "conditions": adaptation.condition_report(live_probe, frozen_probe, stream.era_names),
    }
    report["adaptation"]["evaluation_sizes"] = _evaluation_sizes(stream)
    if config.get("save_final_well", False):
        warmup.save_well(method, out_dir / "checkpoints" / f"adapted_s{seed}.pt")
    panel_seeds = config.get("probe_panel_seeds", [])
    if panel_seeds:
        if evaluation_source is None:
            raise ValueError("Repeated evaluation panels require an independent evaluation source.")
        panels = []
        for panel_seed in panel_seeds:
            panel = instantiate(
                config.stream, source=source, evaluation_source=evaluation_source, canary_seed=int(panel_seed)
            )
            panels.append(_final_panel(method, frozen, panel, int(panel_seed), int(config.condition_knn_k)))
        report["adaptation"]["probe_panels"] = panels
    logger.info(
        f"seed {seed}: global gains "
        f"kNN={report['adaptation']['global']['knn']['gain']:+.4f}, "
        f"linear={report['adaptation']['global']['linear']['gain']:+.4f}"
    )
    return report, stream.era_names, stream.era_composition()


def _evaluation_sizes(stream: RegimeStream) -> dict[str, Any]:
    """Record actual query counts, including sparse condition cells."""
    return {
        "support": len(stream.eval_sets.probe_support.labels),
        "global_query": len(stream.eval_sets.probe_query.labels),
        "per_condition_query": {str(era): len(data.labels) for era, data in stream.eval_sets.per_era.items()},
    }


def _final_panel(
    method: SSLMethod, frozen: SSLMethod, stream: RegimeStream, panel_seed: int, knn_k: int
) -> dict[str, Any]:
    """Re-evaluate final models on a predeclared support/query reservation without updating them."""
    training_modes = method.training, frozen.training
    method.eval()
    frozen.eval()
    try:
        live_probe = PerEraProbe(stream.eval_sets, probe="linear", knn_k=knn_k)
        frozen_probe = PerEraProbe(stream.eval_sets, probe="linear", knn_k=knn_k)
        final_era = max(stream.era_names)
        live_probe.record(method.encode, final_era)
        frozen_probe.record(frozen.encode, final_era)
        return {
            "panel_seed": panel_seed,
            "global": _global_scores(method, frozen, stream, knn_k),
            "final_per_condition": adaptation.condition_report(live_probe, frozen_probe, stream.era_names)[
                "final_per_condition"
            ],
            "evaluation_sizes": _evaluation_sizes(stream),
        }
    finally:
        method.train(training_modes[0])
        frozen.train(training_modes[1])


def _write_condition_csv(path: Path, seed_reports: list[dict[str, Any]]) -> Path:
    """Write the matched condition-history records in analysis-friendly long form."""
    rows = [{"seed": report["seed"], **row} for report in seed_reports for row in report["conditions"]["history"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_experiment(config: DictConfig, out_dir: Path) -> dict[str, Path]:
    """Run the seed ensemble and write every P1.2 artifact."""
    seeds = [int(seed) for seed in config.seeds]
    reports: list[tuple[int, dict[str, Any]]] = []
    seed_summaries: list[dict[str, Any]] = []
    era_names: dict[int, str] = {}
    composition: dict[int, Any] = {}
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # The attributed corpus is identical across drive seeds. Reuse its decoded tensor cache while
    # each RegimeStream still receives the current seed for its independent within-regime order.
    shared_source = _shared_source(config.data)
    evaluation_config = config.get("evaluation_data")
    shared_evaluation_source = _shared_source(evaluation_config) if evaluation_config else None
    filter_design = _filter_design(config)
    for seed in seeds:
        report, era_names, composition = _run_seed(config, out_dir, seed, shared_source, shared_evaluation_source)
        reports.append((seed, report))
        seed_summary = report["adaptation"]
        seed_summaries.append(seed_summary)
        (reports_dir / f"adaptation_s{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    corpus_paths = deploy_corpus.write_corpus(
        out_dir / "corpus",
        reports=reports,
        era_names=era_names,
        composition=composition,
        manifest={
            "study": "P1.2",
            "git_sha": _git_sha(),
            "n_seeds": len(seeds),
            "eval_every": int(config.eval_every),
            "lr": float(config.optim.lr),
            "update_every": int(config.update_every),
            "train_scope": str(config.get("train_scope", "full")),
            "adapter_bottleneck": int(config.get("adapter_bottleneck", 16)),
            "teacher_retention_weight": float(config.get("teacher_retention_weight", 1.0)),
            "matched_frozen": True,
            "condition_probe": str(config.condition_probe),
            "init_mode": str(config.init.mode),
            "warm": bool(config.get("well")),
            "seed_split": str(config.get("seed_split", "unspecified")),
            "shared_source_across_seeds": True,
            "independent_evaluation_source": shared_evaluation_source is not None,
            "selection": filter_design,
        },
    )
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "study": "P1.2",
        "design": {
            "matched_initial_weights": True,
            "matched_evaluation_examples": True,
            "independent_evaluation_source": shared_evaluation_source is not None,
            "frozen_arm_updated": False,
            "condition_probe": str(config.condition_probe),
            "lr": float(config.optim.lr),
            "update_every": int(config.update_every),
            "train_scope": str(config.get("train_scope", "full")),
            "adapter_bottleneck": int(config.get("adapter_bottleneck", 16)),
            "teacher_retention_weight": float(config.get("teacher_retention_weight", 1.0)),
            "parameter_budget": seed_summaries[0].get("parameter_budget"),
            "init_mode": str(config.init.mode),
            "warm": bool(config.get("well")),
            "seed_split": str(config.get("seed_split", "unspecified")),
            "selection": filter_design,
        },
        "seeds": seed_summaries,
        "aggregate": adaptation.aggregate_seed_reports(seed_summaries),
    }
    comparison["aggregate"]["few_shot"] = adaptation.aggregate_few_shot(seed_summaries)
    comparison_path = out_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    condition_path = _write_condition_csv(out_dir / "condition_accuracy.csv", seed_summaries)
    logger.info(f"wrote P1.2 comparison to {comparison_path}")
    return {**corpus_paths, "comparison": comparison_path, "condition_accuracy": condition_path}


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="adaptation")  # type: ignore[misc]
def main(config: DictConfig) -> None:
    """Run P1.2 in Hydra's configured output directory."""
    run_experiment(config, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
