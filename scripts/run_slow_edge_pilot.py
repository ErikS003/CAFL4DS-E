"""Bounded open-loop qualification and paired alpha branches; no MPC efficacy claim."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import AttributedImages, SyntheticAttributeSource
from cafl4ds.data.bdd_subset import BDDSubsetIndex
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.filters.reservoir import ReservoirReplay
from cafl4ds.filters.steerable import SteerableReplay, fit_anchors
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.recurrence import (
    Evaluation,
    arrival_orders,
    evaluation_pool,
    prediction_scores,
    restrict_scenes,
    visit_quotas,
)
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.factory import build_mae
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest, features, probe_predictions
from scripts.run_adaptation_experiment import _method

ROOT = Path(__file__).resolve().parents[1]


def readout(method: SSLMethod, data: Evaluation) -> dict[str, Any]:
    """Read external development probes without modifying model or training randomness."""
    modes = {module: module.training for module in method.modules()}
    try:
        with torch.random.fork_rng(), torch.no_grad():
            z = features(method.encoder, data.images)
            support = next(iter(data.supports.values()))
            return {
                kind: prediction_scores(
                    probe_predictions(kind, z[support], data.labels[support], z[data.query]),
                    data.labels[data.query],
                    data.domains[data.query],
                )
                for kind in ("knn", "linear")
            }
    finally:
        for module, training in modes.items():
            module.training = training


def update(
    method: SSLMethod,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    step: int,
    selector: Filter | ReservoirReplay | None,
) -> dict[str, Any]:
    """Apply one label-free update, auditing the exact optimizer image count."""
    method.train()
    ctx = FilterContext(method, step)
    if isinstance(selector, Filter):
        selected = selector.select(StreamBatch(images, era=0, step=step), ctx)
    elif selector is not None:
        selected = selector.observe(images, ctx)
    else:
        selected = images
    if len(images) != 32 or len(selected) != 32:
        raise ValueError("Expected exactly 32 arrivals and optimizer images")
    optimizer.zero_grad()
    loss = method.training_step(selected)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    torch.autograd.backward(loss)
    norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm), "optimizer_images": len(selected)}


def optimizer_for(method: SSLMethod) -> torch.optim.AdamW:
    """Pin the qualified vehicle's learning recipe for all paired arms."""
    return torch.optim.AdamW(method.parameters(), lr=1e-4, weight_decay=0.05)


def qualify(
    initial: SSLMethod, images: torch.Tensor, evaluation: Evaluation, seed: int, visit_steps: int
) -> dict[str, Any]:
    """Measure a bounded recurrence and shuffled control; no confirmation threshold fitting."""
    rows: dict[str, Any] = {}
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed + 1))
    for shuffled in (False, True):
        arrivals = images[order] if shuffled else images
        for use_replay in (False, True):
            torch.manual_seed(seed)
            method = copy.deepcopy(initial)
            optimizer = optimizer_for(method)
            replay = ReservoirReplay(256, 16, seed, max_train_batch=32) if use_replay else None
            trajectory: list[dict[str, Any]] = [{"step": 0, "scores": readout(method, evaluation)}]
            exposures = 0
            for step, start in enumerate(range(0, len(arrivals), 32)):
                batch = arrivals[start : start + 32]
                exposures += update(method, optimizer, batch, step, replay)["optimizer_images"]
                if (step + 1) % visit_steps == 0:
                    trajectory.append({"step": step + 1, "scores": readout(method, evaluation)})
            name = f"{'shuffled' if shuffled else 'recurrence'}_{'replay' if use_replay else 'no_replay'}"
            rows[name] = {
                "trajectory": trajectory,
                "optimizer_images": exposures,
                "arrivals": len(images),
                "updates": len(images) // 32,
                "interference": {
                    kind: statistics.fmean(
                        [
                            trajectory[1]["scores"][kind]["A"] - trajectory[2]["scores"][kind]["A"],
                            trajectory[2]["scores"][kind]["B"] - trajectory[3]["scores"][kind]["B"],
                        ]
                    )
                    for kind in ("knn", "linear")
                },
            }
    return rows


def branches(  # noqa: C901 - explicit snapshot, paired randomization and budget audits
    initial: SSLMethod,
    actuator: SteerableReplay,
    images: torch.Tensor,
    evaluation: Evaluation,
    seed: int,
    prefix_steps: int,
    window: int,
    directory: Path,
) -> dict[str, Any]:
    """Branch three actions from one exact learning/memory/RNG state."""
    torch.manual_seed(seed)
    common = copy.deepcopy(initial)
    common_optimizer = optimizer_for(common)
    shared = copy.deepcopy(actuator)
    shared.set_action(0.5)
    for step in range(prefix_steps):
        update(common, common_optimizer, images[step * 32 : (step + 1) * 32], step, shared)
    observed = readout(common, evaluation)
    snapshot: dict[str, Any] = {
        "method": copy.deepcopy(common.state_dict()),
        "optimizer": copy.deepcopy(common_optimizer.state_dict()),
        "actuator": shared.state_dict(),
        "rng": torch.get_rng_state().clone(),
        "prefix_steps": prefix_steps,
    }
    torch.save(snapshot, directory / "branch_snapshot.pt")
    actions = [0.0, 0.5, 1.0]
    permutation = torch.randperm(3, generator=torch.Generator().manual_seed(seed + 17)).tolist()
    rows: dict[str, Any] = {}
    for position in permutation:
        alpha = actions[position]
        method = copy.deepcopy(common)
        optimizer = optimizer_for(method)
        optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        selected = copy.deepcopy(shared)
        selected.load_state_dict(snapshot["actuator"])
        selected.set_action(alpha)
        torch.set_rng_state(snapshot["rng"])
        decisions, health = [], [{"step": prefix_steps, "scores": observed}]
        exposures = 0
        for offset in range(2 * window):
            step = prefix_steps + offset
            exposures += update(method, optimizer, images[step * 32 : (step + 1) * 32], step, selected)[
                "optimizer_images"
            ]
            decisions.append(copy.deepcopy(selected.last))
            if (offset + 1) % window == 0:
                health.append({"step": step + 1, "scores": readout(method, evaluation)})
        rows[str(alpha)] = {
            "decisions": decisions,
            "trajectory": health,
            "optimizer_images": exposures,
            "reservoir_hash": tensor_hash(torch.stack(selected.reservoir._buffer)),
            "reservoir_seen": selected.reservoir._seen,
        }
        torch.save({"method": method.state_dict(), "actuator": selected.state_dict()}, directory / f"alpha_{alpha}.pt")
    if len({row["reservoir_hash"] for row in rows.values()}) != 1:
        raise ValueError("Actions changed memory admission or reservoir draws")
    overlaps = []
    for first, second in zip(rows["0.0"]["decisions"], rows["1.0"]["decisions"], strict=True):
        a, b = set(first["current_indices"]), set(second["current_indices"])
        overlaps.append(len(a & b) / len(a | b))
    return {
        "execution_order": [actions[p] for p in permutation],
        "common_observation": observed,
        "alpha_endpoints_mean_jaccard": statistics.fmean(overlaps),
        "matched_reservoir": True,
        "rows": rows,
    }


def load_inputs(
    data: str,
    *,
    train_per_cell: int | None = None,
    evaluation_per_cell: int | None = None,
    stream_seed: int = 62003,
    excluded_paths: set[str] | None = None,
) -> tuple[SSLMethod, torch.Tensor, torch.Tensor, Evaluation, dict[str, Any]]:
    """Load disjoint warm/train/evaluation inputs; preserve the BDD train partition."""
    settings = {
        "pool_seed": 62001,
        "support_per_domain_class": 12,
        "query_per_domain_class": 24,
        "min_query_per_domain_class": 12,
        "draw_seeds": [62003],
        "shots_per_class": 12,
    }
    if train_per_cell is not None and (train_per_cell < 64 or train_per_cell % 64):
        raise ValueError("Training cell size must be a multiple of 64")
    if evaluation_per_cell is not None:
        if evaluation_per_cell < 36:
            raise ValueError("Evaluation cells require at least 36 images")
        settings["query_per_domain_class"] = evaluation_per_cell - 12
    if data == "synthetic":
        stream_count = 96 if train_per_cell is None else train_per_cell
        source = SyntheticAttributeSource(
            num_regimes=2,
            num_canary_classes=2,
            per_cell=32 + stream_count + max(64, evaluation_per_cell or 36),
            img_size=16,
            long_tail=False,
            seed=62000,
        ).load()
        # Same class patterns, disjoint images in every role.
        roles: list[torch.Tensor] = []
        for domain in range(2):
            for label in range(2):
                roles.append(((source.era_key == domain) & (source.canary == label)).nonzero().flatten())
        warm_indices = torch.cat([r[:32] for r in roles])
        stream_indices = torch.cat([r[32 : 32 + stream_count] for r in roles])
        validation_indices = torch.cat([r[32 + stream_count :] for r in roles])

        def subset(indices: torch.Tensor) -> AttributedImages:
            return type(source)(
                source.images[indices],
                source.era_key[indices],
                source.canary[indices],
                source.regime_order,
                source.regime_names,
                source.canary_names,
            )

        train, validation = subset(stream_indices), subset(validation_indices)
        initial: SSLMethod = build_mae(TinyViTEncoder(img_size=16, embed_dim=24, depth=1, num_heads=3), decoder_dim=24)
        warm = source.images[warm_indices]
        names = ["regime0", "regime1"]
        provenance: dict[str, Any] = {"data": data, "purpose": "engineering only; random initialization"}
    else:
        with initialize_config_dir(config_dir=str(ROOT / "cafl4ds/configs"), version_base=None):
            config = compose(config_name="adaptation_bdd_recurrence")
        config.well = str((ROOT / str(config.well)).resolve())
        initial = _method(config)
        warm_config = copy.deepcopy(config.data)
        warm_config.partition_role = "warm"
        index = BDDSubsetIndex()
        warm_data, warm_paths = index.load(instantiate(warm_config), seed=62000, total=1024)
        warm = warm_data.images
        names = list(config.recurrence.domains)
        scene_names = list(config.recurrence.scene_classes)
        train, train_paths = index.load(
            instantiate(config.data),
            seed=stream_seed,
            per_cell=train_per_cell or 64,
            scenes=scene_names,
            domains=names,
            excluded_paths=excluded_paths,
        )
        validation, val_paths = index.load(
            instantiate(config.evaluation_data),
            seed=62001,
            per_cell=evaluation_per_cell or 36,
            scenes=scene_names,
            domains=names,
        )
        train, _ = restrict_scenes(train, scene_names)
        validation, _ = restrict_scenes(validation, scene_names)
        if set(warm_paths) & set(train_paths) or set(val_paths) & (set(warm_paths) | set(train_paths)):
            raise ValueError("BDD role overlap")
        provenance = {
            "data": data,
            "well_sha256": digest(Path(config.well)),
            "config": OmegaConf.to_container(config, resolve=True),
            "purpose": "bounded development; not locked recurrence confirmation",
            "warm_paths": warm_paths,
            "train_paths": train_paths,
            "validation_paths": val_paths,
        }
    evaluation = evaluation_pool(validation, names, settings)
    quotas = visit_quotas(train, names, maximum=32 if train_per_cell is None else train_per_cell // 2, minimum=32)
    # Candidate ids, not evaluation labels, supply paired arrival order.
    orders = arrival_orders(train, names, quotas, stream_seed)["recurrence"]
    provenance.update(
        quotas=quotas,
        evaluation=evaluation.manifest,
        train_source_order=orders.tolist(),
        arrivals_sha256=tensor_hash(train.images[orders]),
        warm_sha256=tensor_hash(warm),
        evaluation_sha256=tensor_hash(evaluation.images),
        stream_seed=stream_seed,
        excluded_stream_paths=len(excluded_paths or set()),
    )
    return initial, warm, train.images[orders].clone(), evaluation, provenance


def main() -> None:
    """Run a short development pilot in a new directory and preserve every prior artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["synthetic", "bdd"], default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[6203, 6211, 6217])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Pilot seeds must be nonempty and distinct")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(62000)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    source_paths = [
        *sorted((ROOT / "cafl4ds").rglob("*.py")),
        *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    atomic_json(out / "progress.json", {"state": "loading", "data": args.data})
    try:
        initial, warm, arrivals, evaluation, provenance = load_inputs(args.data)
        centers, target = fit_anchors(features(initial.encoder, warm), seed=62000)
        del warm
        atomic_json(
            out / "inputs.json",
            {
                **provenance,
                "source_hashes": hashes,
                "seeds": args.seeds,
                "torch": torch.__version__,
                "centroids": centers.tolist(),
                "target": target.tolist(),
            },
        )
        visit_steps = len(arrivals) // (4 * 32)
        prefix, window = visit_steps + 1, 2
        if prefix + 2 * window > len(arrivals) // 32:
            raise ValueError("Insufficient arrivals for branch horizon")
        results = []
        for seed in args.seeds:
            directory = out / f"seed_{seed}"
            directory.mkdir()
            atomic_json(out / "progress.json", {"state": "qualification", "seed": seed})
            qualification = qualify(initial, arrivals, evaluation, seed, visit_steps)
            atomic_json(directory / "qualification.json", qualification)
            atomic_json(out / "progress.json", {"state": "branches", "seed": seed})
            actuator = SteerableReplay(initial.encoder, centers, target, seed=seed)
            response = branches(initial, actuator, arrivals, evaluation, seed, prefix, window, directory)
            result = {"seed": seed, "qualification": qualification, "branches": response}
            atomic_json(directory / "result.json", result)
            results.append(result)
        summary = {
            "status": "development_only",
            "data": args.data,
            "seeds": args.seeds,
            "visit_updates": visit_steps,
            "branch_prefix_updates": prefix,
            "window": window,
            "horizon_windows": 2,
            "elapsed_seconds": time.time() - started,
            "results": results,
            "limitations": [
                "Short horizon; not an MPC comparison",
                "Refit probes, not fixed-head controller signals",
                "Shared warm well, data and evaluation; seeds vary training randomness",
                "A single branch point just after first shift",
                "No fine-tuning or final confirmation pool",
            ],
        }
        if any(digest(ROOT / path) != expected for path, expected in hashes.items()):
            raise ValueError("Pilot sources changed during execution")
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {"state": "completed", "elapsed_seconds": time.time() - started})
        print(json.dumps({"output": str(out), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    except Exception as error:
        atomic_json(out / "progress.json", {"state": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
