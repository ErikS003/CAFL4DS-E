"""Run the locked four-arm BDD recurrence/retention study without changing prior experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from sklearn import __version__ as sklearn_version

from cafl4ds import measurements, warmup
from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import AttributedImages
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.reservoir import ReservoirReplay
from cafl4ds.recurrence import (
    Evaluation,
    arrival_orders,
    evaluation_pool,
    observation_grid,
    prediction_scores,
    recovery_latency,
    restrict_scenes,
    retention_readout,
    visit_quotas,
)
from cafl4ds.ssl.base import SSLMethod
from scripts.confirm_last_block import tensor_hash
from scripts.confirm_replay_advantage import paired_interval
from scripts.evaluate_checkpoint_robustness import digest, features, finetune, lock_json, probe_predictions
from scripts.run_adaptation_experiment import _method

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/adaptation-bdd/recurrence-retention/20261001"
PROTOCOL = ROOT / "docs/experiments/phase1/P1.3-recurrence-retention.md"
ARMS = ["recurrence_no_replay", "recurrence_replay", "shuffled_no_replay", "shuffled_replay"]
READOUTS = ["knn", "linear", "finetune"]


def evaluate(
    method: SSLMethod,
    data: Evaluation,
    reference: torch.Tensor | None,
    step: int,
    boundary: bool,
    fine_steps: int,
    directory: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Read a model with isolated RNG and copied downstream training, preserving SSL state."""
    was_training = method.training
    panels: dict[str, Any] = {}
    predictions = {}
    audits = {}
    try:
        with torch.random.fork_rng():
            torch.manual_seed(51031)
            method.eval()
            encoded = features(method.encoder, data.images)
            query_y, query_d = data.labels[data.query], data.domains[data.query]
            for draw, support in data.supports.items():
                results = {}
                for kind in ("knn", "linear"):
                    predicted = probe_predictions(kind, encoded[support], data.labels[support], encoded[data.query])
                    results[kind] = prediction_scores(predicted, query_y, query_d)
                    predictions[f"{draw}_{kind}"] = predicted.numpy()
                if boundary:
                    tuned, head, audit = finetune(
                        method.encoder, data.images[support], data.labels[support], draw + 25, steps=fine_steps
                    )
                    with torch.no_grad():
                        predicted = head(features(tuned, data.images[data.query])).argmax(1)
                    results["finetune"] = prediction_scores(predicted, query_y, query_d)
                    predictions[f"{draw}_finetune"] = predicted.numpy()
                    audits[str(draw)] = audit
                panels[str(draw)] = results
            health = {}
            for domain, name in enumerate(("A", "B")):
                positions = data.query[query_d == domain]
                z = encoded[positions]
                torch.manual_seed(51031 + domain)
                loss = float(
                    torch.cat(
                        [
                            method.per_sample_loss(data.images[positions[start : start + 32]])
                            for start in range(0, len(positions), 32)
                        ]
                    ).mean()
                )
                health[name] = {
                    "rankme": measurements.rankme(z),
                    "reconstruction_loss": loss,
                    "cosine_drift": 0.0 if reference is None else measurements.cosine_drift(reference[positions], z),
                }
            kinds = READOUTS if boundary else READOUTS[:2]
            scores = {
                kind: {
                    metric: statistics.fmean(p[kind][metric] for p in panels.values())
                    for metric in ("A", "B", "balanced", "micro")
                }
                for kind in kinds
            }
            record = {
                "step": step,
                "boundary": boundary,
                "scores": scores,
                "panels": panels,
                "health": health,
                "fine_tuning_audits": audits,
            }
            np.savez_compressed(directory / f"predictions_{step:06d}.npz", allow_pickle=False, **predictions)
            return record, encoded
    finally:
        method.train(was_training)


def train_arm(  # noqa: C901 - explicit budget, finite-update and checkpoint guards
    config: DictConfig,
    initial: SSLMethod,
    train: AttributedImages,
    evaluation: Evaluation,
    order: torch.Tensor,
    seed: int,
    arm: str,
    frozen: dict[str, Any],
    reference: torch.Tensor,
    out: Path,
    progress: Callable[..., None],
) -> dict[str, Any]:
    """Run a fresh continuous optimizer/memory trajectory; never silently overwrite a partial arm."""
    directory = out / arm / f"seed_{seed}"
    if (directory / "result.json").exists():
        return cast(dict[str, Any], json.loads((directory / "result.json").read_text(encoding="utf-8")))
    if directory.exists():
        raise ValueError(f"Partial arm requires explicit recovery; preserving {directory}")
    directory.mkdir(parents=True)
    (directory / "checkpoints").mkdir()
    torch.manual_seed(seed)
    method = copy.deepcopy(initial)
    optimizer = instantiate(config.optim, params=[p for p in method.parameters() if p.requires_grad])
    replay = None if arm.endswith("no_replay") else ReservoirReplay(256, 16, seed, max_train_batch=32)
    nsteps = len(order) // 32
    visit_steps = nsteps // 4
    grid = set(observation_grid(visit_steps))
    trajectory = [copy.deepcopy(frozen)]
    losses = []
    optimizer_images = 0
    for index, start in enumerate(range(0, len(order), 32)):
        indices = order[start : start + 32]
        step = index + 1
        current = train.images[indices]
        selected = current if replay is None else replay.observe(current, FilterContext(method=method, step=index))
        if len(current) != 32 or len(selected) != 32:
            raise ValueError("Exact batch-budget contract violated")
        method.train()
        optimizer.zero_grad()
        loss = method.training_step(selected)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite SSL loss at update {step}")
        torch.autograd.backward(loss)
        norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer_images += len(selected)
        losses.append({"step": step, "loss": float(loss.detach()), "grad_norm": float(norm)})
        if step % 32 == 0 or step in grid:
            progress(state="training", seed=seed, arm=arm, step=step, updates_per_arm=nsteps)
        if step in grid:
            boundary = step % visit_steps == 0
            progress(state="evaluating", seed=seed, arm=arm, step=step, updates_per_arm=nsteps)
            record, _ = evaluate(
                method, evaluation, reference, step, boundary, int(config.recurrence.fine_tune_steps), directory
            )
            trajectory.append(record)
            if boundary:
                warmup.save_well(method, directory / "checkpoints" / f"boundary_{step // visit_steps}.pt")
            atomic_json(directory / "trajectory.json", {"trajectory": trajectory})
            atomic_json(directory / "losses.json", {"losses": losses})
    metrics = {kind: retention_readout(trajectory, visit_steps, kind) for kind in READOUTS}
    encoder_delta = (
        sum(
            float((p.detach() - q.detach()).square().sum())
            for p, q in zip(method.encoder.parameters(), initial.encoder.parameters(), strict=True)
        )
        ** 0.5
    )
    if encoder_delta == 0:
        raise RuntimeError("Online encoder did not change")
    result = {
        "seed": seed,
        "arm": arm,
        "trajectory": trajectory,
        "metrics": metrics,
        "recovery_linear": recovery_latency(trajectory, visit_steps, float(config.recurrence.recovery_tolerance))
        if arm.startswith("recurrence")
        else None,
        "budget": {
            "arrivals": len(order),
            "updates": nsteps,
            "optimizer_images": optimizer_images,
            "reservoir_seen": 0 if replay is None else replay._seen,
            "reservoir_size": 0 if replay is None else len(replay._buffer),
        },
        "encoder_delta_l2": encoder_delta,
        "arrival_order_sha256": tensor_hash(order),
    }
    if optimizer_images != len(order) or (replay is not None and replay._seen != len(order)):
        raise ValueError("Final exposure audit failed")
    atomic_json(directory / "result.json", result)
    return result


def summarize(out: Path, seeds: list[int]) -> dict[str, Any]:
    """Apply the locked conjunction, reporting all arms/readouts without selecting a winner."""
    rows = {
        arm: [json.loads((out / arm / f"seed_{seed}" / "result.json").read_text(encoding="utf-8")) for seed in seeds]
        for arm in ARMS
    }
    result: dict[str, Any] = {"seeds": seeds, "readouts": {}}
    pairs = [
        ("replay_minus_no_replay", "recurrence_replay", "recurrence_no_replay"),
        ("order_effect_no_replay", "recurrence_no_replay", "shuffled_no_replay"),
        ("order_effect_replay", "recurrence_replay", "shuffled_replay"),
        ("shuffled_replay_effect", "shuffled_replay", "shuffled_no_replay"),
    ]
    for kind in READOUTS:
        means = {
            arm: {
                metric: 100 * statistics.fmean(r["metrics"][kind][metric] for r in runs)
                for metric in runs[0]["metrics"][kind]
            }
            for arm, runs in rows.items()
        }
        contrasts = {
            name: {
                metric: paired_interval(
                    [
                        a["metrics"][kind][metric] - b["metrics"][kind][metric]
                        for a, b in zip(rows[first], rows[second], strict=True)
                    ]
                )
                for metric in ("interference", "final_balanced", "bwt", "forgetting")
            }
            for name, first, second in pairs
        }
        absolute = {
            arm: {
                metric: paired_interval([r["metrics"][kind][metric] for r in runs])
                for metric in ("interference", "final_vs_warm")
            }
            for arm, runs in rows.items()
        }
        result["readouts"][kind] = {"means_percent_or_pp": means, "contrasts": contrasts, "against_zero": absolute}
    primary = result["readouts"]["finetune"]
    result["primary_gates"] = {
        "degradation_present": primary["against_zero"]["recurrence_no_replay"]["interference"]["ci95_pp"][0] > 0,
        "replay_reduces_interference": primary["contrasts"]["replay_minus_no_replay"]["interference"]["ci95_pp"][1] < 0,
        "final_accuracy_noninferior": primary["contrasts"]["replay_minus_no_replay"]["final_balanced"]["ci95_pp"][0]
        > -1,
    }
    result["retention_supported"] = all(result["primary_gates"].values())
    result["replay_final_advantage_vs_warm"] = (
        primary["against_zero"]["recurrence_replay"]["final_vs_warm"]["ci95_pp"][0] > 0
    )
    result["recovery_by_seed"] = {arm: [r["recovery_linear"] for r in rows[arm]] for arm in ARMS[:2]}
    atomic_json(out / "summary.json", result)
    lines = [
        "# BDD recurrence / retention",
        "",
        "Nominal paired 95% intervals across five seeds; support draws averaged within seed.",
        "Validation and warm well reused; attribute-ordered images, not consecutive video.",
        "",
        f"Primary retention conjunction supported: {result['retention_supported']}",
        f"Individual gates: {result['primary_gates']}",
        "",
        "| Readout | Contrast | Interference change, pp [95% CI] | Final accuracy change, pp [95% CI] |",
        "|---|---|---|---|",
    ]
    for kind, data in result["readouts"].items():
        for contrast, scores in data["contrasts"].items():
            cells = [
                f"{scores[m]['mean_pp']:+.2f} [{scores[m]['ci95_pp'][0]:+.2f}, {scores[m]['ci95_pp'][1]:+.2f}]"
                for m in ("interference", "final_balanced")
            ]
            lines.append(f"| {kind} | {contrast} | {' | '.join(cells)} |")
    lines += [
        "",
        "Lower interference is better. Shuffled temporal contrasts are sample-clock controls, not task boundaries.",
        "See summary.json for all individual gates, final-vs-warm effects, and censored recovery latencies.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure = Figure(figsize=(13, 8), constrained_layout=True)
    axes = figure.subplots(2, 3)
    for row, domain in enumerate(("A", "B")):
        for col, kind in enumerate(READOUTS):
            ax = axes[row, col]
            for arm, runs in rows.items():
                curves = [
                    [(r["step"], r["scores"][kind][domain]) for r in run["trajectory"] if kind in r["scores"]]
                    for run in runs
                ]
                x = np.array([p[0] for p in curves[0]])
                y = np.array([[p[1] for p in curve] for curve in curves]) * 100
                ax.plot(x, y.mean(0), label=arm)
                for curve in y:
                    ax.plot(x, curve, alpha=0.12, linewidth=0.6)
            ax.axhline(y[0, 0], color="black", linestyle="--", linewidth=0.8, label="warm frozen")
            total = rows[ARMS[0]][0]["budget"]["updates"]
            for boundary in (total / 4, total / 2, total * 3 / 4):
                ax.axvline(boundary, color="gray", alpha=0.3)
            ax.set(title=f"{domain}: {kind}", xlabel="SSL updates", ylabel="Class-balanced accuracy (%)")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("A1 → B1 → A2 → B2 • thin seed curves, bold means • shuffled controls share the sample clock")
    figure.savefig(out / "trajectories.png", dpi=150)
    return result


def main() -> None:  # noqa: C901 - explicit provenance and run-state checks
    """Lock inputs, construct diets from counts, then run four matched arms per fresh seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prior-run", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(ROOT / "cafl4ds/configs"), version_base=None):
        config = compose(config_name="adaptation_bdd_recurrence", overrides=args.override)
    config.well = str((ROOT / str(config.well)).resolve())
    settings = cast(dict[str, Any], OmegaConf.to_container(config.recurrence, resolve=True))
    seeds = [int(seed) for seed in config.seeds]
    if len(seeds) != 5 or len(set(seeds)) != 5 or config.device != "cpu" or int(config.batch_size) != 32:
        raise ValueError("This protocol requires five distinct seeds, CPU and batches of 32")
    torch.set_num_threads(int(settings["num_threads"]))
    torch.use_deterministic_algorithms(True)
    started = time.time()
    state: dict[str, Any] = {"pid": os.getpid(), "completed_arms": 0, "total_arms": 20}

    def progress(**fields: str | int | bool) -> None:
        state.update(fields, elapsed_seconds=time.time() - started)
        atomic_json(out / "progress.json", state)
        print(json.dumps(state), flush=True)

    try:
        progress(state="locking_inputs")
        source_paths = [
            *sorted((ROOT / "cafl4ds").rglob("*.py")),
            *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
            *sorted((ROOT / "scripts").glob("*.py")),
            PROTOCOL,
            Path(config.well),
        ]
        hashes = {str(p): digest(p) for p in source_paths}
        lock_json(
            out / "input_lock.json",
            {
                "hashes": hashes,
                "config": OmegaConf.to_container(config, resolve=True),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn_version,
            },
        )
        OmegaConf.save(config, out / "config.yaml")
        progress(state="loading_train")
        train, train_source_indices = restrict_scenes(instantiate(config.data).load(), list(settings["scene_classes"]))
        progress(state="loading_validation")
        validation, val_source_indices = restrict_scenes(
            instantiate(config.evaluation_data).load(), list(settings["scene_classes"])
        )
        if train.canary_names != validation.canary_names:
            raise ValueError("Train and evaluation class mappings differ")
        names = list(settings["domains"])
        quotas = visit_quotas(
            train, names, int(settings["max_per_class_per_visit"]), int(settings["min_per_class_per_visit"])
        )
        evaluation = evaluation_pool(validation, names, settings)
        orders = {seed: arrival_orders(train, names, quotas, seed) for seed in seeds}
        lock_json(
            out / "data_lock.json",
            {
                "quotas": quotas,
                "train_source_indices": train_source_indices.tolist(),
                "val_source_indices": val_source_indices.tolist(),
                "evaluation": evaluation.manifest,
                "train_images_sha256": tensor_hash(train.images),
                "val_images_sha256": tensor_hash(validation.images),
                "train_labels_sha256": tensor_hash(train.canary),
                "train_regimes_sha256": tensor_hash(train.era_key),
                "val_labels_sha256": tensor_hash(validation.canary),
                "val_regimes_sha256": tensor_hash(validation.era_key),
                "orders": {
                    str(seed): {key: order.tolist() for key, order in rows.items()} for seed, rows in orders.items()
                },
            },
        )
        del validation
        arrays_path = out / "evaluation_arrays.npz"
        arrays = {"labels": evaluation.labels.numpy(), "domains": evaluation.domains.numpy(),
            "query": evaluation.query.numpy()}
        if arrays_path.exists():
            with np.load(arrays_path, allow_pickle=False) as saved:
                if any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                    raise ValueError("Evaluation array snapshot differs")
        else:
            np.savez_compressed(arrays_path, allow_pickle=False, **arrays)
        initial = _method(config)
        baseline_dir = out / "frozen"
        baseline_dir.mkdir(exist_ok=True)
        if (baseline_dir / "result.json").exists():
            frozen = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
            reference = features(initial.encoder, evaluation.images)
        else:
            progress(state="evaluating_warm_frozen")
            evaluation_started = time.perf_counter()
            frozen, reference = evaluate(
                initial, evaluation, None, 0, True, int(settings["fine_tune_steps"]), baseline_dir
            )
            atomic_json(baseline_dir / "result.json", frozen)
            atomic_json(out / "baseline_timing.json", {"seconds": time.perf_counter() - evaluation_started})
        if args.prior_run is not None:
            previous = args.prior_run.resolve()
            old_inputs = json.loads((previous / "input_lock.json").read_text(encoding="utf-8"))["payload"]
            old_data = json.loads((previous / "data_lock.json").read_text(encoding="utf-8"))["payload"]
            new_data = json.loads((out / "data_lock.json").read_text(encoding="utf-8"))["payload"]
            if old_inputs["config"] != OmegaConf.to_container(config, resolve=True) or old_data != new_data:
                raise ValueError("Restart must reproduce the original configuration and entire data allocation exactly")
            if any(old_inputs[key] != current for key, current in (
                ("torch", torch.__version__), ("numpy", np.__version__), ("sklearn", sklearn_version),
            )):
                raise ValueError("Restart numerical runtime differs from original")
            previous_frozen = json.loads((previous / "frozen/result.json").read_text(encoding="utf-8"))
            if frozen != previous_frozen:
                raise ValueError("Restart frozen evaluation differs from preserved attempt")
            atomic_json(out / "restart_provenance.json", {
                "previous_run": str(previous), "configuration_and_data_exact": True,
                "frozen_evaluation_exact": True, "partial_optimizer_state_reused": False,
                "prior_hashes": {str(previous / name): digest(previous / name) for name in (
                    "input_lock.json", "data_lock.json", "progress.json", "frozen/result.json",
                )},
                "changed_preexisting_sources": [name for name, sha in old_inputs["hashes"].items()
                    if name in hashes and hashes[name] != sha],
            })
        if args.preflight_only:
            initial_hash = {key: tensor_hash(value) for key, value in initial.state_dict().items()}
            timings = {}
            with torch.random.fork_rng():
                for arm in ARMS[:2]:
                    torch.manual_seed(seeds[0])
                    copied = copy.deepcopy(initial)
                    optimizer = instantiate(config.optim, params=[p for p in copied.parameters() if p.requires_grad])
                    replay = (
                        None if arm.endswith("no_replay") else ReservoirReplay(256, 16, seeds[0], max_train_batch=32)
                    )
                    tick = time.perf_counter()
                    for offset in range(0, 128, 32):
                        indices = orders[seeds[0]]["recurrence"][offset : offset + 32]
                        current = train.images[indices]
                        images = current if replay is None else replay.observe(current, FilterContext(copied, 0))
                        optimizer.zero_grad()
                        loss = copied.training_step(images)
                        if len(images) != 32 or not torch.isfinite(loss):
                            raise RuntimeError("Disposable preflight update failed")
                        torch.autograd.backward(loss)
                        torch.nn.utils.clip_grad_norm_(copied.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                    timings[arm] = (time.perf_counter() - tick) / 4
            if initial_hash != {key: tensor_hash(value) for key, value in initial.state_dict().items()}:
                raise ValueError("Preflight modified the shared initialization")
            baseline_seconds = json.loads((out / "baseline_timing.json").read_text())["seconds"]
            steps = len(orders[seeds[0]]["recurrence"]) // 32
            estimate = 20 * (steps * max(timings.values()) + 16 * baseline_seconds)
            atomic_json(out / "preflight.json", {
                "passed": True, "disposable_optimizer_updates": 8, "collection_updates": 0,
                "seconds_per_update": timings, "baseline_evaluation_seconds": baseline_seconds,
                "conservative_projected_collection_seconds": estimate,
                "updates_per_arm": steps, "arms": 20, "initialization_unchanged": True,
                "evaluation_arrays_sha256": digest(out / "evaluation_arrays.npz"),
                "note": "All16 evaluation points costed as full fine-tuning boundaries; projection is approximate.",
            })
            progress(state="preflight_passed", updates_per_arm=steps)
            return
        for position, seed in enumerate(seeds):
            for arm in ARMS if position % 2 == 0 else ARMS[::-1]:
                order = orders[seed]["recurrence" if arm.startswith("recurrence") else "shuffled"]
                train_arm(config, initial, train, evaluation, order, seed, arm, frozen, reference, out, progress)
                state["completed_arms"] += 1
                progress(state="arm_completed", seed=seed, arm=arm)
        if any(digest(Path(path)) != expected for path, expected in hashes.items()):
            raise ValueError("Locked inputs changed during the run")
        summarize(out, seeds)
        progress(state="completed", inputs_unchanged=True)
    except Exception as error:
        progress(state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
