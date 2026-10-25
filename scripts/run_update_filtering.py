"""Collect the frozen current-only half-rate SSL study with exact reused controls."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn import __version__ as sklearn_version

from cafl4ds import warmup
from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import AttributedImages
from cafl4ds.recurrence import (
    Evaluation,
    arrival_orders,
    evaluation_pool,
    observation_grid,
    restrict_scenes,
    visit_quotas,
)
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.update_filtering import optimize_current, update_mask
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_recurrence_retention import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/adaptation-bdd/recurrence-retention/20261010-ssl-control-restart"
MEMORY = ROOT / "outputs/memory-admission/20261010-static-minred"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-update-filtering-frozen.md"
OUTPUT = ROOT / "outputs/update-filtering/20261010-half-rate"
SNAPSHOTS = (36, 108, 180, 252, 324, 396, 468, 540)


def collect_arm(  # noqa: C901 - explicit clocks, observations and state preservation
    config: DictConfig,
    initial: SSLMethod,
    train: AttributedImages,
    evaluation: Evaluation,
    order: torch.Tensor,
    seed: int,
    schedule: str,
    policy: str,
    frozen: dict[str, Any],
    reference: torch.Tensor,
    out: Path,
    progress: Callable[..., None],
) -> dict[str, Any]:
    """Process every arriving batch; optimize only selected batches without delayed reuse."""
    directory = out / f"{schedule}_{policy}" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    (directory / "decision_states").mkdir()
    mask = update_mask(policy, seed)
    atomic_json(directory / "mask.json", {"policy": policy, "seed": seed, "accepted": mask})
    torch.manual_seed(seed)
    model = copy.deepcopy(initial)
    optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
    trajectory: list[dict[str, Any]] = [copy.deepcopy(frozen)]
    losses: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    online_seconds, updates = 0.0, 0
    grid = set(observation_grid(144))
    started = time.perf_counter()
    for offset in range(0, len(order), 32):
        step = offset // 32 + 1
        tick = time.perf_counter()
        ids = order[offset : offset + 32]
        current = train.images[ids]
        accepted = mask[step - 1]
        observation = optimize_current(model, optimizer, current, accepted)
        if observation is not None:
            updates += 1
            losses.append(
                {"step": step, "optimizer_step": updates, "loss": observation[0], "grad_norm": observation[1]}
            )
        entry: dict[str, Any] = {
            "step": step,
            "current": ids.tolist(),
            "accepted": accepted,
            "optimizer_steps": updates,
            "optimizer_images": updates * 32,
        }
        seconds = time.perf_counter() - tick
        online_seconds += seconds
        entry["online_seconds"] = seconds
        ledger.append(entry)
        if step in SNAPSHOTS:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "model_training": model.training,
                    "arrival_step": step,
                    "optimizer_steps": updates,
                    "mask": mask,
                    "arrival_order_sha256": tensor_hash(order),
                },
                directory / "decision_states" / f"arrival_{step:06d}.pt",
            )
        if step % 32 == 0 or step in grid:
            progress(
                state="training",
                seed=seed,
                schedule=schedule,
                policy=policy,
                arrival_step=step,
                optimizer_steps=updates,
                updates_per_arm=sum(mask),
            )
        if step in grid:
            progress(
                state="evaluating",
                seed=seed,
                schedule=schedule,
                policy=policy,
                arrival_step=step,
                optimizer_steps=updates,
                updates_per_arm=sum(mask),
            )
            record, _ = evaluate(
                model, evaluation, reference, step, step % 144 == 0, int(config.recurrence.fine_tune_steps), directory
            )
            trajectory.append(record)
            if step % 144 == 0:
                warmup.save_well(model, directory / "checkpoints" / f"boundary_{step // 144}.pt")
            atomic_json(directory / "trajectory.json", {"trajectory": trajectory})
            atomic_json(directory / "losses.json", {"losses": losses})
            atomic_json(directory / "ledger.json", {"steps": ledger})
    delta = (
        sum(
            float((p.detach() - q.detach()).square().sum())
            for p, q in zip(model.encoder.parameters(), initial.encoder.parameters(), strict=True)
        )
        ** 0.5
    )
    if not np.isfinite(delta) or delta <= 0 or updates != sum(mask):
        raise ValueError("Finite encoder/update contract failed")
    result = {
        "seed": seed,
        "schedule": schedule,
        "policy": policy,
        "trajectory": trajectory,
        "budget": {
            "arrivals": len(order),
            "updates": updates,
            "optimizer_images": updates * 32,
            "reservoir_seen": 0,
            "reservoir_size": 0,
        },
        "arrival_order_sha256": tensor_hash(order),
        "encoder_delta_l2": delta,
        "online_seconds": online_seconds,
        "block_online_seconds": [sum(row["online_seconds"] for row in ledger[i : i + 144]) for i in range(0, 576, 144)],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(directory / "result.json", result)
    return result


def main() -> None:  # noqa: C901 - frozen complete data/control validation
    """Preflight and reproduce controls before collecting twenty filtered trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status: dict[str, Any] = {"pid": os.getpid(), "completed_arms": 0, "total_arms": 20}

    def progress(**fields: str | int | bool) -> None:
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        progress(stage="checking_prior_audit")
        old_inputs = json.loads((PRIOR / "input_lock.json").read_text())["payload"]
        old_data = json.loads((PRIOR / "data_lock.json").read_text())["payload"]
        verification = json.loads((PRIOR / "independent_verification.json").read_text())
        if not verification["passed"] or verification["arms_checked"] != 20:
            raise ValueError("Prior collection is not fully audited")
        for name, sha in old_inputs["hashes"].items():
            if digest(Path(name)) != sha:
                raise ValueError(f"Prior scientific source/input changed: {name}")
        for name, sha in verification["artifact_hashes"].items():
            if digest(PRIOR / name) != sha:
                raise ValueError(f"Prior scored artifact changed: {name}")
        versions = {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn_version}
        if any(versions[k] != old_inputs[k] for k in versions):
            raise ValueError("Numerical runtime differs from controls")
        memory_inputs = json.loads((MEMORY / "input_lock.json").read_text())
        memory_audit = json.loads((MEMORY / "independent_verification.json").read_text())
        if not memory_audit["passed"] or memory_audit["new_arms_checked"] != 10:
            raise ValueError("Prior memory study audit incomplete")
        for name, sha in memory_inputs["hashes"].items():
            if digest(Path(name)) != sha:
                raise ValueError(f"Memory-study input changed: {name}")
        for name, sha in memory_audit["artifact_hashes"].items():
            if digest(MEMORY / name) != sha:
                raise ValueError(f"Memory-study artifact changed: {name}")
        config = OmegaConf.create(old_inputs["config"])
        settings = old_inputs["config"]["recurrence"]
        seeds = old_inputs["config"]["seeds"]
        torch.set_num_threads(settings["num_threads"])
        torch.use_deterministic_algorithms(True)
        additional = [
            Path(__file__),
            ROOT / "cafl4ds/update_filtering.py",
            ROOT / "scripts/report_update_filtering.py",
            ROOT / "scripts/run_update_filtering_background.py",
            PROTOCOL,
        ]
        hashes = {**old_inputs["hashes"], **{str(p): digest(p) for p in additional}}
        for name in ("input_lock.json", "data_lock.json", "independent_verification.json", "evaluation_arrays.npz"):
            hashes[str(PRIOR / name)] = digest(PRIOR / name)
        hashes.update({str(PRIOR / name): sha for name, sha in verification["artifact_hashes"].items()})
        hashes.update(memory_inputs["hashes"])
        hashes.update({str(MEMORY / name): sha for name, sha in memory_audit["artifact_hashes"].items()})
        hashes[str(MEMORY / "independent_verification.json")] = digest(MEMORY / "independent_verification.json")
        atomic_json(
            out / "input_lock.json",
            {
                "hashes": hashes,
                "config": old_inputs["config"],
                "versions": versions,
                "prior": str(PRIOR),
                "protocol_sha256": digest(PROTOCOL),
            },
        )
        for name in old_inputs["hashes"]:
            p = Path(name)
            if p.suffix in {".py", ".yaml", ".md"}:
                target = out / "source_snapshot" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
        for p in additional:
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        progress(stage="loading_train")
        train, train_indices = restrict_scenes(instantiate(config.data).load(), settings["scene_classes"])
        progress(stage="loading_validation")
        validation, val_indices = restrict_scenes(instantiate(config.evaluation_data).load(), settings["scene_classes"])
        quotas = visit_quotas(
            train, settings["domains"], settings["max_per_class_per_visit"], settings["min_per_class_per_visit"]
        )
        evaluation = evaluation_pool(validation, settings["domains"], settings)
        orders = {s: arrival_orders(train, settings["domains"], quotas, s) for s in seeds}
        data = {
            "quotas": quotas,
            "train_source_indices": train_indices.tolist(),
            "val_source_indices": val_indices.tolist(),
            "evaluation": evaluation.manifest,
            "train_images_sha256": tensor_hash(train.images),
            "val_images_sha256": tensor_hash(validation.images),
            "train_labels_sha256": tensor_hash(train.canary),
            "train_regimes_sha256": tensor_hash(train.era_key),
            "val_labels_sha256": tensor_hash(validation.canary),
            "val_regimes_sha256": tensor_hash(validation.era_key),
            "orders": {str(s): {k: v.tolist() for k, v in row.items()} for s, row in orders.items()},
        }
        if json.loads(json.dumps(data)) != old_data:
            raise ValueError("Complete data allocation differs from controls")
        atomic_json(out / "data_lock.json", data)
        shutil.copyfile(PRIOR / "evaluation_arrays.npz", out / "evaluation_arrays.npz")
        del validation
        initial = _method(config)
        initial_hashes = {k: tensor_hash(v) for k, v in initial.state_dict().items()}
        frozen_dir = out / "frozen"
        frozen_dir.mkdir()
        progress(stage="reproducing_frozen")
        tick = time.perf_counter()
        frozen, reference = evaluate(initial, evaluation, None, 0, True, settings["fine_tune_steps"], frozen_dir)
        baseline_seconds = time.perf_counter() - tick
        if frozen != json.loads((PRIOR / "frozen/result.json").read_text()):
            raise ValueError("Complete frozen readout does not reproduce")
        atomic_json(frozen_dir / "result.json", frozen)
        with torch.random.fork_rng():
            for policy, count in (("always", 4), ("fixed_half", 2), ("random_half", 2)):
                torch.manual_seed(seeds[0])
                model = copy.deepcopy(initial)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                mask = update_mask(policy, seeds[0])
                selected = [i for i, accept in enumerate(mask) if accept][:count]
                for index in selected:
                    ids = orders[seeds[0]]["recurrence"][index * 32 : (index + 1) * 32]
                    optimize_current(model, optimizer, train.images[ids], True)
                before = torch.get_rng_state().clone()
                weights = {k: tensor_hash(v) for k, v in model.state_dict().items()}
                optimize_current(model, optimizer, train.images[ids], False)
                if not torch.equal(before, torch.get_rng_state()) or weights != {
                    k: tensor_hash(v) for k, v in model.state_dict().items()
                }:
                    raise ValueError("Skip changed model or RNG")
        if initial_hashes != {k: tensor_hash(v) for k, v in initial.state_dict().items()}:
            raise ValueError("Disposable preflight changed initialization")
        masks = {str(s): {p: update_mask(p, s) for p in ("fixed_half", "random_half")} for s in seeds}
        atomic_json(out / "mask_manifest.json", masks)
        atomic_json(
            out / "preflight.json",
            {
                "passed": True,
                "data_and_frozen_exact": True,
                "disposable_optimizer_updates": 8,
                "baseline_evaluation_seconds": baseline_seconds,
            },
        )
        progress(stage="compatibility")
        control = collect_arm(
            config,
            initial,
            train,
            evaluation,
            orders[seeds[0]]["recurrence"],
            seeds[0],
            "recurrence",
            "always",
            frozen,
            reference,
            out / "compatibility",
            lambda **f: progress(stage="compatibility", **f),
        )
        expected_dir = PRIOR / "recurrence_no_replay" / f"seed_{seeds[0]}"
        reproduced = out / "compatibility/recurrence_always" / f"seed_{seeds[0]}"
        expected = json.loads((expected_dir / "result.json").read_text())
        for key in ("seed", "trajectory", "budget", "encoder_delta_l2", "arrival_order_sha256"):
            if control[key] != expected[key]:
                raise ValueError(f"Full current-only compatibility differs: {key}")
        actual_losses = json.loads((reproduced / "losses.json").read_text())["losses"]
        stripped = [{k: row[k] for k in ("step", "loss", "grad_norm")} for row in actual_losses]
        if stripped != json.loads((expected_dir / "losses.json").read_text())["losses"]:
            raise ValueError("Compatibility optimizer observations differ")
        for original in expected_dir.glob("predictions_*.npz"):
            with (
                np.load(original, allow_pickle=False) as a,
                np.load(reproduced / original.name, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError("Compatibility predictions differ")
        atomic_json(
            out / "compatibility.json",
            {
                "passed": True,
                "seed": seeds[0],
                "all_predictions_trajectory_losses_exact": True,
                "optimizer_updates": 576,
                "optimizer_images": 18432,
                "online_seconds": control["online_seconds"],
            },
        )
        for position, seed in enumerate(seeds):
            schedules = ("recurrence", "shuffled") if position % 2 == 0 else ("shuffled", "recurrence")
            policies = ("fixed_half", "random_half") if position % 2 == 0 else ("random_half", "fixed_half")
            for schedule in schedules:
                for policy in policies:
                    collect_arm(
                        config,
                        initial,
                        train,
                        evaluation,
                        orders[seed][schedule],
                        seed,
                        schedule,
                        policy,
                        frozen,
                        reference,
                        out,
                        lambda **f: progress(stage="training", **f),
                    )
                    if initial_hashes != {k: tensor_hash(v) for k, v in initial.state_dict().items()}:
                        raise ValueError("Shared initialization changed")
                    status["completed_arms"] += 1
                    progress(stage="arm_completed", seed=seed, schedule=schedule, policy=policy)
        if any(digest(Path(p)) != sha for p, sha in hashes.items()):
            raise ValueError("Locked scientific inputs changed during collection")
        progress(stage="collection_complete", inputs_unchanged=True)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
