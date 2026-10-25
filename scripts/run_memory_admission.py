"""Run the frozen static memory-admission study with reused audited recurrence controls."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn import __version__ as sklearn_version

from cafl4ds import warmup
from cafl4ds.artifacts import atomic_json
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.memory_admission import MemoryAdmissionReplay
from cafl4ds.filters.reservoir import ReservoirReplay
from cafl4ds.recurrence import arrival_orders, evaluation_pool, observation_grid, restrict_scenes, visit_quotas
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.report_memory_admission import membership
from scripts.run_adaptation_experiment import _method
from scripts.run_recurrence_retention import evaluate, train_arm

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/adaptation-bdd/recurrence-retention/20261010-ssl-control-restart"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-memory-admission-frozen.md"
OUTPUT = ROOT / "outputs/memory-admission/20261010-static-minred"


def main() -> None:  # noqa: C901 - explicit frozen compatibility and complete training guards
    """Verify controls, reproduce one arm, then collect the ten declared filter trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status: dict[str, Any] = {"pid": os.getpid(), "completed_arms": 0, "total_arms": 10}

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
        config = OmegaConf.create(old_inputs["config"])
        settings = old_inputs["config"]["recurrence"]
        seeds = old_inputs["config"]["seeds"]
        torch.set_num_threads(settings["num_threads"])
        torch.use_deterministic_algorithms(True)
        additional = [
            Path(__file__),
            ROOT / "cafl4ds/filters/memory_admission.py",
            ROOT / "scripts/report_memory_admission.py",
            ROOT / "scripts/run_memory_admission_background.py",
            PROTOCOL,
        ]
        hashes = {**old_inputs["hashes"], **{str(p): digest(p) for p in additional}}
        for name in ("input_lock.json", "data_lock.json", "independent_verification.json", "evaluation_arrays.npz"):
            hashes[str(PRIOR / name)] = digest(PRIOR / name)
        hashes.update({str(PRIOR / name): sha for name, sha in verification["artifact_hashes"].items()})
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
        timing = {}
        with torch.random.fork_rng():
            for policy in ("reservoir", "minred"):
                torch.manual_seed(seeds[0])
                model = copy.deepcopy(initial)
                buffer = MemoryAdmissionReplay(
                    policy, seeds[0], copy.deepcopy(initial.encoder) if policy == "minred" else None
                )
                old_buffer = ReservoirReplay(256, 16, seeds[0], max_train_batch=32)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                tick = time.perf_counter()
                for offset in range(0, 128, 32):
                    indices = orders[seeds[0]]["recurrence"][offset : offset + 32]
                    current = train.images[indices]
                    selected = buffer.observe_with_ids(current, indices)
                    if policy == "reservoir" and not torch.equal(
                        selected, old_buffer.observe(current, FilterContext(model, offset // 32))
                    ):
                        raise ValueError("Traced reservoir differs from original selector")
                    optimizer.zero_grad()
                    loss = model.training_step(selected)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite disposable preflight loss")
                    torch.autograd.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                timing[policy] = (time.perf_counter() - tick) / 4
        if initial_hashes != {k: tensor_hash(v) for k, v in initial.state_dict().items()}:
            raise ValueError("Disposable preflight changed shared initialization")
        selection_buffer = MemoryAdmissionReplay("minred", seeds[0], copy.deepcopy(initial.encoder))
        selection_ledger, selection_raw = [], []
        overflow_timings = []
        prefix = orders[seeds[0]]["recurrence"][:320]
        for offset in range(0, 320, 32):
            ids = prefix[offset : offset + 32]
            tick = time.perf_counter()
            selection_buffer.observe_with_ids(train.images[ids], ids)
            if offset >= 256:
                overflow_timings.append(time.perf_counter() - tick)
            selection_ledger.append({"step": offset // 32 + 1, **selection_buffer.last_trace})
            selection_raw.append(selection_buffer.last_features.numpy())
        membership(prefix.tolist(), np.concatenate(selection_raw), selection_ledger, seeds[0])
        atomic_json(
            out / "preflight.json",
            {
                "passed": True,
                "data_and_frozen_exact": True,
                "disposable_optimizer_updates": 8,
                "selection_only_arrivals_independently_reconstructed": 320,
                "overflow_selection_seconds_per_batch": float(np.mean(overflow_timings)),
                "seconds_per_update": timing,
                "projected_collection_seconds": 11
                * (576 * (max(timing.values()) + float(np.mean(overflow_timings))) + 16 * baseline_seconds),
                "note": "Approximate conservative projection; all observations costed as full fine-tuning boundaries.",
            },
        )
        progress(stage="reproducing_reservoir", seed=seeds[0])
        control = train_arm(
            config,
            initial,
            train,
            evaluation,
            orders[seeds[0]]["recurrence"],
            seeds[0],
            "recurrence_replay",
            frozen,
            reference,
            out / "compatibility",
            lambda **f: progress(stage="compatibility", **f),
        )
        expected = json.loads((PRIOR / "recurrence_replay" / f"seed_{seeds[0]}" / "result.json").read_text())
        if control != expected:
            raise ValueError("Full compatibility control differs; refusing control reuse")
        for original in (PRIOR / "recurrence_replay" / f"seed_{seeds[0]}").glob("predictions_*.npz"):
            reproduced = out / "compatibility/recurrence_replay" / f"seed_{seeds[0]}" / original.name
            with np.load(original, allow_pickle=False) as a, np.load(reproduced, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError("Compatibility predictions differ")
        atomic_json(
            out / "compatibility.json",
            {
                "passed": True,
                "seed": seeds[0],
                "all_predictions_and_result_exact": True,
                "optimizer_updates": 576,
                "optimizer_images": 18432,
            },
        )
        for position, seed in enumerate(seeds):
            for schedule in ("recurrence", "shuffled") if position % 2 == 0 else ("shuffled", "recurrence"):
                directory = out / f"{schedule}_minred" / f"seed_{seed}"
                (directory / "checkpoints").mkdir(parents=True)
                torch.manual_seed(seed)
                model = copy.deepcopy(initial)
                buffer = MemoryAdmissionReplay("minred", seed, copy.deepcopy(initial.encoder))
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                trajectory, losses, ledger, descriptors = [copy.deepcopy(frozen)], [], [], []
                order = orders[seed][schedule]
                grid = set(observation_grid(144))
                tick = time.perf_counter()
                for offset in range(0, len(order), 32):
                    step = offset // 32 + 1
                    indices = order[offset : offset + 32]
                    selected = buffer.observe_with_ids(train.images[indices], indices)
                    ledger.append({"step": step, **buffer.last_trace})
                    descriptors.append(buffer.last_features.numpy())
                    model.train()
                    optimizer.zero_grad()
                    loss = model.training_step(selected)
                    if len(selected) != 32 or not torch.isfinite(loss):
                        raise RuntimeError("Finite/exposure contract failed")
                    torch.autograd.backward(loss)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append({"step": step, "loss": float(loss.detach()), "grad_norm": float(norm)})
                    if step % 32 == 0 or step in grid:
                        progress(stage="training", seed=seed, schedule=schedule, step=step, updates_per_arm=576)
                    if step in grid:
                        record, _ = evaluate(
                            model, evaluation, reference, step, step % 144 == 0, settings["fine_tune_steps"], directory
                        )
                        trajectory.append(record)
                        if step % 144 == 0:
                            warmup.save_well(model, directory / "checkpoints" / f"boundary_{step // 144}.pt")
                        atomic_json(directory / "trajectory.json", {"trajectory": trajectory})
                        atomic_json(directory / "ledger.json", {"steps": ledger})
                        atomic_json(directory / "losses.json", {"losses": losses})
                if initial_hashes != {k: tensor_hash(v) for k, v in initial.state_dict().items()}:
                    raise ValueError("Shared initial weights changed")
                if buffer.reference is None or any(
                    not torch.equal(v, initial.encoder.state_dict()[k])
                    for k, v in buffer.reference.state_dict().items()
                ):
                    raise ValueError("Fixed reference changed")
                encoder_delta = (
                    sum(
                        float((p.detach() - q.detach()).square().sum())
                        for p, q in zip(model.encoder.parameters(), initial.encoder.parameters(), strict=True)
                    )
                    ** 0.5
                )
                if not np.isfinite(encoder_delta) or encoder_delta <= 0:
                    raise ValueError("SSL encoder did not make finite updates")
                np.save(directory / "reference_features.npy", np.concatenate(descriptors), allow_pickle=False)
                atomic_json(
                    directory / "result.json",
                    {
                        "seed": seed,
                        "schedule": schedule,
                        "trajectory": trajectory,
                        "budget": {
                            "arrivals": len(order),
                            "updates": 576,
                            "optimizer_images": len(order),
                            "reservoir_seen": buffer.seen,
                            "reservoir_size": len(buffer.images),
                        },
                        "arrival_order_sha256": tensor_hash(order),
                        "elapsed_seconds": time.perf_counter() - tick,
                        "encoding_seconds": buffer.encoding_seconds,
                        "admission_seconds": buffer.admission_seconds,
                        "persistent_bytes": buffer.persistent_bytes(),
                        "reference_unchanged": True,
                        "encoder_delta_l2": encoder_delta,
                    },
                )
                status["completed_arms"] += 1
                progress(stage="arm_completed", seed=seed, schedule=schedule, step=576)
        if any(digest(Path(p)) != sha for p, sha in hashes.items()):
            raise ValueError("Locked source/input changed during collection")
        progress(stage="collection_complete", inputs_unchanged=True)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
