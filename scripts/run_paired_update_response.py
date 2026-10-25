"""Run fixed-horizon full/half SSL branches from audited complete parent states."""

from __future__ import annotations

import argparse
import hashlib
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

from cafl4ds.artifacts import atomic_json
from cafl4ds.paired_update_response import identical, restore, run_window
from cafl4ds.recurrence import arrival_orders, evaluation_pool, restrict_scenes, visit_quotas
from cafl4ds.ssl.base import SSLMethod
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_recurrence_retention import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/adaptation-bdd/recurrence-retention/20261010-ssl-control-restart"
PARENT = ROOT / "outputs/update-filtering/20261010-half-rate"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-paired-update-response-frozen.md"
OUTPUT = ROOT / "outputs/paired-update-response/20261010-states"
OFFSETS = (36, 108, 180, 252, 324, 396, 468, 540)


def fingerprint(value: Any) -> str:  # noqa: ANN401 - complete nested saved state
    """Hash tensors and scalar optimizer fields without pickle serialization variability."""

    def encode(item: Any) -> Any:  # noqa: ANN401 - recursive optimizer serialization
        if isinstance(item, torch.Tensor):
            value = hashlib.sha256(str((tuple(item.shape), item.dtype)).encode())
            value.update(item.detach().cpu().contiguous().numpy().tobytes())
            return {"tensor": value.hexdigest()}
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(v) for v in item]
        return item

    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


def state_record(model: SSLMethod, optimizer: torch.optim.Optimizer, arrival: int, updates: int) -> dict[str, Any]:
    """Capture only learning state, separate from parent mask provenance."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "model_training": model.training,
        "arrival_step": arrival,
        "optimizer_steps": updates,
    }


def main() -> None:  # noqa: C901 - explicit immutable parent/data/branch guards
    """Require exact restored continuation before collecting paired terminal outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status: dict[str, Any] = {"pid": os.getpid(), "completed_branches": 0, "total_branches": 80}

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
        parent_inputs = json.loads((PARENT / "input_lock.json").read_text())
        parent_audit = json.loads((PARENT / "independent_verification.json").read_text())
        if not parent_audit["passed"] or parent_audit["new_arms_checked"] != 20:
            raise ValueError("Parent collection audit incomplete")
        for name, sha in parent_inputs["hashes"].items():
            if digest(Path(name)) != sha:
                raise ValueError(f"Parent input changed: {name}")
        for name, sha in parent_audit["artifact_hashes"].items():
            if digest(PARENT / name) != sha:
                raise ValueError(f"Parent artifact changed: {name}")
        config = OmegaConf.create(old_inputs["config"])
        settings = old_inputs["config"]["recurrence"]
        seeds = old_inputs["config"]["seeds"]
        torch.set_num_threads(settings["num_threads"])
        torch.use_deterministic_algorithms(True)
        additional = [
            Path(__file__),
            ROOT / "cafl4ds/paired_update_response.py",
            ROOT / "scripts/report_paired_update_response.py",
            ROOT / "scripts/run_paired_update_response_background.py",
            PROTOCOL,
        ]
        hashes = {**old_inputs["hashes"], **{str(p): digest(p) for p in additional}}
        for name in ("input_lock.json", "data_lock.json", "independent_verification.json", "evaluation_arrays.npz"):
            hashes[str(PRIOR / name)] = digest(PRIOR / name)
        hashes.update({str(PRIOR / name): sha for name, sha in verification["artifact_hashes"].items()})
        hashes.update(parent_inputs["hashes"])
        hashes.update({str(PARENT / name): sha for name, sha in parent_audit["artifact_hashes"].items()})
        hashes[str(PARENT / "independent_verification.json")] = digest(PARENT / "independent_verification.json")
        atomic_json(
            out / "input_lock.json",
            {
                "hashes": hashes,
                "config": old_inputs["config"],
                "versions": versions,
                "prior": str(PRIOR),
                "parent": str(PARENT),
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
        progress(stage="restored_continuation_preflight")
        first_dir = PARENT / "recurrence_fixed_half" / f"seed_{seeds[0]}"
        parent_state = torch.load(
            first_dir / "decision_states/arrival_000036.pt", map_location="cpu", weights_only=False
        )
        next_state = torch.load(first_dir / "decision_states/arrival_000108.pt", map_location="cpu", weights_only=False)
        compatibility_dir = out / "compatibility"
        compatibility_dir.mkdir()
        model, optimizer = restore(initial, config, parent_state)
        continued = run_window(model, optimizer, train.images, orders[seeds[0]]["recurrence"], 36, 18, "half", 72)
        actual = state_record(model, optimizer, 108, 54)
        wanted = {k: next_state[k] for k in actual}
        if not identical(actual, wanted):
            raise ValueError("Full restored continuation model/optimizer/RNG/mode differs")
        expected_losses = json.loads((first_dir / "losses.json").read_text())["losses"]
        expected_losses = [p for p in expected_losses if 36 < p["step"] <= 108]
        if continued["losses"] != expected_losses:
            raise ValueError("Restored continuation losses/gradients differ")
        torch.save(actual, compatibility_dir / "final_state.pt")
        record, _ = evaluate(model, evaluation, reference, 108, True, settings["fine_tune_steps"], compatibility_dir)
        original = next(
            p for p in json.loads((first_dir / "trajectory.json").read_text())["trajectory"] if p["step"] == 108
        )
        if record["health"] != original["health"] or any(
            record["scores"][k] != original["scores"][k] for k in ("knn", "linear")
        ):
            raise ValueError("Restored continuation readout differs")
        with (
            np.load(first_dir / "predictions_000108.npz", allow_pickle=False) as a,
            np.load(compatibility_dir / "predictions_000108.npz", allow_pickle=False) as b,
        ):
            if any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("Restored continuation prediction panels differ")
        atomic_json(compatibility_dir / "result.json", {"window": continued, "terminal": record})
        torch.save(wanted, compatibility_dir / "expected_state.pt")
        model, optimizer = restore(initial, config, parent_state)
        full_preflight = run_window(model, optimizer, train.images, orders[seeds[0]]["recurrence"], 36, 18, "full", 4)
        atomic_json(
            out / "preflight.json",
            {
                "passed": True,
                "data_and_frozen_exact": True,
                "restored_continuation_exact": True,
                "continuation_updates": 36,
                "discarded_full_updates": full_preflight["updates"],
                "baseline_evaluation_seconds": baseline_seconds,
            },
        )
        for seed_position, seed in enumerate(seeds):
            order = orders[seed]["recurrence"]
            parent_dir = PARENT / "recurrence_fixed_half" / f"seed_{seed}"
            for offset_position, offset in enumerate(OFFSETS):
                parent_file = parent_dir / "decision_states" / f"arrival_{offset:06d}.pt"
                state = torch.load(parent_file, map_location="cpu", weights_only=False)
                if (
                    state["arrival_step"] != offset
                    or state["optimizer_steps"] != offset // 2
                    or (state["arrival_order_sha256"] != tensor_hash(order))
                ):
                    raise ValueError("Parent clock/order identity differs")
                parent_before = fingerprint(state)
                policies = ("full", "half") if (seed_position + offset_position) % 2 == 0 else ("half", "full")
                for policy in policies:
                    directory = out / f"seed_{seed}" / f"arrival_{offset:06d}" / policy
                    directory.mkdir(parents=True, exist_ok=False)
                    progress(stage="branch_training", seed=seed, parent_arrival=offset, policy=policy)
                    tick = time.perf_counter()
                    model, optimizer = restore(initial, config, state)
                    restoration_seconds = time.perf_counter() - tick
                    before = state_record(model, optimizer, offset, offset // 2)
                    expected_before = {k: state[k] for k in before}
                    if fingerprint(before) != fingerprint(expected_before):
                        raise ValueError("Restored parent fingerprint differs")
                    window = run_window(model, optimizer, train.images, order, offset, offset // 2, policy)
                    final_state = state_record(model, optimizer, offset + 36, offset // 2 + window["updates"])
                    torch.save(final_state, directory / "final_state.pt")
                    rng = torch.get_rng_state().clone()
                    weights = fingerprint(model.state_dict())
                    progress(stage="branch_evaluating", seed=seed, parent_arrival=offset, policy=policy)
                    terminal, _ = evaluate(
                        model, evaluation, reference, offset + 36, True, settings["fine_tune_steps"], directory
                    )
                    if not torch.equal(rng, torch.get_rng_state()) or weights != fingerprint(model.state_dict()):
                        raise ValueError("Terminal evaluation changed learning state")
                    if parent_before != fingerprint(state):
                        raise ValueError("Branch mutated shared parent state")
                    if initial_hashes != {k: tensor_hash(v) for k, v in initial.state_dict().items()}:
                        raise ValueError("Shared initialization changed")
                    result = {
                        "seed": seed,
                        "offset": offset,
                        "policy": policy,
                        "parent_sha256": digest(parent_file),
                        "restored_fingerprint": fingerprint(before),
                        "arrival_order_sha256": tensor_hash(order),
                        "window": window,
                        "terminal": terminal,
                        "final_state_fingerprint": fingerprint(final_state),
                        "restoration_seconds": restoration_seconds,
                        "elapsed_seconds": time.perf_counter() - tick,
                        "parent_unchanged": True,
                    }
                    atomic_json(directory / "result.json", result)
                    status["completed_branches"] += 1
                    progress(stage="branch_completed", seed=seed, parent_arrival=offset, policy=policy)
        if any(digest(Path(name)) != sha for name, sha in hashes.items()):
            raise ValueError("Locked source/input changed during collection")
        progress(stage="collection_complete", inputs_unchanged=True)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
