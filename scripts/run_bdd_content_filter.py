"""Collect matched-work temporal BDD frame filtering with no downstream labels at runtime."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import evaluation_pool, restrict_scenes
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.temporal_content_filter import RANDOM_SEED_OFFSET, optimize_window
from scripts.confirm_last_block import tensor_hash
from scripts.decode_bdd_temporal import sha256
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_recurrence_retention import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/temporal-bdd/20261011-feasibility-v2"
INTAKE = ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-bdd-content-filter-frozen.md"
POLICIES = ("full", "fixed_half", "random_half", "coverage_half")
SEEDS = (5003, 5009, 5011, 5021, 5023)


def learning_state(
    model: SSLMethod, optimizer: torch.optim.Optimizer, updates: int = 0, arrivals: int = 0,
) -> dict[str, Any]:
    """Deep-copy complete state so recording a fingerprint cannot alias future weights."""
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "torch_rng": torch.get_rng_state(), "training": model.training,
                          "arrivals": arrivals, "optimizer_updates": updates})


def verify_control(directory: Path, policy: str) -> None:
    """Require exact existing full/fixed seed-5003 learning and readout reproduction."""
    previous = PRIOR / "chronological" / ("full" if policy == "full" else "half") / "seed_5003"
    old = torch.load(previous / "final_state.pt", map_location="cpu", weights_only=False)
    new = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
    if fingerprint(new) != fingerprint(old):
        raise ValueError(f"{policy} final model/optimizer/RNG differs from prior control")
    original_losses = [r["loss_gradient"] for r in json.loads((previous / "arrivals.json").read_text())["records"]
                       if r["accepted"]]
    new_losses = [p for w in json.loads((directory / "windows.json").read_text())["windows"]
                  for p in w["loss_gradients"]]
    if new_losses != original_losses:
        raise ValueError(f"{policy} loss/gradient sequence differs from prior control")
    with np.load(previous / "predictions_000192.npz") as a, np.load(directory / "predictions_000192.npz") as b:
        if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError(f"{policy} terminal predictions differ from prior control")


def main() -> None:  # noqa: C901 - explicit lock, compatibility and collection gates
    """Collect a new fixed experiment, preserving failures and every completed arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def progress(**fields: Any) -> None:  # noqa: ANN401 - JSON progress fields
        atomic_json(out / "status.json", {"started_unix": started, "updated_unix": time.time(), **fields})

    try:
        progress(stage="validating_prior")
        audit_path = PRIOR / "independent_verification.json"
        audit = json.loads(audit_path.read_text())
        if not audit["passed"] or audit["arms_checked"] != 20:
            raise ValueError("temporal feasibility is not audited")
        old = json.loads((PRIOR / "input_lock.json").read_text())
        hashes = {**old["hashes"], **{
            str(PRIOR / name): expected for name, expected in audit["artifact_hashes"].items()
        }}
        for name, expected in hashes.items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"prior source/input/artifact changed: {name}")
        additional = [Path(__file__), PROTOCOL, audit_path, ROOT / "cafl4ds/temporal_content_filter.py",
                      ROOT / "scripts/report_bdd_content_filter.py", ROOT / "scripts/run_bdd_content_background.py"]
        hashes.update({str(p): sha256(p) for p in additional})
        intake = json.loads(INTAKE.read_text())
        if not intake["complete"] or len(intake["clips"]) != 48:
            raise ValueError("incomplete temporal intake")
        arrays = [np.load(clip["frames_path"], allow_pickle=False) for clip in intake["clips"]]
        config = OmegaConf.create(old["config"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        images = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
        del arrays
        if images.shape != (6144, 3, 64, 64):
            raise ValueError("invalid arrival shape")
        prior_data = json.loads((PRIOR / "data_lock.json").read_text())
        progress(stage="loading_fixed_validation")
        settings = old["config"]["recurrence"]
        validation, _ = restrict_scenes(instantiate(config.evaluation_data).load(), settings["scene_classes"])
        previous_data_path = ROOT / "outputs/paired-update-response/20261010-states/data_lock.json"
        previous_data = json.loads(previous_data_path.read_text())
        validation_hash = tensor_hash(validation.images)
        if validation_hash != previous_data["val_images_sha256"]:
            raise ValueError("complete validation pixel tensor differs from earlier controls")
        evaluation = evaluation_pool(validation, settings["domains"], settings)
        if json.loads(json.dumps(evaluation.manifest)) != prior_data["evaluation"]:
            raise ValueError("fixed readout allocation differs")
        del validation
        torch.save(evaluation.images, out / "evaluation_images.pt")
        np.savez_compressed(out / "evaluation_arrays.npz", labels=evaluation.labels.numpy(),
                            domains=evaluation.domains.numpy(), query=evaluation.query.numpy())
        hashes[str(out / "evaluation_images.pt")] = sha256(out / "evaluation_images.pt")
        hashes[str(out / "evaluation_arrays.npz")] = sha256(out / "evaluation_arrays.npz")
        atomic_json(out / "input_lock.json", {"hashes": hashes, "config": old["config"],
                    "seeds": SEEDS, "policies": POLICIES, "protocol_sha256": sha256(PROTOCOL), "prior": str(PRIOR)})
        atomic_json(out / "data_lock.json", {**prior_data, "validation_tensor_sha256": validation_hash,
                                             "window_frames": 64, "window_count": 96,
                                             "intake_path": str(INTAKE), "intake_sha256": sha256(INTAKE)})
        for name in hashes:
            source = Path(name)
            if source.suffix in {".py", ".md", ".yaml"} and source.is_relative_to(ROOT):
                target = out / "source_snapshot" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        torch.manual_seed(11)
        initial = _method(config)
        frozen_dir = out / "frozen"
        frozen_dir.mkdir()
        progress(stage="reproducing_frozen")
        frozen, reference = evaluate(initial, evaluation, None, 0, True, 100, frozen_dir)
        if frozen != json.loads((PRIOR / "frozen/result.json").read_text()):
            raise ValueError("frozen development readout differs")
        atomic_json(frozen_dir / "result.json", frozen)
        smoke = []
        progress(stage="discarded_preflight")
        with torch.random.fork_rng():
            for policy in POLICIES:
                torch.manual_seed(SEEDS[0])
                model = copy.deepcopy(initial)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                start_state = learning_state(model, optimizer)
                marker = fingerprint(start_state)
                record = optimize_window(model, optimizer, policy, images, 0,
                                         torch.Generator().manual_seed(SEEDS[0] + RANDOM_SEED_OFFSET))
                if fingerprint(start_state) != marker:
                    raise ValueError("preflight initial snapshot mutated")
                smoke.append({"policy": policy, **record})
        atomic_json(out / "preflight.json", {"passed": True, "discarded_updates": 5, "windows": smoke})
        completed = 0
        for seed_index, seed in enumerate(SEEDS):
            rotation = seed_index % len(POLICIES)
            for policy in POLICIES[rotation:] + POLICIES[:rotation]:
                directory = out / policy / f"seed_{seed}"
                directory.mkdir(parents=True)
                torch.manual_seed(seed)
                model = copy.deepcopy(initial)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                random = torch.Generator().manual_seed(seed + RANDOM_SEED_OFFSET)
                start_state = learning_state(model, optimizer)
                marker = fingerprint(start_state)
                torch.save(start_state, directory / "initial_state.pt")
                marker_file = sha256(directory / "initial_state.pt")
                windows = []
                for window in range(96):
                    record = optimize_window(model, optimizer, policy, images, window * 64, random)
                    windows.append({"window": window + 1, "clip_id": intake["clips"][window // 2]["clip_id"], **record})
                    if (window + 1) % 8 == 0:
                        progress(stage="collecting", policy=policy, seed=seed, window=window + 1,
                                 total_windows=96, completed_arms=completed, total_arms=20)
                updates = sum(w["optimizer_updates"] for w in windows)
                if fingerprint(start_state) != marker or sha256(directory / "initial_state.pt") != marker_file:
                    raise ValueError("initial snapshot changed")
                torch.save(learning_state(model, optimizer, updates, 6144), directory / "final_state.pt")
                atomic_json(directory / "windows.json", {"windows": windows})
                before = fingerprint(learning_state(model, optimizer, updates, 6144))
                progress(stage="terminal_readout", policy=policy, seed=seed, completed_arms=completed, total_arms=20)
                outcome, _ = evaluate(model, evaluation, reference, 192, True, 100, directory)
                if before != fingerprint(learning_state(model, optimizer, updates, 6144)):
                    raise ValueError("readout changed learning state")
                atomic_json(directory / "result.json", {"outcome": outcome, "seed": seed, "policy": policy,
                            "optimizer_updates": updates, "accepted_frames": sum(w["accepted_frames"] for w in windows),
                            "online_seconds": sum(w["online_seconds"] for w in windows),
                            "selection_seconds": sum(w["selection_seconds"] for w in windows),
                            "ssl_seconds": sum(w["ssl_seconds"] for w in windows),
                            "initial_fingerprint": marker, "initial_file_sha256": marker_file,
                            "readout_preserved_learning_state": True})
                if seed == SEEDS[0] and policy in {"full", "fixed_half"}:
                    verify_control(directory, policy)
                    atomic_json(directory / "control_reproduction.json", {"passed": True,
                                "model_optimizer_rng_loss_and_predictions_exact": True})
                completed += 1
        progress(stage="collection_complete", completed_arms=20, total_arms=20)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
