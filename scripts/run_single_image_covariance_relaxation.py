"""N76: one N75-matched trajectory with a 1.30-warm covariance ceiling."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint
from scripts.report_single_image_covariance_relaxation import report
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_filter_factorial import environment
from scripts.run_single_image_gate_protection import initial_context
from scripts.run_single_image_learning_rate import run_arm, smoke, validate
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_covariance_relaxation.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-covariance-relaxation-frozen.md"


def prepare(root: Path) -> dict[str, Any]:
    """Freeze a new pilot while preserving every inherited scientific input."""
    if (root / "input-lock.json").exists():
        return validate(root)
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG))
    prior = ROOT / cfg["prior_learning_rate"]
    old = validate(prior)
    if read(prior / "status.json")["running"] is not False:
        raise ValueError("Stop N75 before preparing this pilot")
    if cfg["covariance_warm_multiplier"] != 1.30:
        raise ValueError("This pilot requires the agreed 1.30-warm ceiling")
    oldbranch = prior / "trajectories" / old["history"] / f"seed_{old['seed']}" / old["settings"]["arm"]
    generation, checkpoint = load_checkpoint(oldbranch)
    with np.load(ROOT / old["settings"]["source"] / "references.npz") as reference:
        warm_covariance = float(reference["selector"][2])
    np.testing.assert_allclose(old["covariance_cap"], 1.25 * warm_covariance, rtol=1e-12)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("forecaster.joblib", "calibration.json"):
        shutil.copy2(prior / name, root / name)
    files = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "scripts/report_single_image_covariance_relaxation.py",
        ROOT / "scripts/report_single_image_reconstruction.py",
        prior / "input-lock.json",
        prior / "stop.json",
        generation / "commit.json",
        generation / "trace.npz",
        generation / "model.pt",
        root / "forecaster.joblib",
        root / "calibration.json",
    ]
    lock = dict(
        old,
        settings={**old["settings"], **cfg},
        covariance_cap=warm_covariance * cfg["covariance_warm_multiplier"],
        warm_covariance=warm_covariance,
        previous_covariance_cap=old["covariance_cap"],
        prior_learning_rate_generation=str(generation),
        prior_learning_rate_offered=checkpoint["done"],
        hashes={**old["hashes"], **{str(p.resolve()): digest(p) for p in files}},
        created_utc=datetime.now(timezone.utc).isoformat(),
        policy_change="Only covariance ceiling: 1.25 to 1.30 times warm; original parent and LR trials",
    )
    if lock["parent"] != old["parent"] or lock["paths"] != old["paths"]:
        raise ValueError("Parent or stream changed")
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "status.json", {"stage": "prepared", "running": False, "total_trajectories": 1})
    report(root)
    return lock


def recover_publication(root: Path, lock: dict[str, Any], error: PermissionError) -> None:
    """Retry only a fully written pending-generation rename after Windows denies it."""
    if getattr(error, "winerror", None) != 5 or not error.filename:
        raise error
    branch = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / lock["settings"]["arm"]
    pending = Path(error.filename).resolve()
    if pending.parent != (branch / "resume_states").resolve() or not pending.name.endswith(".pending"):
        raise error
    record = read(pending / "commit.json")
    if not all(digest(pending / name) == expected for name, expected in record["file_hashes"].items()):
        raise ValueError("Pending checkpoint file hash mismatch")
    state = torch.load(pending / "model.pt", map_location="cpu", weights_only=True)
    if fingerprint(state) != record["fingerprint"]:
        raise ValueError("Pending checkpoint state fingerprint mismatch")
    with np.load(pending / "trace.npz") as trace:
        if not all(np.isfinite(trace[k]).all() for k in trace.files):
            raise ValueError("Non-finite pending checkpoint trace")
        if trace["accepted"][record["done"] :].any():
            raise ValueError("Pending checkpoint has decisions beyond its committed prefix")
    destination = pending.with_name(pending.name.removesuffix(".pending"))
    for attempt in range(10):
        try:
            pending.rename(destination)
            atomic_json(
                root / f"publication-recovery-{record['done']}.json",
                {
                    "offered": record["done"],
                    "verified_hashes": True,
                    "verified_fingerprint": True,
                    "rename_attempts": attempt + 1,
                    "error": repr(error),
                },
            )
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def run(root: Path, lock: dict[str, Any], only_smoke: bool = False) -> None:
    """Run the original LR policy with the new locked ceiling and synchronous plots."""
    setup()
    env = environment(ROOT / lock["settings"]["source"])
    env[0].update(lock["settings"])
    forecaster = joblib.load(root / "forecaster.joblib")
    parent, observations, directory = initial_context(root, lock, env)
    smoke(root, lock, env, forecaster, parent, observations)
    if only_smoke:
        return
    began = time.perf_counter()

    def emit(**event: object) -> None:
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                total_trajectories=1,
                target_offered=lock["settings"]["offered"],
                covariance_warm_multiplier=1.30,
                elapsed_seconds=time.perf_counter() - began,
                free_gib=shutil.disk_usage(ROOT).free / 2**30,
                **event,
            ),
        )
        print(json.dumps(read(root / "status.json")), flush=True)
        report(root)

    emit(stage="starting")
    for attempt in range(4):
        try:
            result = run_arm(root, lock, env, forecaster, directory, parent, observations, emit)
            break
        except PermissionError as error:
            if attempt == 3:
                raise
            recover_publication(root, lock, error)
    validate(root)
    atomic_json(
        root / "verification.json",
        {
            "passed": True,
            "evaluated_trajectories": 1,
            "new_accept_all": 0,
            "accepted": result["accepted"],
            "rescued_updates": result["rescued_updates"],
        },
    )
    emit(stage="complete", offered=result["offered"], accepted=result["accepted"])
    atomic_json(root / "status.json", dict(read(root / "status.json"), running=False))
    report(root)


def main() -> None:
    """Prepare, preflight, resume or report the single authorized trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "smoke", "run", "report"), default="run")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "report":
        report(root)
        return
    lock = prepare(root)
    if args.phase == "prepare":
        return
    try:
        with threadpool_limits(limits=1):
            run(root, lock, only_smoke=args.phase == "smoke")
    except Exception as error:
        atomic_json(root / "failure.json", {"pid": os.getpid(), "error": repr(error)})
        atomic_json(root / "status.json", dict(read(root / "status.json"), stage="failed", running=False))
        raise


if __name__ == "__main__":
    main()
