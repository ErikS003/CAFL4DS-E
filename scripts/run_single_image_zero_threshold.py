"""Queue N78 after N77: the same covariance controller with an easier J cutoff."""

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
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from scripts.report_single_image_zero_threshold import report
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_covariance_regularization import run_arm, smoke
from scripts.run_single_image_covariance_relaxation import recover_publication
from scripts.run_single_image_filter_factorial import environment
from scripts.run_single_image_gate_protection import initial_context
from scripts.run_single_image_learning_rate import validate
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_zero_threshold.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-zero-threshold-frozen.md"


def predecessor_ready(
    status: dict[str, Any], trajectory: dict[str, Any] | None, verification: dict[str, Any] | None, target: int
) -> bool:
    """Start only after successful, verified completion; failure never releases GPU work."""
    if status.get("stage") in ("failed", "stopped", "cancelled") or str(status.get("stage", "")).startswith("stopped_"):
        raise RuntimeError("N77 did not complete successfully; queued N78 will not start")
    if status.get("running") or status.get("stage") != "complete":
        return False
    if not trajectory or not verification or not verification.get("passed"):
        return False
    if trajectory.get("offered") != target:
        raise RuntimeError("N77 completed record has an unexpected arrival horizon")
    return True


def prepare(root: Path) -> dict[str, Any]:
    """Freeze one prospective threshold change without loading a learner or GPU."""
    if (root / "input-lock.json").exists():
        return validate(root)
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG))
    prior = ROOT / cfg["predecessor"]
    old = validate(prior)
    if cfg["threshold"] != 0.0 or cfg["threshold"] <= old["threshold"]:
        raise ValueError("N78 requires the user-selected easier numerical cutoff of zero")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prior / "forecaster.joblib", root / "forecaster.joblib")
    calibration = read(prior / "calibration.json")
    calibration.update(
        original_threshold=old["threshold"],
        threshold=0.0,
        threshold_source="Explicit user choice: admit predicted delta J below zero",
    )
    atomic_json(root / "calibration.json", calibration)
    original_common = prior / "trajectories" / old["history"] / f"seed_{old['seed']}" / "common"
    common = root / "trajectories" / old["history"] / f"seed_{old['seed']}" / "common"
    common.mkdir(parents=True, exist_ok=True)
    for name in ("readout.json", "raw.npz"):
        shutil.copy2(original_common / name, common / name)
    files = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "scripts/report_single_image_zero_threshold.py",
        prior / "input-lock.json",
        prior / "calibration.json",
        root / "forecaster.joblib",
        root / "calibration.json",
        common / "readout.json",
        common / "raw.npz",
    ]
    lock = dict(
        old,
        settings={**old["settings"], **cfg},
        threshold=0.0,
        previous_J_threshold=old["threshold"],
        reference_N77_root=str(prior),
        created_utc=datetime.now(timezone.utc).isoformat(),
        hashes={**old["hashes"], **{str(p.resolve()): digest(p) for p in files}},
        policy_change="Only upstream J admission cutoff: inherited negative threshold to zero",
    )
    atomic_json(root / "input-lock.json", lock)
    atomic_json(
        root / "status.json",
        {
            "stage": "queued",
            "running": False,
            "predecessor": str(prior),
            "target_offered": lock["settings"]["offered"],
            "threshold": 0.0,
        },
    )
    atomic_json(
        root / "queue.json",
        {
            "stage": "queued",
            "created_utc": lock["created_utc"],
            "predecessor": str(prior),
            "threshold": 0.0,
            "previous_threshold": old["threshold"],
            "start_condition": "N77 successful verified completion",
            "gpu_work_started": False,
        },
    )
    report(root)
    return lock


def wait_for_predecessor(root: Path, lock: dict[str, Any]) -> bool:
    """Wait in this background worker, with no learner construction or GPU work."""
    prior = Path(lock["reference_N77_root"])
    inherited = read(prior / "input-lock.json")
    branch = prior / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / inherited["settings"]["arm"]
    atomic_json(root / "queue.json", dict(read(root / "queue.json"), worker_pid=os.getpid()))
    while True:
        if (root / "cancel-queue.json").exists():
            atomic_json(root / "status.json", dict(read(root / "status.json"), stage="cancelled", running=False))
            atomic_json(root / "queue.json", dict(read(root / "queue.json"), stage="cancelled"))
            return False
        status = read(prior / "status.json")
        release_path = root / "authorized-release.json"
        if release_path.exists():
            release = read(release_path)
            if status.get("running") or status.get("stage") != "stopped" or release.get("release_for") != "N78":
                raise RuntimeError("Explicit N78 release requires N77's complete stopped checkpoint")
            validate(prior)
            for name, expected in release["hashes"].items():
                if digest(Path(name)) != expected:
                    raise ValueError("Authorized predecessor-stop checkpoint changed")
            atomic_json(root / "predecessor-completion.json", release)
            atomic_json(
                root / "queue.json",
                dict(
                    read(root / "queue.json"),
                    stage="released",
                    released_utc=datetime.now(timezone.utc).isoformat(),
                    release_reason="Explicit user instruction after N77 stop",
                ),
            )
            return True
        trajectory = read(branch / "trajectory.json") if (branch / "trajectory.json").exists() else None
        verification = read(branch / "verification.json") if (branch / "verification.json").exists() else None
        if predecessor_ready(status, trajectory, verification, inherited["settings"]["offered"]):
            validate(prior)
            files = [branch / name for name in ("trajectory.json", "verification.json", "final_state.pt", "trace.npz")]
            atomic_json(
                root / "predecessor-completion.json",
                {
                    "fingerprint": trajectory["final_fingerprint"],
                    "offered": trajectory["offered"],
                    "hashes": {str(p.resolve()): digest(p) for p in files},
                },
            )
            atomic_json(
                root / "queue.json",
                dict(read(root / "queue.json"), stage="released", released_utc=datetime.now(timezone.utc).isoformat()),
            )
            return True
        atomic_json(
            root / "status.json",
            dict(
                read(root / "status.json"),
                stage="queued",
                running=False,
                queue_worker_pid=os.getpid(),
                predecessor_offered=status.get("offered", 0),
                predecessor_stage=status.get("stage"),
                checked_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        time.sleep(lock["settings"]["poll_seconds"])


def run(root: Path, lock: dict[str, Any]) -> None:
    """Run exactly the inherited controller after the predecessor has released it."""
    atomic_json(
        root / "queue.json",
        dict(
            read(root / "queue.json"),
            stage="preflight",
            gpu_work_started=True,
            started_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    setup()
    env = environment(ROOT / lock["settings"]["source"])
    env[0].update(lock["settings"])
    forecaster = joblib.load(root / "forecaster.joblib")
    parent, observations, directory = initial_context(root, lock, env)
    began = time.perf_counter()

    def emit(**event: object) -> None:
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                target_offered=lock["settings"]["offered"],
                threshold=lock["threshold"],
                elapsed_seconds=time.perf_counter() - began,
                free_gib=shutil.disk_usage(ROOT).free / 2**30,
                **event,
            ),
        )
        print(json.dumps(read(root / "status.json")), flush=True)
        if event.get("checkpoint") or event.get("stage") in ("preflight", "starting", "complete", "stopped"):
            report(root)

    emit(stage="preflight")
    smoke(root, lock, env, forecaster, parent, observations)
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
    done = result["offered"] == lock["settings"]["offered"]
    emit(stage="complete" if done else "stopped", offered=result["offered"], accepted=result["accepted"])
    atomic_json(root / "status.json", dict(read(root / "status.json"), running=False))
    atomic_json(root / "queue.json", dict(read(root / "queue.json"), stage="complete" if done else "stopped"))
    report(root)


def main() -> None:
    """Prepare or launch the exclusive, conditional background job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "queue", "report"), default="queue")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "report":
        report(root)
        return
    lock = prepare(root)
    if args.phase == "prepare":
        return
    lease = root / "queue-worker.lease"
    descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        if wait_for_predecessor(root, lock):
            with threadpool_limits(limits=1):
                run(root, lock)
    except Exception as error:
        atomic_json(root / "failure.json", {"pid": os.getpid(), "error": repr(error)})
        atomic_json(root / "status.json", dict(read(root / "status.json"), running=False, stage="failed"))
        atomic_json(root / "queue.json", dict(read(root / "queue.json"), stage="failed", error=repr(error)))
        raise
    finally:
        lease.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
