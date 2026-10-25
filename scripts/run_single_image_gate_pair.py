"""Run exactly one original-gate/full pair beyond fifty thousand arrivals."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import restore
from cafl4ds.single_image_extended_gate import FrameStream
from cafl4ds.single_image_gate_pair import select_clips
from scripts.prepare_single_image_extended_gate import scientific_sources
from scripts.report_single_image_gate_pair import report
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_extended_gate import assert_inputs, measure, run_arm, verify_arm
from scripts.run_single_image_filter_factorial import environment
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_gate_pair.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-gate-pair-frozen.md"
ARMS = ("full", "gate_original")


def validate(root):
    """Reject any changed source, membership, parent, panel or cached pixels."""
    lock = read(root / "input-lock.json")
    for name, expected in lock["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen pair input changed: {name}")
    return lock


def prepare(root):
    """Freeze a longer cached stream without copying or downloading pixels."""
    if (root / "input-lock.json").exists():
        return validate(root)
    config = OmegaConf.to_container(OmegaConf.load(CONFIG))
    prior = ROOT / config["prior"]
    inherited = assert_inputs(prior)
    source = ROOT / inherited["settings"]["source"]
    meta = inherited["histories"][config["history"]]
    settings = dict(inherited["settings"])
    settings.update(
        offered=config["offered"],
        checkpoint_stride=config["checkpoint_stride"],
        minimum_free_gib=config["minimum_free_gib"],
        trajectory_seeds=[config["seed"]],
    )
    clips = select_clips(
        read(prior / "intake/intake.json")["clips"],
        config["history"],
        meta["pattern"],
        settings["start"],
        settings["offered"],
        config["frames_per_clip"],
    )
    prefix_count = settings["start"] // config["frames_per_clip"]
    prefix_ids, prefix_paths = meta["clip_ids"][:prefix_count], meta["paths"][:prefix_count]
    assert not set(prefix_ids) & {row["clip_id"] for row in clips}
    paths = prefix_paths + [row["frames_path"] for row in clips]
    assert len(paths) * config["frames_per_clip"] == settings["start"] + settings["offered"]
    assert [r["clip_id"] for r in clips[:200]] == meta["clip_ids"][prefix_count:]
    files = [Path(name) for name in inherited["hashes"] if "intake/frames/" not in name.replace("\\", "/")]
    files += [Path(name) for name in prefix_paths]
    files += scientific_sources() + [
        CONFIG,
        PROTOCOL,
        Path(__file__).resolve(),
        ROOT / "cafl4ds/single_image_gate_pair.py",
        ROOT / "scripts/report_single_image_gate_pair.py",
        prior / "input-lock.json",
        prior / "execution-amendment.json",
        prior / "smoke/smoke-verification.json",
        prior / "smoke/resume-verification.json",
    ]
    hashes = {str(path.resolve()): digest(path) for path in files}
    for row in clips:
        path = Path(row["frames_path"])
        assert digest(path) == row["frames_sha256"]
        pixels = np.load(path, mmap_mode="r", allow_pickle=False)
        assert pixels.shape == (96, 64, 64, 3) and pixels.dtype == np.uint8
        hashes[str(path.resolve())] = row["frames_sha256"]
    assert read(prior / "smoke/smoke-verification.json")["passed"]
    assert read(prior / "smoke/resume-verification.json")["passed"]
    threshold = read(source / "model-selection.json")["J_rich"]["threshold"]
    lock = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        settings=settings,
        history=config["history"],
        seed=config["seed"],
        pattern=meta["pattern"],
        specs=dict(full=dict(kind="full"), gate_original=dict(kind="gate", threshold=threshold)),
        paths=paths,
        continuation_clips=clips,
        continuation_unique_clips=len(clips),
        reused_cached_frames=True,
        exploratory=True,
        pair_chosen_by="First frozen-order history and first seed, not highest observed accuracy",
        hashes=hashes,
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "collection.json", dict(records=[]))
    atomic_json(root / "status.json", dict(stage="prepared", running=False, total_arms=2, completed_arms=0))
    report(root)
    return lock


def common_readout(directory, original, env, parent):
    if not (directory / "common/readout.json").exists():
        model, optim = restore(env[2], env[1].optim, parent)
        observed = measure(model, optim, env, directory / "common")
        expected = read(original / "common/readout.json")
        for metric, value in observed["metrics"].items():
            np.testing.assert_allclose(value, expected["metrics"][metric], rtol=1e-10, atol=1e-10)
        del model, optim


def smoke(root, lock, env, models, original, parent, observations):
    """Require exact source-prefix reproduction for both retained policies."""
    work = root / "smoke"
    if (work / "verification.json").exists():
        assert read(work / "verification.json")["passed"]
        return
    work.mkdir(exist_ok=True)
    common_readout(work, original, env, parent)
    source = ROOT / lock["settings"]["source"]
    original_stream = read(source / "intake/intake.json")["histories"][lock["history"]]
    arrivals = FrameStream(original_stream["paths"])
    for arm in ARMS:
        result = run_arm(
            root,
            work,
            arm,
            lock["specs"][arm],
            env,
            models,
            arrivals,
            lock["history"],
            lock["seed"],
            parent,
            observations,
            None,
            env[0]["start"] + 96,
            lambda **event: print(json.dumps(dict(smoke=True, **event)), flush=True),
        )
        verify_arm(work / arm, lock["specs"][arm], None, env, models["J_rich"][lock["seed"]])
        expected = read(original / ("J_rich" if arm == "gate_original" else "full") / "trajectory.json")
        assert result["final_fingerprint"] == expected["checkpoints"][str(env[0]["start"] + 96)]["fingerprint"]
        with np.load(work / arm / "trace.npz") as trace:
            for i, row in enumerate(expected["records"][:96]):
                assert bool(trace["accepted"][i]) == row["accepted"]
                if arm == "gate_original":
                    np.testing.assert_allclose(trace["features"][i], row["features"], rtol=1e-9, atol=1e-9)
                if row["accepted"]:
                    np.testing.assert_allclose(trace["training"][i], row["training"], rtol=1e-7, atol=1e-8)
    atomic_json(work / "verification.json", dict(passed=True, arms=list(ARMS), exact_source_prefix=96))


def run(root, lock):
    settings = lock["settings"]
    source = ROOT / settings["source"]
    env = environment(source)
    env[0].update(settings)
    models = joblib.load(source / "forecasters.joblib")
    original = source / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    parent = torch.load(original / "parent.pt", map_location="cpu", weights_only=True)
    with np.load(original / "parent-observations.npz") as data:
        observations = data["values"]
    smoke(root, lock, env, models, original, parent, observations)
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    common_readout(directory, original, env, parent)
    arrivals = FrameStream(lock["paths"])
    began, roster = time.perf_counter(), []

    def emit(**event):
        status = dict(
            pid=os.getpid(),
            running=True,
            total_arms=2,
            completed_arms=len(roster),
            target_offered=settings["offered"],
            elapsed_seconds=time.perf_counter() - began,
            free_gib=shutil.disk_usage(ROOT).free / 2**30,
            **event,
        )
        atomic_json(root / "status.json", status)
        print(json.dumps(status), flush=True)
        report(root)

    for arm in ARMS:
        emit(stage="starting_arm", history=lock["history"], seed=lock["seed"], arm=arm)
        record = run_arm(
            root,
            directory,
            arm,
            lock["specs"][arm],
            env,
            models,
            arrivals,
            lock["history"],
            lock["seed"],
            parent,
            observations,
            None,
            settings["start"] + settings["offered"],
            emit,
        )
        verify_arm(directory / arm, lock["specs"][arm], None, env, models["J_rich"][lock["seed"]])
        assert record["offered"] == settings["offered"]
        roster.append(
            dict(history=lock["history"], seed=lock["seed"], arm=arm, path=str((directory / arm).relative_to(root)))
        )
        atomic_json(root / "collection.json", dict(records=roster))
        emit(stage="arm_complete", history=lock["history"], seed=lock["seed"], arm=arm)
    validate(root)
    atomic_json(root / "verification.json", dict(passed=True, completed_arms=2, offered_each=settings["offered"]))
    emit(stage="complete")
    status = read(root / "status.json")
    atomic_json(root / "status.json", dict(status, running=False))
    report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "run", "pipeline"), default="pipeline")
    args = parser.parse_args()
    root = args.output.resolve()
    lock = prepare(root) if args.phase != "run" else validate(root)
    if args.phase == "prepare":
        return
    setup()
    try:
        with threadpool_limits(limits=1):
            run(root, lock)
    except Exception as error:
        atomic_json(root / "failure.json", dict(pid=os.getpid(), error=repr(error)))
        prior = read(root / "status.json")
        atomic_json(root / "status.json", dict(prior, stage="failed", running=False, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
