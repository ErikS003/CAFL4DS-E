"""Run three resumable, development-only protection pilots on the N73 stream."""

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
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_extended_gate import FrameStream, commit_checkpoint, load_checkpoint, remove_generation
from cafl4ds.single_image_filter_factorial import feature_vectors
from cafl4ds.single_image_gate_protection import (
    ARMS,
    GROUPS,
    confirmed_anchor,
    development_scores,
    fit_corrected,
    protection_passes,
    rollback,
)
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import seeded_step
from scripts.report_single_image_gate_protection import report
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_extended_gate import measure
from scripts.run_single_image_filter_factorial import environment, make_observation
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_gate_protection.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-gate-protection-frozen.md"


def check_storage(settings):
    if shutil.disk_usage(ROOT).free < settings["minimum_free_gib"] * 2**30:
        raise RuntimeError("Two-GiB storage reserve reached; committed state is resumable")


def validate(root):
    lock = read(root / "input-lock.json")
    for name, expected in lock["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen pilot input changed: {name}")
    return lock


def prepare(root):
    if (root / "input-lock.json").exists():
        return validate(root)
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG))
    prior = ROOT / cfg["prior"]
    inherited = read(prior / "input-lock.json")
    assert cfg["history"] == inherited["history"] and cfg["seed"] == inherited["seed"]
    assert cfg["offered"] == inherited["settings"]["offered"]
    assert list(cfg["arms"]) == list(ARMS)
    for name, expected in inherited["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"N73 inherited input changed: {name}")
    source = ROOT / cfg["source"]
    settings = OmegaConf.to_container(OmegaConf.load(ROOT / "cafl4ds/configs/single_image_filter_factorial.yaml"))
    settings["max_validation_admission"] = cfg["max_validation_admission"]
    with np.load(source / "calibration.npz") as arrays:
        model, selection = fit_corrected(
            arrays, read(source / "feature-schema.json")["rich"], settings, cfg["excluded_features"], cfg["seed"]
        )
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / "forecaster.joblib")
    atomic_json(root / "calibration.json", selection)
    files = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "scripts/report_single_image_gate_protection.py",
        ROOT / "cafl4ds/single_image_gate_protection.py",
        root / "forecaster.joblib",
        root / "calibration.json",
        source / "feature-schema.json",
        prior / "input-lock.json",
        prior / "trajectories" / cfg["history"] / f"seed_{cfg['seed']}" / "full/trajectory.json",
    ]
    hashes = {**inherited["hashes"], **{str(path.resolve()): digest(path) for path in files}}
    frozen_settings = {**inherited["settings"], **cfg}
    lock = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        settings=frozen_settings,
        paths=inherited["paths"],
        hashes=hashes,
        threshold=selection["threshold"],
        covariance_cap=float(np.load(source / "references.npz")["selector"][2] * cfg["covariance_warm_multiplier"]),
        history=cfg["history"],
        seed=cfg["seed"],
        arms=list(ARMS),
        inherited_full_reference=str(files[-1]),
        parent=str(source / "trajectories" / cfg["history"] / f"seed_{cfg['seed']}" / "parent.pt"),
        final_queries_control_admissions=False,
        new_accept_all_trajectories=0,
        exploratory=True,
        corrected_calibration_scope="Clock exclusion, retained-coordinate bounding, development refit",
    )
    assert len(lock["paths"]) * 96 == frozen_settings["start"] + cfg["offered"]
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "collection.json", dict(records=[]))
    atomic_json(root / "status.json", dict(stage="prepared", running=False, completed_arms=0, total_arms=3))
    report(root)
    return lock


def initial_context(root, lock, env):
    source = ROOT / lock["settings"]["source"]
    original = source / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    parent = torch.load(original / "parent.pt", map_location="cpu", weights_only=True)
    observations = np.load(original / "parent-observations.npz")["values"]
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    common = directory / "common"
    if not (common / "readout.json").exists():
        model, optim = restore(env[2], env[1].optim, parent)
        value = measure(model, optim, env, common)
        expected = read(original / "common/readout.json")
        for name, actual in value["metrics"].items():
            np.testing.assert_allclose(actual, expected["metrics"][name], rtol=1e-10, atol=1e-10)
        assert fingerprint(capture(model, optim)) == fingerprint(parent)
        del model, optim
    return parent, observations, directory


def empty_trace(length):
    return dict(
        accepted=np.zeros(length, bool),
        j_admitted=np.zeros(length, bool),
        tried=np.zeros(length, bool),
        protection_pass=np.zeros(length, bool),
        predictions=np.zeros(length),
        features=np.zeros((length, 86)),
        clipping=np.zeros(length),
        training=np.zeros((length, 2)),
        trial_training=np.zeros((length, 2)),
        trial_health=np.zeros((length, 7)),
        trial_development=np.zeros((length, 6)),
        development_after=np.zeros((length, 6)),
        anchors_before=np.zeros((length, 3)),
    )


def add_development(record, scores, anchor):
    for i, group in enumerate(GROUPS):
        record["metrics"][f"development_accuracy_{group}"] = float(scores[i] * 100)
        record["metrics"][f"development_soft_error_{group}"] = float(scores[i + 3])
        record["metrics"][f"development_anchor_{group}"] = float(anchor[i] * 100)
    return record


def run_arm(root, lock, env, forecaster, directory, arm, parent, initial_obs, emit, target=None):
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, _, _, _ = env
    length, start = settings["offered"], settings["start"]
    target = length if target is None else target
    branch = directory / arm
    branch.mkdir(parents=True, exist_ok=True)
    if (branch / "trajectory.json").exists():
        result = read(branch / "trajectory.json")
        assert result["offered"] == length
        return result
    trace = empty_trace(length)
    costs = dict(features=0.0, prediction=0.0, trial_training=0.0, observations=0.0, rollback=0.0)
    checkpoints = {str(start): read(directory / "common/readout.json")}
    observations = [(value, None, None) for value in initial_obs]
    state, done, research_seconds, rollbacks_verified = parent, 0, 0.0, 0
    anchor, previous_checkpoint = None, None
    resumed = load_checkpoint(branch)
    if resumed is not None:
        generation, saved = resumed
        state = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
        assert fingerprint(state) == saved["fingerprint"]
        done, costs, checkpoints = saved["done"], saved["costs"], saved["checkpoints"]
        research_seconds = saved["research_seconds"]
        rollbacks_verified = saved["rollbacks_verified"]
        anchor, previous_checkpoint = np.array(saved["anchor"]), np.array(saved["previous_checkpoint"])
        with np.load(generation / "trace.npz") as arrays:
            for key in trace:
                trace[key][:] = arrays[key]
            observations = [(value, None, None) for value in arrays["last_observations"]]
    model, optim = restore(initial, config.optim, state)
    tick = time.perf_counter()
    current, probe, task = make_observation(model, bank, warm, dev, settings, rich=True)
    scores = development_scores(probe, dev, settings["soft_accuracy_temperature"])
    observations[-1] = current
    if anchor is None:
        anchor, previous_checkpoint = scores[:3].copy(), scores[:3].copy()
    if done == 0:
        checkpoints[str(start)] = add_development(checkpoints[str(start)], scores, anchor)
        costs["observations"] += time.perf_counter() - tick
    # Resume reconstruction is research overhead; it is not a second scientific observation/update.
    arrivals = FrameStream(lock["paths"])
    threshold = lock["threshold"]
    for i in range(done, target):
        if i % settings["checkpoint_stride"] == 0:
            check_storage(settings)
        t = start + i
        tick = time.perf_counter()
        vector = feature_vectors(
            model,
            optim,
            arrivals,
            t,
            observations[-4:],
            centroid,
            projection,
            clusters,
            settings,
            view_seed(settings["stream_namespace"], lock["history"], t, lock["seed"]),
            task,
            rich=True,
        )["rich"]
        trace["features"][i] = vector
        trace["clipping"][i] = forecaster.clipped_fraction(vector)
        costs["features"] += time.perf_counter() - tick
        tick = time.perf_counter()
        prediction = float(forecaster.predict(vector)[0])
        j_admitted = prediction < threshold
        trace["predictions"][i], trace["j_admitted"][i] = prediction, j_admitted
        trace["anchors_before"][i] = anchor
        costs["prediction"] += time.perf_counter() - tick
        accepted = False
        if j_admitted:
            protected = arm != "corrected_J"
            before = capture(model, optim) if protected else None
            tick = time.perf_counter()
            training = seeded_step(
                model,
                optim,
                arrivals[t : t + 1].cuda(),
                view_seed(settings["stream_namespace"], lock["history"], t, lock["seed"]),
            )
            trace["tried"][i] = True
            trace["trial_training"][i] = training
            costs["trial_training"] += time.perf_counter() - tick
            tick = time.perf_counter()
            candidate, candidate_probe, candidate_task = make_observation(model, bank, warm, dev, settings, rich=True)
            candidate_scores = development_scores(candidate_probe, dev, settings["soft_accuracy_temperature"])
            trace["trial_health"][i] = candidate[0]
            trace["trial_development"][i] = candidate_scores
            accepted = protection_passes(
                arm,
                candidate[0][2],
                candidate_scores,
                anchor,
                lock["covariance_cap"],
                settings["development_accuracy_tolerance"],
            )
            trace["protection_pass"][i] = accepted
            costs["observations"] += time.perf_counter() - tick
            if accepted:
                current, task, scores = candidate, candidate_task, candidate_scores
                trace["training"][i] = training
            else:
                tick = time.perf_counter()
                rollback(model, optim, before)
                assert fingerprint(capture(model, optim)) == fingerprint(before)
                rollbacks_verified += 1
                costs["rollback"] += time.perf_counter() - tick
            del before
        trace["accepted"][i] = accepted
        trace["development_after"][i] = scores
        observations = [*observations[-3:], current]
        if (i + 1) % settings["checkpoint_stride"] == 0 or i + 1 == target:
            tick = time.perf_counter()
            # Only development scores update anchors; final-query readers are observed afterward.
            anchor = confirmed_anchor(anchor, previous_checkpoint, scores[:3])
            previous_checkpoint = scores[:3].copy()
            checkpoints[str(t + 1)] = add_development(
                measure(model, optim, env, branch / f"checkpoint_{t + 1}"), scores, anchor
            )
            research_seconds += time.perf_counter() - tick
            saved = dict(
                done=i + 1,
                costs=costs,
                checkpoints=checkpoints,
                research_seconds=research_seconds,
                anchor=anchor.tolist(),
                previous_checkpoint=previous_checkpoint.tolist(),
                rollbacks_verified=rollbacks_verified,
                fingerprint=fingerprint(capture(model, optim)),
            )
            commit_checkpoint(
                branch,
                capture(model, optim),
                {**trace, "last_observations": np.array([value[0] for value in observations])},
                saved,
            )
            emit(
                stage="running",
                arm=arm,
                offered=i + 1,
                accepted=int(trace["accepted"].sum()),
                rejected_trials=int(trace["tried"].sum() - trace["accepted"].sum()),
                accuracy=checkpoints[str(t + 1)]["metrics"]["linear_accuracy_balanced"],
                development_accuracy=(100 * scores[:3]).tolist(),
            )
    generation, saved = load_checkpoint(branch)
    record = dict(
        **saved,
        arm=arm,
        offered=target,
        accepted=int(trace["accepted"].sum()),
        j_admitted=int(trace["j_admitted"].sum()),
        rejected_trials=rollbacks_verified,
        parent_fingerprint=fingerprint(parent),
        final_fingerprint=saved["fingerprint"],
        policy_seconds=sum(costs.values()),
        threshold=threshold,
    )
    if target == length:
        shutil.copy2(generation / "model.pt", branch / "final_state.pt")
        shutil.copy2(generation / "trace.npz", branch / "trace.npz")
        atomic_json(branch / "trajectory.json", record)
        verify_arm(branch, lock, forecaster)
        for generation in list((branch / "resume_states").glob("step_*")):
            remove_generation(generation)
    del model, optim
    return record


def verify_arm(branch, lock, forecaster):
    record = read(branch / "trajectory.json")
    with np.load(branch / "trace.npz") as trace:
        n, arm = record["offered"], record["arm"]
        np.testing.assert_allclose(forecaster.predict(trace["features"][:n]), trace["predictions"][:n], rtol=1e-10)
        np.testing.assert_array_equal(trace["j_admitted"][:n], trace["predictions"][:n] < lock["threshold"])
        np.testing.assert_array_equal(trace["tried"][:n], trace["j_admitted"][:n])
        reconstructed = np.zeros(n, bool)
        for i in np.flatnonzero(trace["j_admitted"][:n]):
            reconstructed[i] = protection_passes(
                arm,
                trace["trial_health"][i, 2],
                trace["trial_development"][i],
                trace["anchors_before"][i],
                lock["covariance_cap"],
                lock["settings"]["development_accuracy_tolerance"],
            )
        np.testing.assert_array_equal(trace["accepted"][:n], reconstructed)
        assert reconstructed.sum() == record["accepted"]
        assert int(trace["tried"][:n].sum() - reconstructed.sum()) == record["rejected_trials"]
        assert all(np.isfinite(trace[key]).all() for key in trace.files)
    state = torch.load(branch / "final_state.pt", map_location="cpu", weights_only=True)
    assert fingerprint(state) == record["final_fingerprint"]
    atomic_json(
        branch / "verification.json", dict(passed=True, decisions=n, exact_rollback_checks=record["rejected_trials"])
    )


def smoke(root, lock, env, forecaster, parent, observations):
    folder = root / "smoke"
    if (folder / "verification.json").exists():
        assert read(folder / "verification.json")["passed"]
        return
    folder.mkdir(exist_ok=True)
    smoke_settings = dict(env[0], offered=64, checkpoint_stride=16)
    smoke_env = (smoke_settings, *env[1:])
    smoke_lock = dict(lock, settings=smoke_settings)
    common = folder / "common"
    common.mkdir(exist_ok=True)
    source_common = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "common"
    for name in ("readout.json", "raw.npz"):
        shutil.copy2(source_common / name, common / name)
    for arm in ARMS:
        run_arm(root, smoke_lock, smoke_env, forecaster, folder, arm, parent, observations, lambda **_: None)
    uninterrupted = read(folder / "corrected_development/trajectory.json")
    resumed_dir = folder / "resume"
    (resumed_dir / "common").mkdir(parents=True, exist_ok=True)
    shutil.copy2(common / "readout.json", resumed_dir / "common/readout.json")
    run_arm(
        root,
        smoke_lock,
        smoke_env,
        forecaster,
        resumed_dir,
        "corrected_development",
        parent,
        observations,
        lambda **_: None,
        target=32,
    )
    resumed = run_arm(
        root,
        smoke_lock,
        smoke_env,
        forecaster,
        resumed_dir,
        "corrected_development",
        parent,
        observations,
        lambda **_: None,
    )
    assert resumed["final_fingerprint"] == uninterrupted["final_fingerprint"]
    assert resumed["anchor"] == uninterrupted["anchor"]
    with np.load(folder / "corrected_development/trace.npz") as expected:
        with np.load(resumed_dir / "corrected_development/trace.npz") as actual:
            for name in expected.files:
                np.testing.assert_array_equal(actual[name], expected[name])
    # Exercise a real rejected trial even if the ordinary 64-image smoke makes no rejection.
    model, optim = restore(env[2], env[1].optim, parent)
    before = capture(model, optim)
    arrivals = FrameStream(lock["paths"])
    seeded_step(model, optim, arrivals[env[0]["start"] : env[0]["start"] + 1].cuda(), 94731)
    assert fingerprint(capture(model, optim)) != fingerprint(before)
    rollback(model, optim, before)
    assert fingerprint(capture(model, optim)) == fingerprint(before)
    del model, optim
    atomic_json(
        folder / "verification.json",
        dict(
            passed=True,
            policies=3,
            arrivals_each=64,
            exact_interrupted_resume=True,
            real_trial_rollback=True,
            same_N73_parent=True,
        ),
    )


def run(root, lock, only_smoke=False):
    setup()
    source = ROOT / lock["settings"]["source"]
    env = environment(source)
    env[0].update(lock["settings"])
    forecaster = joblib.load(root / "forecaster.joblib")
    parent, observations, directory = initial_context(root, lock, env)
    smoke(root, lock, env, forecaster, parent, observations)
    if only_smoke:
        return
    began, records = time.perf_counter(), []

    def emit(**event):
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                total_arms=3,
                completed_arms=len(records),
                target_offered_each=lock["settings"]["offered"],
                elapsed_seconds=time.perf_counter() - began,
                free_gib=shutil.disk_usage(ROOT).free / 2**30,
                **event,
            ),
        )
        print(json.dumps(read(root / "status.json")), flush=True)
        report(root)

    for arm in ARMS:
        emit(stage="starting_arm", arm=arm)
        record = run_arm(root, lock, env, forecaster, directory, arm, parent, observations, emit)
        records.append(dict(arm=arm, path=str((directory / arm).relative_to(root))))
        atomic_json(root / "collection.json", dict(records=records))
        emit(stage="arm_complete", arm=arm, offered=record["offered"], accepted=record["accepted"])
    validate(root)
    atomic_json(
        root / "verification.json",
        dict(passed=True, completed_trajectories=3, new_accept_all=0, parent_fingerprint=fingerprint(parent)),
    )
    emit(stage="complete", arm=None)
    atomic_json(root / "status.json", dict(read(root / "status.json"), running=False))
    report(root)


def main():
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
        atomic_json(root / "failure.json", dict(pid=os.getpid(), error=repr(error)))
        atomic_json(root / "status.json", dict(read(root / "status.json"), stage="failed", running=False))
        raise


if __name__ == "__main__":
    main()
