"""One covariance-protected trajectory with genuine AdamW LR backtracking."""

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
from cafl4ds.single_image_gate_protection import development_scores, rollback
from cafl4ds.single_image_learning_rate import choose_first_passing, set_rates, validate_factors
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import seeded_step
from scripts.report_single_image_learning_rate import report
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_extended_gate import measure
from scripts.run_single_image_filter_factorial import environment, make_observation
from scripts.run_single_image_gate_protection import add_development, check_storage, initial_context
from scripts.run_single_image_nonlinear import health_read
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_learning_rate.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-learning-rate-frozen.md"


def validate(root):
    lock = read(root / "input-lock.json")
    for name, expected in lock["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen LR input changed: {name}")
    return lock


def prepare(root):
    if (root / "input-lock.json").exists():
        return validate(root)
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG))
    validate_factors(cfg["learning_rate_factors"])
    prior = ROOT / cfg["prior"]
    old = read(prior / "input-lock.json")
    assert read(prior / "status.json")["running"] is False
    assert old["history"] == cfg["history"] and old["seed"] == cfg["seed"]
    assert old["settings"]["offered"] == cfg["offered"]
    for name, expected in old["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"N74 source changed: {name}")
    fixed = prior / "trajectories" / cfg["history"] / f"seed_{cfg['seed']}" / "corrected_covariance"
    generation, checkpoint = load_checkpoint(fixed)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("forecaster.joblib", "calibration.json"):
        shutil.copy2(prior / name, root / name)
    files = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "cafl4ds/single_image_learning_rate.py",
        ROOT / "scripts/report_single_image_learning_rate.py",
        prior / "input-lock.json",
        prior / "stop.json",
        root / "forecaster.joblib",
        root / "calibration.json",
        generation / "commit.json",
        generation / "model.pt",
        generation / "trace.npz",
    ]
    lock = dict(
        settings={**old["settings"], **cfg},
        history=cfg["history"],
        seed=cfg["seed"],
        paths=old["paths"],
        parent=old["parent"],
        threshold=old["threshold"],
        covariance_cap=old["covariance_cap"],
        prior_fixed_branch=str(fixed),
        prior_fixed_generation=str(generation),
        prior_fixed_offered=checkpoint["done"],
        hashes={**old["hashes"], **{str(p.resolve()): digest(p) for p in files}},
        created_utc=datetime.now(timezone.utc).isoformat(),
        evaluated_trajectories=1,
        new_accept_all=0,
        final_queries_control_rates=False,
        development_queries_control_rates=False,
        policy_change="Ordered actual-AdamW learning-rate retries after measured covariance failure",
    )
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "status.json", dict(stage="prepared", running=False, completed_trajectories=0))
    report(root)
    return lock


def empty_trace(length, count):
    return dict(
        accepted=np.zeros(length, bool),
        j_admitted=np.zeros(length, bool),
        predictions=np.zeros(length),
        features=np.zeros((length, 86)),
        clipping=np.zeros(length),
        selected_factor=np.zeros(length),
        attempted=np.zeros((length, count), bool),
        trial_health=np.zeros((length, count, 7)),
        trial_development=np.zeros((length, count, 6)),
        trial_training=np.zeros((length, count, 2)),
        training=np.zeros((length, 2)),
        development_after=np.zeros((length, 6)),
        covariance_after=np.zeros(length),
    )


def run_arm(root, lock, env, forecaster, directory, parent, initial_obs, emit, target=None):
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, _, _, _ = env
    factors = validate_factors(settings["learning_rate_factors"])
    length, start = settings["offered"], settings["start"]
    target = length if target is None else target
    branch = directory / settings["arm"]
    branch.mkdir(parents=True, exist_ok=True)
    if (branch / "trajectory.json").exists():
        return read(branch / "trajectory.json")
    arrays = empty_trace(length, len(factors))
    costs = dict(features=0.0, prediction=0.0, state_copy=0.0, trial_training=0.0, observations=0.0, rollback=0.0)
    checkpoints = {str(start): read(directory / "common/readout.json")}
    state, done, research_seconds, restorations = parent, 0, 0.0, 0
    observations = [(v, None, None) for v in initial_obs]
    resumed = load_checkpoint(branch)
    if resumed is not None:
        generation, saved = resumed
        state = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
        assert fingerprint(state) == saved["fingerprint"]
        done, costs, checkpoints = saved["done"], saved["costs"], saved["checkpoints"]
        research_seconds, restorations = saved["research_seconds"], saved["restorations"]
        with np.load(generation / "trace.npz") as tr:
            for key in arrays:
                arrays[key][:] = tr[key]
            observations = [(v, None, None) for v in tr["last_observations"]]
    model, optim = restore(initial, config.optim, state)
    nominal = [float(g["lr"]) for g in parent["optimizer"]["param_groups"]]
    np.testing.assert_array_equal([g["lr"] for g in optim.param_groups], nominal)
    tick = time.perf_counter()
    current, probe, task = make_observation(model, bank, warm, dev, settings, rich=True)
    scores = development_scores(probe, dev, settings["soft_accuracy_temperature"])
    observations[-1] = current
    if not done:
        checkpoints[str(start)] = add_development(checkpoints[str(start)], scores, scores[:3])
        costs["observations"] += time.perf_counter() - tick
    arrivals = FrameStream(lock["paths"])
    for i in range(done, target):
        if i % settings["checkpoint_stride"] == 0:
            check_storage(settings)
        t = start + i
        draw = view_seed(settings["stream_namespace"], lock["history"], t, lock["seed"])
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
            draw,
            task,
            rich=True,
        )["rich"]
        arrays["features"][i] = vector
        arrays["clipping"][i] = forecaster.clipped_fraction(vector)
        costs["features"] += time.perf_counter() - tick
        tick = time.perf_counter()
        prediction = float(forecaster.predict(vector)[0])
        admitted = prediction < lock["threshold"]
        arrays["predictions"][i], arrays["j_admitted"][i] = prediction, admitted
        costs["prediction"] += time.perf_counter() - tick
        if admitted:
            tick = time.perf_counter()
            before = capture(model, optim)
            before_hash = fingerprint(before)
            costs["state_copy"] += time.perf_counter() - tick
            for j, factor in enumerate(factors):
                tick = time.perf_counter()
                set_rates(optim, nominal, factor)
                training = seeded_step(model, optim, arrivals[t : t + 1].cuda(), draw)
                arrays["attempted"][i, j] = True
                arrays["trial_training"][i, j] = training
                costs["trial_training"] += time.perf_counter() - tick
                tick = time.perf_counter()
                candidate, candidate_probe, candidate_task = make_observation(
                    model, bank, warm, dev, settings, rich=True
                )
                candidate_scores = development_scores(candidate_probe, dev, settings["soft_accuracy_temperature"])
                arrays["trial_health"][i, j] = candidate[0]
                arrays["trial_development"][i, j] = candidate_scores
                costs["observations"] += time.perf_counter() - tick
                if candidate[0][2] <= lock["covariance_cap"] + 1e-12:
                    set_rates(optim, nominal, 1.0)
                    arrays["accepted"][i] = True
                    arrays["selected_factor"][i] = factor
                    arrays["training"][i] = training
                    current, task, scores = candidate, candidate_task, candidate_scores
                    break
                tick = time.perf_counter()
                rollback(model, optim, before)
                assert fingerprint(capture(model, optim)) == before_hash
                restorations += 1
                costs["rollback"] += time.perf_counter() - tick
            del before
        arrays["development_after"][i] = scores
        arrays["covariance_after"][i] = current[0][2]
        observations = [*observations[-3:], current]
        if (i + 1) % settings["checkpoint_stride"] == 0 or i + 1 == target:
            tick = time.perf_counter()
            checkpoints[str(t + 1)] = add_development(
                measure(model, optim, env, branch / f"checkpoint_{t + 1}"), scores, scores[:3]
            )
            research_seconds += time.perf_counter() - tick
            saved = dict(
                done=i + 1,
                costs=costs,
                checkpoints=checkpoints,
                research_seconds=research_seconds,
                restorations=restorations,
                nominal_rates=nominal,
                fingerprint=fingerprint(capture(model, optim)),
            )
            commit_checkpoint(
                branch,
                capture(model, optim),
                {**arrays, "last_observations": np.array([v[0] for v in observations])},
                saved,
            )
            emit(
                stage="running",
                offered=i + 1,
                accepted=int(arrays["accepted"].sum()),
                rescued_updates=int(np.sum((arrays["selected_factor"] > 0) & (arrays["selected_factor"] < 1))),
                rejected_trial_steps=restorations,
                trial_steps=int(arrays["attempted"].sum()),
                nominal_rate_equivalent=float(arrays["selected_factor"].sum()),
                accuracy=checkpoints[str(t + 1)]["metrics"]["linear_accuracy_balanced"],
            )
    generation, saved = load_checkpoint(branch)
    result = dict(
        **saved,
        arm=settings["arm"],
        offered=target,
        accepted=int(arrays["accepted"].sum()),
        rescued_updates=int(np.sum((arrays["selected_factor"] > 0) & (arrays["selected_factor"] < 1))),
        trial_steps=int(arrays["attempted"].sum()),
        nominal_rate_equivalent=float(arrays["selected_factor"].sum()),
        parent_fingerprint=fingerprint(parent),
        final_fingerprint=saved["fingerprint"],
        policy_seconds=sum(costs.values()),
    )
    if target == length:
        shutil.copy2(generation / "model.pt", branch / "final_state.pt")
        shutil.copy2(generation / "trace.npz", branch / "trace.npz")
        atomic_json(branch / "trajectory.json", result)
        verify_arm(branch, lock, forecaster)
        for generation in list((branch / "resume_states").glob("step_*")):
            remove_generation(generation)
    del model, optim
    return result


def verify_arm(branch, lock, forecaster):
    record = read(branch / "trajectory.json")
    with np.load(branch / "trace.npz") as tr:
        n = record["offered"]
        np.testing.assert_allclose(forecaster.predict(tr["features"][:n]), tr["predictions"][:n], rtol=1e-10)
        np.testing.assert_array_equal(tr["j_admitted"][:n], tr["predictions"][:n] < lock["threshold"])
        selected = np.array(
            [
                choose_first_passing(
                    tr["attempted"][i],
                    tr["trial_health"][i, :, 2],
                    lock["settings"]["learning_rate_factors"],
                    lock["covariance_cap"],
                )
                for i in range(n)
            ]
        )
        np.testing.assert_array_equal(selected, tr["selected_factor"][:n])
        np.testing.assert_array_equal(selected > 0, tr["accepted"][:n])
        np.testing.assert_array_equal(tr["attempted"][:n, 0], tr["j_admitted"][:n])
        assert int(tr["attempted"][:n].sum() - tr["accepted"][:n].sum()) == record["restorations"]
        assert np.all(tr["covariance_after"][:n] <= lock["covariance_cap"] + 1e-12)
        assert all(np.isfinite(tr[k]).all() for k in tr.files)
    state = torch.load(branch / "final_state.pt", map_location="cpu", weights_only=True)
    parent = torch.load(lock["parent"], map_location="cpu", weights_only=True)
    assert fingerprint(state) == record["final_fingerprint"]
    np.testing.assert_array_equal([g["lr"] for g in state["optimizer"]["param_groups"]], record["nominal_rates"])
    for key, original in parent["optimizer"]["state"].items():
        if "step" in original:
            assert float(state["optimizer"]["state"][key]["step"]) - float(original["step"]) == record["accepted"]
    atomic_json(
        branch / "verification.json",
        dict(
            passed=True,
            decisions=n,
            exact_rollbacks=record["restorations"],
            optimizer_steps_equal_retained_updates=True,
        ),
    )


def rescue_smoke(root, lock, env):
    """A real stalled N74 state must admit a smaller step that fails at nominal."""
    generation = Path(lock["prior_fixed_generation"])
    parent = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
    nominal = [g["lr"] for g in parent["optimizer"]["param_groups"]]
    arrivals = FrameStream(lock["paths"])
    rows = []
    for offset in range(8):
        t = env[0]["start"] + lock["prior_fixed_offered"] + offset
        model, optim = restore(env[2], env[1].optim, parent)
        covariances = []
        accepted_factor = 0.0
        for factor in lock["settings"]["learning_rate_factors"]:
            set_rates(optim, nominal, factor)
            seeded_step(
                model,
                optim,
                arrivals[t : t + 1].cuda(),
                view_seed(env[0]["stream_namespace"], lock["history"], t, lock["seed"]),
            )
            covariance = float(health_read(model, env[3], env[4], env[0]["assay_seed"])[0][2])
            covariances.append(covariance)
            if covariance <= lock["covariance_cap"] + 1e-12:
                accepted_factor = factor
                set_rates(optim, nominal, 1.0)
                break
            rollback(model, optim, parent)
            assert fingerprint(capture(model, optim)) == fingerprint(parent)
        rows.append(dict(arrival=t - env[0]["start"] + 1, covariances=covariances, selected_factor=accepted_factor))
        del model, optim
        if 0 < accepted_factor < 1:
            atomic_json(
                root / "smoke/real-rescue.json", dict(passed=True, covariance_cap=lock["covariance_cap"], cases=rows)
            )
            return
    raise AssertionError("Bounded stalled-state check did not demonstrate a real smaller-rate rescue")


def smoke(root, lock, env, forecaster, parent, observations):
    folder = root / "smoke"
    if (folder / "verification.json").exists():
        assert read(folder / "verification.json")["passed"]
        return
    folder.mkdir(exist_ok=True)
    settings = dict(env[0], offered=64, checkpoint_stride=16)
    se = (settings, *env[1:])
    sl = dict(lock, settings=settings)
    common = folder / "common"
    common.mkdir(exist_ok=True)
    original = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "common"
    shutil.copy2(original / "readout.json", common / "readout.json")
    run_arm(root, sl, se, forecaster, folder, parent, observations, lambda **_: None)
    expected = read(folder / settings["arm"] / "trajectory.json")
    source = ROOT / lock["settings"]["prior"] / "smoke/corrected_covariance/trajectory.json"
    assert expected["final_fingerprint"] == read(source)["final_fingerprint"]
    resumed = folder / "resume"
    (resumed / "common").mkdir(parents=True, exist_ok=True)
    shutil.copy2(common / "readout.json", resumed / "common/readout.json")
    run_arm(root, sl, se, forecaster, resumed, parent, observations, lambda **_: None, target=32)
    actual = run_arm(root, sl, se, forecaster, resumed, parent, observations, lambda **_: None)
    assert actual["final_fingerprint"] == expected["final_fingerprint"]
    with np.load(folder / settings["arm"] / "trace.npz") as a:
        with np.load(resumed / settings["arm"] / "trace.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    rescue_smoke(root, lock, env)
    atomic_json(
        folder / "verification.json",
        dict(
            passed=True,
            exact_interrupted_resume=True,
            N74_prefix_reproduced=True,
            real_smaller_rate_rescue=True,
            optimizer_step_accounting=True,
        ),
    )


def run(root, lock, only_smoke=False):
    setup()
    env = environment(ROOT / lock["settings"]["source"])
    env[0].update(lock["settings"])
    forecaster = joblib.load(root / "forecaster.joblib")
    parent, observations, directory = initial_context(root, lock, env)
    smoke(root, lock, env, forecaster, parent, observations)
    if only_smoke:
        return
    began = time.perf_counter()

    def emit(**event):
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                total_trajectories=1,
                target_offered=lock["settings"]["offered"],
                elapsed_seconds=time.perf_counter() - began,
                free_gib=shutil.disk_usage(ROOT).free / 2**30,
                **event,
            ),
        )
        print(json.dumps(read(root / "status.json")), flush=True)
        report(root)

    emit(stage="starting")
    result = run_arm(root, lock, env, forecaster, directory, parent, observations, emit)
    validate(root)
    atomic_json(
        root / "verification.json",
        dict(
            passed=True,
            evaluated_trajectories=1,
            new_accept_all=0,
            accepted=result["accepted"],
            rescued_updates=result["rescued_updates"],
        ),
    )
    emit(stage="complete", offered=result["offered"], accepted=result["accepted"])
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
