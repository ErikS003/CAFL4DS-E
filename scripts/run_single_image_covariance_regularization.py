"""N77: bank covariance regularization, adaptive LR and an unchanged hard cap."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_covariance_regularization import apply_gradients, next_factor, prepare_gradients
from cafl4ds.single_image_extended_gate import FrameStream, commit_checkpoint, load_checkpoint
from cafl4ds.single_image_filter_factorial import feature_vectors
from cafl4ds.single_image_gate_protection import BoundedForecaster, development_scores, rollback
from cafl4ds.single_image_learning_rate import set_rates
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import seeded_step
from scripts.report_single_image_covariance_regularization import report
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_covariance_relaxation import recover_publication
from scripts.run_single_image_extended_gate import measure
from scripts.run_single_image_filter_factorial import environment, make_observation
from scripts.run_single_image_gate_protection import add_development, check_storage, initial_context
from scripts.run_single_image_learning_rate import validate
from scripts.run_single_image_nonlinear import health_read
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_covariance_regularization.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-covariance-regularization-frozen.md"


def prepare(root: Path) -> dict[str, Any]:
    """Freeze the matched parent, stream and upstream J heuristic before GPU work."""
    if (root / "input-lock.json").exists():
        return validate(root)
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG))
    prior = ROOT / cfg["prior_covariance"]
    old = validate(prior)
    if read(prior / "status.json")["running"]:
        raise ValueError("Retire the stalled N76 process before replacing it")
    branch = prior / "trajectories" / old["history"] / f"seed_{old['seed']}" / old["settings"]["arm"]
    generation, saved = load_checkpoint(branch)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("forecaster.joblib", "calibration.json"):
        shutil.copy2(prior / name, root / name)
    files = [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "cafl4ds/single_image_covariance_regularization.py",
        ROOT / "scripts/report_single_image_covariance_regularization.py",
        prior / "input-lock.json",
        prior / "stop.json",
        generation / "commit.json",
        generation / "model.pt",
        generation / "trace.npz",
        root / "forecaster.joblib",
        root / "calibration.json",
    ]
    lock = dict(
        old,
        settings={**old["settings"], **cfg},
        prior_covariance_generation=str(generation),
        prior_covariance_offered=saved["done"],
        hashes={**old["hashes"], **{str(p.resolve()): digest(p) for p in files}},
        created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_J_is_heuristic=True,
        final_queries_control_rates=False,
        development_queries_control_rates=False,
        policy_change="Exact full-bank covariance/variance gradient plus measured secant LR backtracking",
    )
    if not (lock["parent"] == old["parent"] and lock["paths"] == old["paths"]):
        raise AssertionError("Scientific state verification failed")
    if not cfg["covariance_warm_multiplier"] == old["settings"]["covariance_warm_multiplier"] == 1.3:
        raise AssertionError("Scientific state verification failed")
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "status.json", {"stage": "prepared", "running": False, "target_offered": cfg["offered"]})
    report(root)
    return lock


def empty_trace(length: int, count: int) -> dict[str, np.ndarray]:
    """Bound all per-arrival logs, including actual continuously chosen trial rates."""
    return {
        "accepted": np.zeros(length, bool),
        "j_admitted": np.zeros(length, bool),
        "predictions": np.zeros(length),
        "features": np.zeros((length, 86)),
        "clipping": np.zeros(length),
        "selected_factor": np.zeros(length),
        "attempted": np.zeros((length, count), bool),
        "trial_factors": np.zeros((length, count)),
        "trial_health": np.zeros((length, count, 7)),
        "trial_development": np.zeros((length, count, 6)),
        "trial_training": np.zeros((length, count, 2)),
        "training": np.zeros((length, 2)),
        "development_after": np.zeros((length, 6)),
        "covariance_after": np.zeros(length),
        "regularization": np.zeros((length, 6)),
        "displacement_norm": np.zeros(length),
    }


def run_arm(  # noqa: C901 - Transactional scientific checks stay together.
    root: Path,
    lock: dict[str, Any],
    env: tuple,
    forecaster: BoundedForecaster,
    directory: Path,
    parent: dict[str, Any],
    initial_obs: np.ndarray,
    emit: Callable[..., None],
    target: int | None = None,
) -> dict[str, Any]:
    """Run arm."""
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, _, _, _ = env
    count = settings["maximum_trials"]
    length, start = (settings["offered"], settings["start"])
    target = length if target is None else target
    branch = directory / settings["arm"]
    branch.mkdir(parents=True, exist_ok=True)
    if (branch / "trajectory.json").exists():
        return read(branch / "trajectory.json")
    arrays = empty_trace(length, count)
    costs = {
        "features": 0.0,
        "prediction": 0.0,
        "state_copy": 0.0,
        "trial_training": 0.0,
        "observations": 0.0,
        "rollback": 0.0,
    }
    checkpoints = {str(start): read(directory / "common/readout.json")}
    state, done, research_seconds, restorations = (parent, 0, 0.0, 0)
    observations = [(v, None, None) for v in initial_obs]
    resumed = load_checkpoint(branch)
    if resumed is not None:
        generation, saved = resumed
        state = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
        if not fingerprint(state) == saved["fingerprint"]:
            raise AssertionError("Scientific state verification failed")
        done, costs, checkpoints = (saved["done"], saved["costs"], saved["checkpoints"])
        research_seconds, restorations = (saved["research_seconds"], saved["restorations"])
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
        arrays["predictions"][i], arrays["j_admitted"][i] = (prediction, admitted)
        costs["prediction"] += time.perf_counter() - tick
        if admitted:
            tick = time.perf_counter()
            before = capture(model, optim)
            before_hash = fingerprint(before)
            costs["state_copy"] += time.perf_counter() - tick
            tick = time.perf_counter()
            gradients, gradient_rng, diagnostic = prepare_gradients(
                model, optim, arrivals[t : t + 1].cuda(), bank, current[1], env[10], draw, settings
            )
            arrays["regularization"][i] = [
                diagnostic[k]
                for k in (
                    "regularization_loss",
                    "covariance_weight",
                    "covariance_penalty",
                    "variance_penalty",
                    "ssl_loss",
                    "gradient_norm",
                )
            ]
            costs["trial_training"] += time.perf_counter() - tick
            rollback(model, optim, before)
            factor = 1.0
            for j in range(count):
                if factor < settings["minimum_rate_factor"]:
                    break
                tick = time.perf_counter()
                set_rates(optim, nominal, factor)
                apply_gradients(model, optim, gradients, gradient_rng)
                training = (diagnostic["ssl_loss"], diagnostic["gradient_norm"])
                arrays["attempted"][i, j] = True
                arrays["trial_factors"][i, j] = factor
                arrays["trial_training"][i, j] = training
                costs["trial_training"] += time.perf_counter() - tick
                tick = time.perf_counter()
                candidate = health_read(model, bank, warm, settings["assay_seed"])
                arrays["trial_health"][i, j] = candidate[0]
                costs["observations"] += time.perf_counter() - tick
                if candidate[0][2] <= lock["covariance_cap"] + 1e-12:
                    tick = time.perf_counter()
                    candidate, candidate_probe, candidate_task = make_observation(
                        model, bank, warm, dev, settings, rich=True
                    )
                    candidate_scores = development_scores(candidate_probe, dev, settings["soft_accuracy_temperature"])
                    arrays["trial_development"][i, j] = candidate_scores
                    costs["observations"] += time.perf_counter() - tick
                    displacement = (
                        sum(
                            (
                                float((value.detach() - before["model"][name].to(value.device)).double().square().sum())
                                for name, value in model.named_parameters()
                            )
                        )
                        ** 0.5
                    )
                    arrays["displacement_norm"][i] = displacement
                    set_rates(optim, nominal, 1.0)
                    arrays["accepted"][i] = True
                    arrays["selected_factor"][i] = factor
                    arrays["training"][i] = training
                    current, task, scores = (candidate, candidate_task, candidate_scores)
                    break
                tick = time.perf_counter()
                rollback(model, optim, before)
                if not fingerprint(capture(model, optim)) == before_hash:
                    raise AssertionError("Scientific state verification failed")
                restorations += 1
                costs["rollback"] += time.perf_counter() - tick
                factor = next_factor(
                    factor, current[0][2], candidate[0][2], lock["covariance_cap"], settings["headroom_safety"]
                )
            del gradients, gradient_rng
            del before
        arrays["development_after"][i] = scores
        arrays["covariance_after"][i] = current[0][2]
        observations = [*observations[-3:], current]
        stopping = (root / "stop-request.json").exists()
        if (i + 1) % settings["checkpoint_stride"] == 0 or i + 1 == target or stopping:
            tick = time.perf_counter()
            checkpoints[str(t + 1)] = add_development(
                measure(model, optim, env, branch / f"checkpoint_{t + 1}"), scores, scores[:3]
            )
            research_seconds += time.perf_counter() - tick
            saved = {
                "done": i + 1,
                "costs": costs,
                "checkpoints": checkpoints,
                "research_seconds": research_seconds,
                "restorations": restorations,
                "nominal_rates": nominal,
                "fingerprint": fingerprint(capture(model, optim)),
            }
            commit_checkpoint(
                branch,
                capture(model, optim),
                {**arrays, "last_observations": np.array([v[0] for v in observations])},
                saved,
            )
            emit(
                stage="running",
                checkpoint=True,
                offered=i + 1,
                accepted=int(arrays["accepted"].sum()),
                rescued_updates=int(np.sum((arrays["selected_factor"] > 0) & (arrays["selected_factor"] < 1))),
                rejected_trial_steps=restorations,
                trial_steps=int(arrays["attempted"].sum()),
                nominal_rate_equivalent=float(arrays["selected_factor"].sum()),
                accuracy=checkpoints[str(t + 1)]["metrics"]["linear_accuracy_balanced"],
            )
        elif (i + 1) % settings["progress_stride"] == 0:
            emit(
                stage="running",
                offered=i + 1,
                accepted=int(arrays["accepted"].sum()),
                covariance_warm_ratio=float(current[0][2] / env[10][2]),
                committed_offered=max(map(int, checkpoints)) - start,
            )
        if stopping:
            target = i + 1
            break
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
    del model, optim
    return result


def verify_arm(branch: Path, lock: dict[str, Any], forecaster: BoundedForecaster) -> None:  # noqa: C901 - Transactional scientific checks stay together.
    """Reconstruct every admission, secant retry, hard barrier and live step count."""
    record = read(branch / "trajectory.json")
    settings = lock["settings"]
    with np.load(branch / "trace.npz") as tr:
        n = record["offered"]
        np.testing.assert_allclose(forecaster.predict(tr["features"][:n]), tr["predictions"][:n], rtol=1e-10)
        np.testing.assert_array_equal(tr["j_admitted"][:n], tr["predictions"][:n] < lock["threshold"])
        parent_cov = record["checkpoints"][str(settings["start"])]["metrics"]["selector_covariance"]
        for i in range(n):
            count = int(tr["attempted"][i].sum())
            if not bool(count) == bool(tr["j_admitted"][i]):
                raise AssertionError("Scientific state verification failed")
            np.testing.assert_array_equal(tr["attempted"][i], np.arange(settings["maximum_trials"]) < count)
            expected = 0.0
            for j in range(count):
                factor = tr["trial_factors"][i, j]
                if not (factor >= settings["minimum_rate_factor"] and factor <= 1):
                    raise AssertionError("Scientific state verification failed")
                if j == 0:
                    if not factor == 1.0:
                        raise AssertionError("Scientific state verification failed")
                else:
                    prior_factor = next_factor(
                        tr["trial_factors"][i, j - 1],
                        parent_cov,
                        tr["trial_health"][i, j - 1, 2],
                        lock["covariance_cap"],
                        settings["headroom_safety"],
                    )
                    if not factor == prior_factor:
                        raise AssertionError("Scientific state verification failed")
                if tr["trial_health"][i, j, 2] <= lock["covariance_cap"] + 1e-12:
                    if not j == count - 1:
                        raise AssertionError("Scientific state verification failed")
                    expected = factor
            if not tr["selected_factor"][i] == expected:
                raise AssertionError("Scientific state verification failed")
            if not tr["accepted"][i] == bool(expected):
                raise AssertionError("Scientific state verification failed")
            if not expected and count:
                if not (
                    count == settings["maximum_trials"]
                    or next_factor(
                        tr["trial_factors"][i, count - 1],
                        parent_cov,
                        tr["trial_health"][i, count - 1, 2],
                        lock["covariance_cap"],
                        settings["headroom_safety"],
                    )
                    < settings["minimum_rate_factor"]
                ):
                    raise AssertionError("Scientific state verification failed")
            parent_cov = float(tr["covariance_after"][i])
        if not np.all(tr["covariance_after"][:n] <= lock["covariance_cap"] + 1e-12):
            raise AssertionError("Scientific state verification failed")
        if not int(tr["attempted"][:n].sum() - tr["accepted"][:n].sum()) == record["restorations"]:
            raise AssertionError("Scientific state verification failed")
        if not all(np.isfinite(tr[k]).all() for k in tr.files):
            raise AssertionError("Scientific state verification failed")
    state = torch.load(branch / "final_state.pt", map_location="cpu", weights_only=True)
    parent = torch.load(lock["parent"], map_location="cpu", weights_only=True)
    if not fingerprint(state) == record["final_fingerprint"]:
        raise AssertionError("Scientific state verification failed")
    for key, original in parent["optimizer"]["state"].items():
        if "step" in original:
            if not float(state["optimizer"]["state"][key]["step"]) - float(original["step"]) == record["accepted"]:
                raise AssertionError("Scientific state verification failed")
    atomic_json(
        branch / "verification.json",
        {
            "passed": True,
            "decisions": n,
            "complete_rollbacks": record["restorations"],
            "optimizer_step_accounting": True,
        },
    )


def smoke(  # noqa: C901 - Transactional scientific checks stay together.
    root: Path,
    lock: dict[str, Any],
    env: tuple,
    forecaster: BoundedForecaster,
    parent: dict[str, Any],
    observations: np.ndarray,
) -> None:
    """Verify real inward pressure, smaller-rate rescue and exact interrupted resume."""
    folder = root / "smoke"
    if (folder / "verification.json").exists():
        if not read(folder / "verification.json")["passed"]:
            raise AssertionError("Scientific state verification failed")
        return
    folder.mkdir(exist_ok=True)
    generation = Path(lock["prior_covariance_generation"])
    stalled = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
    model, optimizer = restore(env[2], env[1].optim, stalled)
    observation = health_read(model, env[3], env[4], env[0]["assay_seed"])
    arrivals = FrameStream(lock["paths"])
    t = env[0]["start"] + lock["prior_covariance_offered"]
    seed = view_seed(env[0]["stream_namespace"], lock["history"], t, lock["seed"])
    rows, decreased = [], False
    nominal = [g["lr"] for g in stalled["optimizer"]["param_groups"]]
    for offset in range(4):
        current = health_read(model, env[3], env[4], env[0]["assay_seed"])
        before = capture(model, optimizer)
        draw = view_seed(env[0]["stream_namespace"], lock["history"], t + offset, lock["seed"])
        gradients, rng, diagnostic = prepare_gradients(
            model, optimizer, arrivals[t + offset : t + offset + 1].cuda(), env[3], current[1], env[10], draw, env[0]
        )
        rollback(model, optimizer, before)
        factor, trials, retained = 1.0, [], False
        for _ in range(env[0]["maximum_trials"]):
            if factor < env[0]["minimum_rate_factor"]:
                break
            set_rates(optimizer, nominal, factor)
            apply_gradients(model, optimizer, gradients, rng)
            after = health_read(model, env[3], env[4], env[0]["assay_seed"])[0]
            trials.append({"factor": factor, "covariance": float(after[2])})
            if after[2] <= lock["covariance_cap"]:
                retained = True
                decreased = bool(after[2] < current[0][2])
                set_rates(optimizer, nominal, 1.0)
                break
            rollback(model, optimizer, before)
            factor = next_factor(factor, current[0][2], after[2], lock["covariance_cap"], 0.8)
        rows.append(
            {
                "offset": offset,
                "before_covariance": float(current[0][2]),
                "retained": retained,
                "decreased": decreased,
                "trials": trials,
                "diagnostic": diagnostic,
            }
        )
        if decreased:
            break
    atomic_json(
        folder / "stalled-direction.json",
        {"passed": decreased, "cases": rows, "covariance_cap": lock["covariance_cap"]},
    )
    if not decreased:
        raise AssertionError("Bounded regularized recovery did not restore stalled-state headroom")
    rollback(model, optimizer, stalled)
    raw = dict(env[0], covariance_weight=0.0, variance_weight=0.0)
    grads, random, _ = prepare_gradients(
        model, optimizer, arrivals[t : t + 1].cuda(), env[3], observation[1], env[10], seed, raw
    )
    rollback(model, optimizer, stalled)
    seeded_step(model, optimizer, arrivals[t : t + 1].cuda(), seed)
    unregularized = capture(model, optimizer)
    rollback(model, optimizer, stalled)
    apply_gradients(model, optimizer, grads, random)
    if not fingerprint(capture(model, optimizer)) == fingerprint(unregularized):
        raise AssertionError("Scientific state verification failed")
    full_cov = health_read(model, env[3], env[4], env[0]["assay_seed"])[0][2]
    if not full_cov > observation[0][2]:
        raise AssertionError("Scientific state verification failed")
    factor = next_factor(1.0, observation[0][2], full_cov, lock["covariance_cap"], 0.8)
    rollback(model, optimizer, stalled)
    before_hash = fingerprint(stalled)
    if not fingerprint(capture(model, optimizer)) == before_hash:
        raise AssertionError("Scientific state verification failed")
    small_cov = full_cov
    for _ in range(env[0]["maximum_trials"]):
        if factor < env[0]["minimum_rate_factor"]:
            break
        set_rates(optimizer, nominal, factor)
        apply_gradients(model, optimizer, grads, random)
        small_cov = health_read(model, env[3], env[4], env[0]["assay_seed"])[0][2]
        if small_cov <= lock["covariance_cap"]:
            break
        rollback(model, optimizer, stalled)
        factor = next_factor(factor, observation[0][2], small_cov, lock["covariance_cap"], 0.8)
    if not small_cov <= lock["covariance_cap"]:
        raise AssertionError("Scientific state verification failed")
    atomic_json(
        folder / "secant-rescue.json",
        {
            "passed": True,
            "factor": factor,
            "nominal_covariance": float(full_cov),
            "smaller_covariance": float(small_cov),
        },
    )
    del model, optimizer, gradients, grads
    settings = dict(env[0], offered=16, checkpoint_stride=8)
    se, sl = ((settings, *env[1:]), dict(lock, settings=settings))
    original = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "common/readout.json"
    for directory in (folder / "uninterrupted", folder / "resume"):
        (directory / "common").mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, directory / "common/readout.json")
    expected = run_arm(root, sl, se, forecaster, folder / "uninterrupted", parent, observations, lambda **_: None)
    run_arm(root, sl, se, forecaster, folder / "resume", parent, observations, lambda **_: None, target=8)
    actual = run_arm(root, sl, se, forecaster, folder / "resume", parent, observations, lambda **_: None)
    if not expected["final_fingerprint"] == actual["final_fingerprint"]:
        raise AssertionError("Scientific state verification failed")
    with np.load(folder / "uninterrupted" / settings["arm"] / "trace.npz") as a:
        with np.load(folder / "resume" / settings["arm"] / "trace.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    atomic_json(
        folder / "verification.json",
        {
            "passed": True,
            "exact_interrupted_resume": True,
            "real_inward_covariance_step": True,
            "real_smaller_rate_rescue": True,
            "zero_penalty_matches_original": True,
            "optimizer_step_accounting": True,
        },
    )


def run(root: Path, lock: dict[str, Any], only_smoke: bool = False) -> None:
    """Run one matched trajectory and publish all metrics after each checkpoint."""
    setup()
    env = environment(ROOT / lock["settings"]["source"])
    env[0].update(lock["settings"])
    forecaster = joblib.load(root / "forecaster.joblib")
    parent, observations, directory = initial_context(root, lock, env)
    began = time.perf_counter()

    def emit(**event: object) -> None:
        """Emit."""
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                target_offered=lock["settings"]["offered"],
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
    if only_smoke:
        atomic_json(root / "status.json", dict(read(root / "status.json"), running=False, stage="preflight_passed"))
        return
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
    report(root)


def main() -> None:
    """Prepare, preflight, resume or report the single authorized run."""
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
            run(root, lock, args.phase == "smoke")
    except Exception as error:
        atomic_json(root / "failure.json", {"pid": os.getpid(), "error": repr(error)})
        atomic_json(root / "status.json", dict(read(root / "status.json"), running=False, stage="failed"))
        raise


if __name__ == "__main__":
    main()
