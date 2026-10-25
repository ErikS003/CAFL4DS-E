"""Online task-informed admission, paired with the saved original heuristic."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_extended_gate import FrameStream, commit_checkpoint, load_checkpoint
from cafl4ds.single_image_filter import objective
from cafl4ds.single_image_filter_factorial import feature_vectors
from cafl4ds.single_image_gate_protection import development_scores, rollback
from cafl4ds.single_image_heuristic import HeuristicController
from cafl4ds.single_image_learning_rate import set_rates
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import seeded_step
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_extended_gate import measure
from scripts.run_single_image_filter_factorial import environment, make_observation
from cafl4ds.online_task_shadow import TaskShadow
from cafl4ds.online_task_admission import adjusted_probability
from scripts.audit_online_controller_replay import FEATURE_NAMES
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_heuristic_controller_proposal.json"
PRIOR = ROOT / "outputs/single-image-gate-protection/20261023-n73-pilot"
PROTOCOL = ROOT / "docs/project-plan/single-image-online-task-admission.md"


def validate(root):
    lock = read(root / "input-lock.json")
    for name, expected in lock["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return lock


def prepare(root):
    if (root / "input-lock.json").exists():
        return validate(root)
    old = read(ROOT / "outputs/single-image-heuristic/20261024-no-rollback/input-lock.json")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("forecaster.joblib", "calibration.json"):
        shutil.copy2(PRIOR / name, root / name)
    shutil.copy2(PROTOCOL, root / "frozen-protocol.md")
    settings = dict(old["settings"], seed=5003, trajectory_seeds=[5003], rank_weight=.25, variance_weight=.25)
    common = root / "trajectories" / old["history"] / "seed_5003/common"
    source_common = PRIOR / "trajectories" / old["history"] / "seed_5003/common"
    common.mkdir(parents=True, exist_ok=True)
    for name in ("readout.json", "raw.npz"):
        shutil.copy2(source_common / name, common / name)
    files = [Path(__file__), PROTOCOL, ROOT / "cafl4ds/online_task_shadow.py",
             ROOT / "scripts/report_single_image_task_admission.py",
             ROOT / "cafl4ds/online_task_admission.py",
             ROOT / "scripts/audit_online_controller_replay.py", root / "forecaster.joblib"]
    lock = dict(old, settings=settings, seed=5003, parent_seed=5003, action_seed=835003,
        authorization="User requested B from original heuristic starting point, reusing A and full",
        shadow=dict(prefix=128, half_life=256, ridge=10., intercept_ridge=.1,
                    target="immediate balanced development soft-error delta", controls_actions=True),
        hashes={**old["hashes"], **{str(p.resolve()): digest(p) for p in files}})
    atomic_json(root / "input-lock.json", lock)
    atomic_json(root / "status.json", dict(stage="prepared", running=False, offered=0, target_offered=settings["offered"]))
    return lock


def initial_context(root, lock, env):
    original = ROOT / lock["settings"]["source"] / "trajectories" / lock["history"] / f"seed_{lock['parent_seed']}"
    parent = torch.load(original / "parent.pt", map_location="cpu", weights_only=True)
    observations = np.load(original / "parent-observations.npz")["values"]
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    return parent, observations, directory


def arrays_for(n):
    return {
        "base_probability": np.zeros(n),
        "shadow_predictions": np.zeros((n, 4)),
        "shadow_ready": np.zeros(n, bool),
        "shadow_outcome": np.zeros(n),
        "shadow_residual_scale": np.zeros(n),
        "shadow_features": np.zeros((n, 19)),
        "accepted": np.zeros(n, bool),
        "attempted": np.zeros(n, bool),
        "features": np.zeros((n, 86)),
        "prediction": np.zeros(n),
        "prediction_valid": np.zeros(n, bool),
        "clipped": np.zeros(n),
        "probability": np.zeros(n),
        "draw": np.zeros(n),
        "rate": np.zeros(n),
        "mode": np.zeros(n, np.int8),
        "burst": np.zeros(n, bool),
        "health_warning": np.zeros(n, bool),
        "numerical_failure": np.zeros(n, bool),
        "task_rollback": np.zeros(n, bool),
        "backup_changed": np.zeros(n, bool),
        "effective_updates": np.zeros(n, np.int64),
        "displacement": np.zeros(n),
        "health": np.zeros((n, 7)),
        "J": np.zeros(n),
        "development": np.zeros((n, 6)),
        "training": np.zeros((n, 2)),
    }


def run_arm(root, lock, env, predictor, directory, parent, initial_obs, arm, emit, target=None, engineering=False, shadow_enabled=True):
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, _, _, _ = env
    cfg, n, start = lock["heuristic"], settings["offered"], settings["start"]
    end = n if target is None else target
    branch = directory / arm
    branch.mkdir(parents=True, exist_ok=True)
    if (branch / "trajectory.json").exists():
        return read(branch / "trajectory.json")
    trace, controller, rng = arrays_for(n), HeuristicController(), np.random.default_rng(lock["action_seed"])
    shadow = TaskShadow(prefix=4 if engineering else 128)
    schema = read(ROOT / "outputs/single-image-filter-factorial/20261021-probe-feature/feature-schema.json")["rich"]
    columns = [schema.index(k) for k in FEATURE_NAMES]
    observations = [(v, None, None) for v in initial_obs]
    state, backup, done, effective, backup_effective = parent, copy.deepcopy(parent), 0, 0, 0
    costs = dict.fromkeys(["features", "training", "observations", "rollback", "readouts", "checkpoint"], 0.0)
    checkpoints = {str(start): read(directory / "common/readout.json")}
    events, pre_brake = [], {}
    saved = load_checkpoint(branch)
    if saved:
        generation, record = saved
        bundle = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
        state, backup = bundle["live"], bundle["backup"]
        assert fingerprint(state) == record["fingerprint"]
        assert fingerprint(backup) == record["backup_fingerprint"]
        done, effective, backup_effective = record["done"], record["effective"], record["backup_effective"]
        controller, costs, checkpoints = (
            HeuristicController(**record["controller"]),
            record["costs"],
            record["checkpoints"],
        )
        rng.bit_generator.state = record["action_rng"]
        events, pre_brake = record["events"], record["pre_brake"]
        shadow = TaskShadow(state=record["shadow_state"])
        with np.load(generation / "trace.npz") as raw:
            for k in trace:
                trace[k][:] = raw[k]
            observations = [(v, None, None) for v in raw["last_observations"]]
    model, optim = restore(initial, config.optim, state)
    nominal = [float(g["lr"]) for g in parent["optimizer"]["param_groups"]]

    def observe():
        tick = time.perf_counter()
        current, probe, task = make_observation(model, bank, warm, dev, settings, rich=True)
        scores = development_scores(probe, dev, settings["soft_accuracy_temperature"])
        costs["observations"] += time.perf_counter() - tick
        return current, scores, task

    current, scores, task = observe()
    observations[-1] = current
    if saved:
        anchor, backup_scores, original_scores = (
            np.array(record["anchor"]),
            np.array(record["backup_scores"]),
            np.array(record["original_scores"]),
        )
    else:
        anchor, backup_scores, original_scores = current[0].copy(), scores.copy(), scores.copy()
    stream = FrameStream(lock["paths"])
    for i in range(done, end):
        if i % 128 == 0 and shutil.disk_usage(ROOT).free < 2.15 * 2**30:
            # Stop at the next durable boundary while preserving the 2-GiB reserve.
            atomic_json(root / "stop-request.json", {"reason": "storage reserve", "offered": i})
        t, draw = start + i, float(rng.random())
        seed = view_seed(settings["stream_namespace"], lock["history"], t, lock["seed"])
        trace["draw"][i] = draw
        if arm == "heuristic":
            tick = time.perf_counter()
            vector = feature_vectors(
                model,
                optim,
                stream,
                t,
                observations[-4:],
                centroid,
                projection,
                clusters,
                settings,
                seed,
                task,
                rich=True,
            )["rich"]
            clipped = float(predictor.clipped_fraction(vector))
            prediction = float(predictor.predict(vector)[0])
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite predictor observation")
            trace["features"][i], trace["clipped"][i] = vector, clipped
            trace["prediction_valid"][i] = np.isfinite(prediction)
            trace["prediction"][i] = prediction if np.isfinite(prediction) else 0.0
            p, factor, reason = controller.choose(i, prediction, clipped, lock["threshold"], draw, cfg)
            costs["features"] += time.perf_counter() - tick
        else:
            p, factor, reason = 0.25, 0.25 if draw < 0.25 else 0.0, "paced"
            if i < controller.blocked_until:
                p, factor, reason = 0.0, 0.0, "cooldown"
        trace["base_probability"][i] = p
        trace["probability"][i], trace["rate"][i] = p, factor
        trace["mode"][i] = {"normal": 0, "caution": 1, "recovery": 2}[controller.mode]
        trace["burst"][i] = reason == "burst"
        # Predict before training. For rejected images, use the LR that would
        # have applied if accepted; rejected images never supply training labels.
        candidate_rate = factor
        if not factor and p > 0:
            candidate_rate = (cfg["burst"]["lr_factor"] if reason == "burst" else
                              cfg["recovery"]["lr_factor"] if controller.mode == "recovery" else
                              cfg["health"][f"{controller.mode}_lr_factor"])
        prior_soft = float(scores[3])
        sx = np.r_[vector[columns], trace["prediction"][i], candidate_rate,
                   float(controller.mode == "caution"), prior_soft]
        if shadow_enabled:
            trace["shadow_features"][i] = sx
            trace["shadow_predictions"][i] = shadow.predict(sx)
            trace["shadow_ready"][i] = shadow.ready
            trace["shadow_residual_scale"][i] = shadow.residual_abs
            p = adjusted_probability(p, trace["shadow_predictions"][i, 3], shadow.ready, reason, controller.mode)
            factor = candidate_rate if draw < p else 0.0
            trace["probability"][i], trace["rate"][i] = p, factor
        if factor:
            tick = time.perf_counter()
            before = capture(model, optim)
            trace["attempted"][i] = True
            try:
                set_rates(optim, nominal, factor)
                training = seeded_step(model, optim, stream[t : t + 1].cuda(), seed)
                if not np.isfinite(training).all() or any(
                    not torch.isfinite(v).all() for v in model.state_dict().values()
                ):
                    raise FloatingPointError("Nonfinite training result")
                if engineering and i == 9:
                    raise FloatingPointError("Engineering injected failure after real update")
            except (FloatingPointError, RuntimeError) as error:
                if not isinstance(error, FloatingPointError) and "non-finite" not in str(error).lower():
                    raise
                rollback(model, optim, before)
                assert fingerprint(capture(model, optim)) == fingerprint(before)
                controller.recover(i + 1, cfg, numerical=True)
                trace["numerical_failure"][i] = True
                events.append(dict(arrival=i + 1, kind="numerical_failure", error=str(error)))
            else:
                set_rates(optim, nominal, 1.0)
                trace["training"][i] = training
                trace["displacement"][i] = (
                    sum(
                        float((p.detach() - before["model"][name].to(p.device)).double().square().sum())
                        for name, p in model.named_parameters()
                    )
                    ** 0.5
                )
                trace["accepted"][i] = True
                effective += 1
                controller.recent_commits.append(i)
            costs["training"] += time.perf_counter() - tick
            if trace["accepted"][i]:
                current, scores, task = observe()
                if shadow_enabled and i > 0:
                    outcome = float(scores[3])-prior_soft
                    trace["shadow_outcome"][i] = outcome
                    shadow.learn(sx, outcome, trace["shadow_predictions"][i, 3])
            del before
        if arm == "heuristic" and (i + 1) % cfg["health"]["check_every_arrivals"] == 0:
            trace["health_warning"][i] = controller.health(current[0], anchor, cfg)
        if arm == "heuristic" and (i + 1) % cfg["task_brake"]["development_readout_every_arrivals"] == 0:
            decision_scores = scores
            if engineering:
                decision_scores = backup_scores + (0.02 if i + 1 <= 8 else -0.06)
            command = controller.task(decision_scores, backup_scores, original_scores, cfg)
            if command == "backup":
                backup, backup_effective = capture(model, optim), effective
                anchor, backup_scores = current[0].copy(), scores.copy()
                trace["backup_changed"][i] = True
                events.append(dict(arrival=i + 1, kind="backup", effective=effective))
            elif command == "rollback":
                # Keep development sampling, task counters and backup promotion identical.
                # Suppress both state restoration and its consequent recovery phase.
                events.append(dict(arrival=i + 1, kind="suppressed_task_rollback",
                                   pre_development=scores.tolist(),
                                   backup_development=backup_scores.tolist()))
        observations = [*observations[-3:], current]
        trace["effective_updates"][i], trace["health"][i], trace["development"][i] = effective, current[0], scores
        trace["J"][i] = float(objective(current[0], env[10], settings))
        stopping = (root / "stop-request.json").exists()
        if (i + 1) % settings["checkpoint_stride"] == 0 or i + 1 == end or stopping:
            tick = time.perf_counter()
            checkpoints[str(t + 1)] = measure(model, optim, env, branch / f"checkpoint_{t + 1}")
            costs["readouts"] += time.perf_counter() - tick
            live = capture(model, optim)
            record = dict(
                done=i + 1,
                offered=i + 1,
                arm=arm,
                effective=effective,
                backup_effective=backup_effective,
                accepted=int(trace["accepted"].sum()),
                costs=costs,
                checkpoints=checkpoints,
                pre_brake=pre_brake,
                events=events,
                controller=asdict(controller),
                action_rng=rng.bit_generator.state,
                anchor=anchor.tolist(),
                backup_scores=backup_scores.tolist(),
                original_scores=original_scores.tolist(),
                fingerprint=fingerprint(live),
                backup_fingerprint=fingerprint(backup),
                engineering=engineering,
                shadow_state=shadow.state(),
            )
            if not engineering and not (root / "paired-prefix-verification.json").exists():
                old_branch = ROOT / "outputs/single-image-heuristic/20261024-no-rollback/trajectories/fresh_00/seed_5003/heuristic"
                old_generation, _ = load_checkpoint(old_branch)
                ready = np.flatnonzero(trace["shadow_ready"][:i+1])
                prefix_end = int(ready[0]) if len(ready) else i+1
                with np.load(old_generation / "trace.npz") as old_trace:
                    for key in ["accepted", "draw", "probability", "rate", "health", "J", "development"]:
                        np.testing.assert_array_equal(trace[key][:prefix_end], old_trace[key][:prefix_end])
                atomic_json(root / "paired-prefix-verification.json",
                            dict(passed=True, arrivals_compared=prefix_end, first_policy_arrival=prefix_end+1 if len(ready) else None))
            tick = time.perf_counter()
            commit_checkpoint(
                branch,
                {"live": live, "backup": backup},
                {**trace, "last_observations": np.array([o[0] for o in observations])},
                record,
            )
            costs["checkpoint"] += time.perf_counter() - tick
            emit(
                stage="running",
                checkpoint=True,
                arm=arm,
                offered=i + 1,
                accepted=record["accepted"],
                effective_updates=effective,
                rollbacks=int(trace["task_rollback"].sum()),
                mode=controller.mode,
                accuracy=checkpoints[str(t + 1)]["metrics"]["linear_accuracy_balanced"],
            )
        elif (i + 1) % 32 == 0:
            emit(stage="running", arm=arm, offered=i + 1, accepted=int(trace["accepted"].sum()), mode=controller.mode)
        if stopping:
            break
    generation, record = load_checkpoint(branch)
    if record["done"] == n:
        atomic_json(branch / "trajectory.json", record)
        verify(branch, parent, lock)
    return record


def verify(branch, parent, lock):
    generation, record = load_checkpoint(branch)
    bundle = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
    assert fingerprint(bundle["live"]) == record["fingerprint"]
    assert fingerprint(bundle["backup"]) == record["backup_fingerprint"]
    for key, old in parent["optimizer"]["state"].items():
        if "step" in old:
            assert float(bundle["live"]["optimizer"]["state"][key]["step"]) - float(old["step"]) == record["effective"]
    with np.load(generation / "trace.npz") as tr:
        n = record["done"]
        assert all(np.isfinite(tr[k]).all() for k in tr.files)
        assert np.all(tr["accepted"] <= tr["attempted"])
        assert int(tr["accepted"].sum()) == record["accepted"]
        assert tr["effective_updates"][n - 1] == record["effective"]
        assert np.all((tr["draw"][:n] >= 0) & (tr["draw"][:n] < 1))
        np.testing.assert_array_equal(tr["attempted"][:n], tr["draw"][:n] < tr["probability"][:n])
        np.testing.assert_array_equal(tr["accepted"][:n], tr["attempted"][:n] & ~tr["numerical_failure"][:n])
        cfg = lock["heuristic"]
        effective, backup_effective = 0, 0
        for i in range(n):
            if record["arm"] == "paced":
                expected_p, expected_rate = 0.25, 0.25
            elif tr["mode"][i] == 2:
                expected_p, expected_rate = cfg["recovery"]["update_probability"], cfg["recovery"]["lr_factor"]
            elif tr["burst"][i]:
                expected_p, expected_rate = 1.0, cfg["burst"]["lr_factor"]
            else:
                p = cfg["prediction"]
                if not tr["prediction_valid"][i] or tr["clipped"][i] > p["clipped_fraction_fallback_above"]:
                    expected_p = p["fallback_probability"]
                elif tr["prediction"][i] < lock["threshold"]:
                    expected_p = p["probability_good"]
                elif tr["prediction"][i] < 0:
                    expected_p = p["probability_borderline"]
                else:
                    expected_p = p["probability_unfavorable"]
                expected_rate = cfg["health"]["normal_lr_factor" if tr["mode"][i] == 0 else "caution_lr_factor"]
            assert tr["base_probability"][i] == expected_p or tr["base_probability"][i] == 0
            reason = "burst" if tr["burst"][i] else "fallback"
            mode = {0:"normal", 1:"caution", 2:"recovery"}[int(tr["mode"][i])]
            expected_p = adjusted_probability(expected_p, tr["shadow_predictions"][i,3],
                                              bool(tr["shadow_ready"][i]), reason, mode)
            if tr["probability"][i] == 0:
                preceding = np.flatnonzero(tr["numerical_failure"][:i])
                assert len(preceding) and i <= preceding[-1] + cfg["numerical_cooldown_arrivals"]
            else:
                assert tr["probability"][i] == expected_p
            assert tr["rate"][i] == (expected_rate if tr["attempted"][i] else 0)
            effective += int(tr["accepted"][i])
            if tr["backup_changed"][i]:
                backup_effective = effective
            if tr["task_rollback"][i]:
                effective = backup_effective
            assert tr["effective_updates"][i] == effective
    atomic_json(
        branch / "verification.json",
        dict(
            passed=True,
            offered=record["done"],
            effective=record["effective"],
            optimizer_accounting=True,
            finite_trace=True,
        ),
    )


def smoke(root, lock, env, parent, observations, predictor):
    folder = root / "smoke"
    if (folder / "verification.json").exists():
        assert read(folder / "verification.json")["passed"]
        return
    cfg = copy.deepcopy(lock["heuristic"])
    cfg["prediction"].update(
        probability_good=1.0, probability_borderline=1.0, probability_unfavorable=1.0, fallback_probability=1.0
    )
    cfg["health"]["check_every_arrivals"] = 4
    cfg["task_brake"]["development_readout_every_arrivals"] = 4
    cfg["numerical_cooldown_arrivals"] = 1
    cfg["recovery"].update(arrivals=2, update_probability=1.0)
    settings = dict(env[0], offered=16, checkpoint_stride=8)
    sl, se = dict(lock, heuristic=cfg, settings=settings), (settings, *env[1:])
    common = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "common/readout.json"
    for variant in ("straight", "resume"):
        (folder / variant / "common").mkdir(parents=True, exist_ok=True)
        shutil.copy2(common, folder / variant / "common/readout.json")
    a = run_arm(
        root,
        sl,
        se,
        predictor,
        folder / "straight",
        parent,
        observations,
        "heuristic",
        lambda **_: None,
        engineering=True,
    )
    run_arm(
        root,
        sl,
        se,
        predictor,
        folder / "resume",
        parent,
        observations,
        "heuristic",
        lambda **_: None,
        target=13,
        engineering=True,
    )
    b = run_arm(
        root,
        sl,
        se,
        predictor,
        folder / "resume",
        parent,
        observations,
        "heuristic",
        lambda **_: None,
        engineering=True,
    )
    assert a["fingerprint"] == b["fingerprint"] and a["backup_fingerprint"] == b["backup_fingerprint"]
    assert a["controller"] == b["controller"] and a["action_rng"] == b["action_rng"]
    ga, _ = load_checkpoint(folder / "straight/heuristic")
    gb, _ = load_checkpoint(folder / "resume/heuristic")
    with np.load(ga / "trace.npz") as x, np.load(gb / "trace.npz") as y:
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
        assert not x["task_rollback"].any() and x["backup_changed"].any() and x["numerical_failure"].any()
        assert any(e["kind"] == "suppressed_task_rollback" for e in a["events"])
    with np.load(ga / "trace.npz") as x:
        assert np.any(x["probability"] != x["base_probability"])
        first = np.flatnonzero(x["shadow_ready"])[0]
        np.testing.assert_array_equal(x["probability"][:first], x["base_probability"][:first])
    atomic_json(
        folder / "verification.json",
        dict(
            passed=True,
            real_GPU_updates=True,
            online_probability_adjustment_verified=True,
            exact_resume=True,
            backup_transfer=True,
            task_rollback=False,
            suppressed_task_rollback=True,
            injected_numerical_rollback=True,
            controller_and_action_rng_resume=True,
            engineering_overrides_only=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=["prepare", "smoke", "run", "report"], default="run")
    args = parser.parse_args()
    root = args.output.resolve()
    from scripts.report_single_image_task_admission import report

    if args.phase == "report":
        report(root)
        return
    lock = prepare(root)
    if args.phase == "prepare":
        report(root)
        return
    lease = root / "worker.lease"
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    began = time.perf_counter()

    def emit(**event):
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                running=True,
                elapsed_seconds=time.perf_counter() - began,
                target_offered=lock["settings"]["offered"],
                free_gib=shutil.disk_usage(ROOT).free / 2**30,
                **event,
            ),
        )
        print(json.dumps(read(root / "status.json")), flush=True)
        if event.get("checkpoint"):
            report(root)

    try:
        with threadpool_limits(limits=1):
            setup()
            env = environment(ROOT / lock["settings"]["source"])
            env[0].update(lock["settings"])
            predictor = joblib.load(root / "forecaster.joblib")
            parent, observations, directory = initial_context(root, lock, env)
            emit(stage="preflight", offered=0)
            smoke(root, lock, env, parent, observations, predictor)
            validate(root)
            if args.phase == "run":
                for arm in lock["arms"]:
                    if (root / "stop-request.json").exists():
                        break
                    run_arm(root, lock, env, predictor, directory, parent, observations, arm, emit)
            stage = (
                "prepared"
                if args.phase == "smoke"
                else "stopped"
                if (root / "stop-request.json").exists()
                else "complete"
            )
            atomic_json(root / "status.json", dict(read(root / "status.json"), running=False, stage=stage))
            validate(root)
            report(root)
    except Exception as error:
        atomic_json(root / "failure.json", dict(error=repr(error), pid=os.getpid()))
        atomic_json(root / "status.json", dict(read(root / "status.json"), stage="failed", running=False))
        raise
    finally:
        lease.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
