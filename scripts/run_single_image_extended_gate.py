"""Run a resumable long-stream threshold sweep with independent random policies."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_execution_amendment import active_specs, capped_histories, validate_hashes
from cafl4ds.single_image_extended_gate import FrameStream, commit_checkpoint, load_checkpoint, remove_generation
from cafl4ds.single_image_filter import matched_random, objective
from cafl4ds.single_image_filter_factorial import feature_vectors, linear_readout
from cafl4ds.single_image_forecast import HEALTH
from cafl4ds.single_image_long_gate import finite_metrics, global_schedule, independent_schedule, policy_specs
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import performance64, seeded_step
from scripts.prepare_single_image_extended_gate import acquire, plan, validate_plan
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_filter_factorial import environment, make_observation
from scripts.run_single_image_nonlinear import health_read
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_extended_gate.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-extended-gate-frozen.md"


def check_storage(settings):
    if shutil.disk_usage(ROOT).free < settings["minimum_free_gib"] * 2**30:
        raise RuntimeError("Storage guard: less than the reserved free space remains; resume after cleanup")


def freeze_schedules(root, settings, histories):
    """Freeze independent schedules before any gate sees the new continuation."""
    if (root / "schedule-lock.json").exists():
        assert digest(root / "independent-schedules.npz") == read(root / "schedule-lock.json")["sha256"]
        return
    specs = policy_specs(settings, read(ROOT / settings["source"] / "model-selection.json")["J_rich"]["threshold"])
    schedules = {}
    for history in histories:
        for seed in settings["trajectory_seeds"]:
            for arm, spec in specs.items():
                if spec["kind"] == "independent":
                    schedules[f"{history}__{seed}__{arm}"] = independent_schedule(
                        settings["offered"],
                        spec["probability"],
                        view_seed(settings["random_namespace"] + arm, history, settings["start"], seed),
                    )
    np.savez_compressed(root / "independent-schedules.npz", **schedules)
    atomic_json(
        root / "schedule-lock.json",
        dict(
            sha256=digest(root / "independent-schedules.npz"),
            fixed_before_any_gate=True,
            gate_decisions_used=False,
            schedules=len(schedules),
        ),
    )


def prepare(root, settings):
    """Seal the acquired membership and hashes, preserving the pre-acquisition source lock."""
    selection = validate_plan(root)
    if (root / "input-lock.json").exists():
        assert_inputs(root)
        return
    source = ROOT / settings["source"]
    intake = read(root / "intake/intake.json")
    assert intake["complete"] and intake["new_unique_clip_ids"] == 1200
    histories = intake["histories"]
    old = read(source / "intake/intake.json")
    prefix = [c for c in old["clips"] if any(c["clip_id"] in meta["clip_ids"][:6] for meta in histories.values())]
    assert len(prefix) == 36
    clips = intake["clips"] + prefix
    assert len({c["clip_id"] for c in clips}) == 1236
    inherited = [source / name for name in ("forecasters.joblib", "model-selection.json", "calibration.npz")]
    inherited += [root / "selection-lock.json", root / "intake/intake.json"]
    inherited += [source / f"intake/{role}-panel.pt" for role in ("development", "evaluation")]
    signal = ROOT / "outputs/single-image-signal/20261021-followup"
    inherited += [signal / name for name in ("input_lock.json", "bank.pt", "warm_bank.npy")]
    inherited.append(Path(read(signal / "input_lock.json")["learner_config"]["well"]))
    for clip in clips:
        assert digest(Path(clip["frames_path"])) == clip["frames_sha256"]
        inherited.append(Path(clip["frames_path"]))
    for history in histories:
        assert len(histories[history]["paths"]) * 96 == settings["start"] + settings["offered"]
        for seed in settings["trajectory_seeds"]:
            directory = source / "trajectories" / history / f"seed_{seed}"
            inherited += [directory / name for name in ("parent.pt", "parent-observations.npz", "common/readout.json")]
    specs = policy_specs(settings, read(source / "model-selection.json")["J_rich"]["threshold"])
    freeze_schedules(root, settings, histories)
    atomic_json(
        root / "input-lock.json",
        dict(
            settings=settings,
            specs=specs,
            histories=histories,
            hashes={**selection["source_hashes"], **{str(p.resolve()): digest(p) for p in inherited}},
            exploratory_followup=True,
            new_continuation_clip_ids=1200,
            reused_prefix_clip_ids=36,
            query_clip_overlap=0,
            test_outcomes_used_to_select_membership=False,
        ),
    )
    atomic_json(root / "status.json", dict(stage="intake_sealed", completed_arms=0, total_arms=168))


def assert_inputs(root):
    lock = read(root / "input-lock.json")
    validate_hashes(root, lock["hashes"])
    assert digest(root / "independent-schedules.npz") == read(root / "schedule-lock.json")["sha256"]
    return lock


def measure(model, optim, env, directory):
    """Observe fixed banks and support-only probes without updating learner or RNG."""
    settings, _, _, bank, warm, _, evaluation, _, _, _, reference, eval_warm, eval_ref = env
    directory.mkdir(parents=True, exist_ok=True)
    parent = fingerprint(capture(model, optim))
    selector = health_read(model, bank, warm, settings["assay_seed"])
    held = health_read(model, evaluation["query_images"], eval_warm, settings["assay_seed"])
    probe = linear_readout(model, evaluation, settings["reader_draws"], settings["assay_seed"])
    metrics = dict(
        selector_objective=float(objective(selector[0], reference, settings)),
        evaluation_objective=float(objective(held[0], eval_ref, settings)),
    )
    metrics.update({"selector_" + k: float(v) for k, v in zip(HEALTH, selector[0], strict=True)})
    metrics.update({"evaluation_" + k: float(v) for k, v in zip(HEALTH, held[0], strict=True)})
    metrics.update(probe["metrics"])
    finite_metrics(metrics)
    assert fingerprint(capture(model, optim)) == parent, "Observation mutated learner or RNG"
    np.savez_compressed(
        directory / "raw.npz",
        selector_values=selector[0],
        evaluation_values=held[0],
        selector_reconstruction=selector[2],
        evaluation_reconstruction=held[2],
        probabilities=probe["probabilities"],
    )
    result = dict(metrics=metrics, panels=probe["panels"], fingerprint=parent)
    atomic_json(directory / "readout.json", result)
    return result


def save_state(path, model, optim):
    temporary = path.with_suffix(".tmp")
    torch.save(capture(model, optim), temporary)
    temporary.replace(path)


def run_arm(
    root, directory, arm, spec, env, models, arrivals, history, seed, parent, initial_obs, schedule, stop, emit
):
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, _, _, _ = env
    start, length = settings["start"], stop - settings["start"]
    branch = directory / arm
    if (branch / "trajectory.json").exists():
        return read(branch / "trajectory.json")
    branch.mkdir(exist_ok=True)
    gate = spec["kind"] == "gate"
    costs = dict(features=0.0, prediction=0.0, observations=0.0, training=0.0)
    checkpoint_records = {str(start): read(directory / "common/readout.json")}
    accepted = np.zeros(length, bool)
    predictions, training = np.zeros(length), np.zeros((length, 2))
    vectors, outside = np.zeros((length, 86 if gate else 0)), np.zeros((length, 2), bool)
    done, research_seconds = 0, 0.0
    observations = [(x, None, None) for x in initial_obs]
    resumed = load_checkpoint(branch)
    if resumed is not None:
        generation, resume = resumed
        done, costs, checkpoint_records = resume["done"], resume["costs"], resume["checkpoints"]
        research_seconds = resume["research_seconds"]
        state = torch.load(generation / "model.pt", map_location="cpu", weights_only=True)
        assert fingerprint(state) == resume["fingerprint"]
        with np.load(generation / "trace.npz") as a:
            accepted[:], predictions[:], training[:], vectors[:], outside[:] = (
                a["accepted"],
                a["predictions"],
                a["training"],
                a["features"],
                a["outside"],
            )
            observations = [(x, None, None) for x in a["last_observations"]]
    else:
        state = parent
    model, optim = restore(initial, config.optim, state)
    task = None
    if gate:
        tick = time.perf_counter()
        current, _, task = make_observation(model, bank, warm, dev, settings, rich=True)
        observations[-1] = current
        costs["observations"] += time.perf_counter() - tick
    for t in range(start + done, stop):
        i = t - start
        if gate:
            tick = time.perf_counter()
            x = feature_vectors(
                model,
                optim,
                arrivals,
                t,
                observations[-4:],
                centroid,
                projection,
                clusters,
                settings,
                view_seed(settings["stream_namespace"], history, t, seed),
                task,
                rich=True,
            )["rich"]
            vectors[i] = x
            forecaster = models["J_rich"][seed]
            outside[i] = [
                np.any((x < forecaster.minimum) | (x > forecaster.maximum)),
                np.any((x[:14] < forecaster.minimum[:14]) | (x[:14] > forecaster.maximum[:14])),
            ]
            costs["features"] += time.perf_counter() - tick
            tick = time.perf_counter()
            predictions[i] = forecaster.predict(x)[0]
            accept = predictions[i] < spec["threshold"]
            costs["prediction"] += time.perf_counter() - tick
        else:
            accept = spec["kind"] == "full" or (schedule is not None and schedule[i])
        accepted[i] = accept
        tick = time.perf_counter()
        if accept:
            training[i] = seeded_step(
                model, optim, arrivals[t : t + 1].cuda(), view_seed(settings["stream_namespace"], history, t, seed)
            )
        costs["training"] += time.perf_counter() - tick
        if gate:
            tick = time.perf_counter()
            if accept:
                current, _, task = make_observation(model, bank, warm, dev, settings, rich=True)
            else:
                current = observations[-1]
            observations = [*observations[-3:], current]
            costs["observations"] += time.perf_counter() - tick
        if (t + 1 - start) % settings["checkpoint_stride"] == 0 or t + 1 == stop:
            check_storage(settings)
            tick = time.perf_counter()
            checkpoint_records[str(t + 1)] = measure(model, optim, env, branch / f"checkpoint_{t + 1}")
            research_seconds += time.perf_counter() - tick
            commit_checkpoint(
                branch,
                capture(model, optim),
                dict(
                    accepted=accepted,
                    predictions=predictions,
                    training=training,
                    features=vectors,
                    outside=outside,
                    last_observations=np.array([x[0] for x in observations]),
                ),
                dict(
                    done=i + 1,
                    costs=costs,
                    checkpoints=checkpoint_records,
                    research_seconds=research_seconds,
                    fingerprint=fingerprint(capture(model, optim)),
                ),
            )
            emit(
                stage="running",
                history=history,
                seed=seed,
                arm=arm,
                offered=i + 1,
                accepted=int(accepted.sum()),
                accuracy=checkpoint_records[str(t + 1)]["metrics"]["linear_accuracy_balanced"],
            )
    final = fingerprint(capture(model, optim))
    if spec["kind"] == "hold":
        assert final == fingerprint(parent)
    generation, _ = load_checkpoint(branch)
    shutil.copy2(generation / "model.pt", branch / "final_state.pt")
    shutil.copy2(generation / "trace.npz", branch / "trace.npz")
    record = dict(
        history=history,
        seed=seed,
        arm=arm,
        spec=spec,
        offered=length,
        accepted=int(accepted.sum()),
        costs=costs,
        policy_seconds=sum(costs.values()),
        research_seconds=research_seconds,
        parent_fingerprint=fingerprint(parent),
        final_fingerprint=final,
        checkpoints=checkpoint_records,
    )
    atomic_json(branch / "trajectory.json", record)
    for generation in list((branch / "resume_states").glob("step_*")):
        remove_generation(generation)
    return record


def verify_arm(branch, spec, schedule, env, forecaster):
    """Reconstruct all admissions and reject nonfinite traces before accepting an arm."""
    record = read(branch / "trajectory.json")
    with np.load(branch / "trace.npz") as a:
        accepted = a["accepted"]
        assert accepted.sum() == record["accepted"]
        assert np.isfinite(a["training"]).all() and np.isfinite(a["features"]).all()
        assert np.isfinite(a["predictions"]).all()
        if spec["kind"] == "gate":
            np.testing.assert_allclose(forecaster.predict(a["features"]), a["predictions"], rtol=1e-9, atol=1e-9)
            np.testing.assert_array_equal(accepted, a["predictions"] < spec["threshold"])
        elif schedule is not None:
            np.testing.assert_array_equal(accepted, schedule)
        else:
            assert np.all(accepted == (spec["kind"] == "full"))
    saved = torch.load(branch / "final_state.pt", map_location="cpu", weights_only=True)
    assert fingerprint(saved) == record["final_fingerprint"]
    settings, _, _, _, _, _, evaluation, _, _, _, reference, _, eval_ref = env
    for clock, checkpoint in record["checkpoints"].items():
        if int(clock) == settings["start"]:
            continue
        with np.load(branch / f"checkpoint_{clock}/raw.npz") as raw:
            np.testing.assert_allclose(
                objective(raw["selector_values"], reference, settings), checkpoint["metrics"]["selector_objective"]
            )
            np.testing.assert_allclose(
                objective(raw["evaluation_values"], eval_ref, settings), checkpoint["metrics"]["evaluation_objective"]
            )
            for p, probabilities in enumerate(raw["probabilities"]):
                recalculated = performance64(
                    dict(linear=probabilities[0], knn=probabilities[1]),
                    evaluation["query_labels"],
                    evaluation["query_domains"],
                )
                for metric, value in recalculated.items():
                    np.testing.assert_allclose(value, checkpoint["panels"][p][metric], rtol=1e-12, atol=1e-12)


def check_resume_equivalence(root, directory, env, models, arrivals, history, seed, parent, observations):
    """Interrupt a real gate after a commit and require exact source replay after recovery."""
    if (root / "smoke/resume-verification.json").exists():
        assert read(root / "smoke/resume-verification.json")["passed"]
        return
    threshold = read(ROOT / env[0]["source"] / "model-selection.json")["J_rich"]["threshold"]
    spec = dict(kind="gate", threshold=threshold)

    class DeliberateInterruption(Exception):
        pass

    def interrupt(**event):
        if event.get("offered") == 96:
            raise DeliberateInterruption

    try:
        run_arm(
            root,
            directory,
            "resume_probe",
            spec,
            env,
            models,
            arrivals,
            history,
            seed,
            parent,
            observations,
            None,
            env[0]["start"] + 192,
            interrupt,
        )
    except DeliberateInterruption:
        pass
    assert load_checkpoint(directory / "resume_probe")[1]["done"] == 96
    result = run_arm(
        root,
        directory,
        "resume_probe",
        spec,
        env,
        models,
        arrivals,
        history,
        seed,
        parent,
        observations,
        None,
        env[0]["start"] + 192,
        lambda **event: None,
    )
    source = ROOT / env[0]["source"] / "trajectories" / history / f"seed_{seed}/J_rich/trajectory.json"
    expected = read(source)
    assert result["final_fingerprint"] == expected["checkpoints"][str(env[0]["start"] + 192)]["fingerprint"]
    with np.load(directory / "resume_probe/trace.npz") as trace:
        for i, row in enumerate(expected["records"][:192]):
            assert bool(trace["accepted"][i]) == row["accepted"]
            np.testing.assert_allclose(trace["features"][i], row["features"], rtol=1e-9, atol=1e-9)
    atomic_json(
        root / "smoke/resume-verification.json",
        dict(passed=True, interrupted_after=96, recovered_to=192, final_model_optimizer_and_rng_match_source=True),
    )


def run(root, smoke=False):
    if smoke:
        selection = validate_plan(root)
        settings = dict(selection["settings"], checkpoint_stride=96)
        source = ROOT / settings["source"]
        lock = dict(
            settings=settings,
            histories=selection["original_histories"],
            specs=policy_specs(settings, read(source / "model-selection.json")["J_rich"]["threshold"]),
        )
        freeze_schedules(root, selection["settings"], lock["histories"])
    else:
        lock = assert_inputs(root)
        lock = dict(lock, specs=active_specs(root, lock["specs"]))
        lock = dict(
            lock,
            histories=capped_histories(
                root, lock["histories"], len(lock["specs"]), len(lock["settings"]["trajectory_seeds"])
            ),
        )
    if not smoke:
        assert read(root / "smoke/smoke-verification.json")["passed"], "GPU smoke must pass before full launch"
    source = ROOT / lock["settings"]["source"]
    env = environment(source)
    env[0].update(lock["settings"])
    settings = env[0]
    models = joblib.load(source / "forecasters.joblib")
    independent = dict(np.load(root / "independent-schedules.npz"))
    work = root / "smoke" if smoke else root
    work.mkdir(exist_ok=True)
    began, roster = time.perf_counter(), []
    total = len(lock["specs"]) * len(lock["histories"]) * len(settings["trajectory_seeds"])
    status = dict(pid=os.getpid(), smoke=smoke, total_arms=(14 if smoke else total))

    def emit(**kwargs):
        status.update(
            kwargs,
            elapsed_seconds=time.perf_counter() - began,
            completed_arms=len(roster),
            free_gib=shutil.disk_usage(ROOT).free / 2**30,
        )
        atomic_json(work / "status.json", status)
        print(json.dumps(status), flush=True)

    histories = list(lock["histories"].items())[:1] if smoke else list(lock["histories"].items())
    seeds = settings["trajectory_seeds"][:1] if smoke else settings["trajectory_seeds"]
    stop = settings["start"] + (96 if smoke else settings["offered"])
    for history, meta in histories:
        arrivals = FrameStream(meta["paths"])
        for seed in seeds:
            directory = work / "trajectories" / history / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            original = source / "trajectories" / history / f"seed_{seed}"
            parent = torch.load(original / "parent.pt", map_location="cpu", weights_only=True)
            with np.load(original / "parent-observations.npz") as a:
                observations = a["values"]
            if not (directory / "common/readout.json").exists():
                model, optim = restore(env[2], env[1].optim, parent)
                common = measure(model, optim, env, directory / "common")
                previous = read(original / "common/readout.json")
                for key, value in common["metrics"].items():
                    np.testing.assert_allclose(value, previous["metrics"][key], atol=1e-10, rtol=1e-10)
                del model, optim
            if smoke:
                check_resume_equivalence(root, directory, env, models, arrivals, history, seed, parent, observations)
            for arm, spec in lock["specs"].items():
                schedule = None
                kind = spec["kind"]
                length = stop - settings["start"]
                if kind == "independent":
                    schedule = independent[f"{history}__{seed}__{arm}"][:length]
                elif kind in ("global", "clip"):
                    with np.load(directory / spec["gate"] / "trace.npz") as a:
                        accepted = a["accepted"]
                    namespace = settings[kind + "_namespace"] + spec["gate"]
                    draw = view_seed(namespace, history, settings["start"], seed)
                    schedule = (
                        global_schedule(accepted, draw)
                        if kind == "global"
                        else matched_random(accepted, np.arange(settings["start"], stop), 96, draw)
                    )
                record = run_arm(
                    root,
                    directory,
                    arm,
                    spec,
                    env,
                    models,
                    arrivals,
                    history,
                    seed,
                    parent,
                    observations,
                    schedule,
                    stop,
                    emit,
                )
                verify_arm(directory / arm, spec, schedule, env, models["J_rich"][seed])
                if smoke and arm in ("gate_original", "full"):
                    old_arm = "J_rich" if arm == "gate_original" else "full"
                    old = read(original / old_arm / "trajectory.json")
                    with np.load(directory / arm / "trace.npz") as a:
                        for i, previous in enumerate(old["records"][:length]):
                            assert bool(a["accepted"][i]) == previous["accepted"]
                            if arm == "gate_original":
                                np.testing.assert_allclose(a["features"][i], previous["features"], rtol=1e-9, atol=1e-9)
                            if previous["accepted"]:
                                np.testing.assert_allclose(a["training"][i], previous["training"], rtol=1e-7, atol=1e-8)
                    assert (
                        record["checkpoints"][str(stop)]["fingerprint"] == old["checkpoints"][str(stop)]["fingerprint"]
                    )
                roster.append(dict(history=history, seed=seed, arm=arm, path=str((directory / arm).relative_to(work))))
                atomic_json(work / "collection.json", dict(records=roster))
                emit(stage="arm_complete", history=history, seed=seed, arm=arm)
            if not smoke:
                from scripts.report_single_image_extended_gate import report

                report(root, partial=True)
        del arrivals
    if smoke:
        smoke_lock = dict(lock)
        smoke_lock["settings"] = dict(lock["settings"], offered=96)
        atomic_json(work / "input-lock.json", smoke_lock)
        from scripts.report_single_image_extended_gate import report

        report(work, partial=True)
        atomic_json(
            work / "smoke-verification.json",
            dict(
                passed=True,
                arms=len(roster),
                original_prefix_reproduced=True,
                full_prefix_reproduced=True,
                observers_immutable=True,
                independent_schedules_frozen=True,
            ),
        )
    else:
        assert_inputs(root)
        atomic_json(
            root / "collection-verification.json",
            dict(
                passed=True,
                arms=len(roster),
                retained_trajectory_target=total,
                independent_schedule_sha256=digest(root / "independent-schedules.npz"),
                scientific_data_and_retained_policy_settings_unchanged=True,
                execution_amendment_sha256=(
                    digest(root / "execution-amendment.json") if (root / "execution-amendment.json").exists() else None
                ),
            ),
        )
        from scripts.report_single_image_extended_gate import report

        report(root, partial=False)
    emit(stage="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "prepare", "smoke", "run", "pipeline"), required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase in ("plan", "pipeline"):
        plan(root)
        selection = validate_plan(root)
        freeze_schedules(root, selection["settings"], selection["original_histories"])
        if args.phase == "plan":
            return
    if args.phase == "prepare":
        prepare(root, OmegaConf.to_container(OmegaConf.load(CONFIG)))
        return
    setup()
    try:
        with threadpool_limits(limits=1):
            if args.phase == "pipeline":
                if not (root / "smoke/smoke-verification.json").exists():
                    run(root, smoke=True)
                assert read(root / "smoke/smoke-verification.json")["passed"]
                acquire(root)
                prepare(root, OmegaConf.to_container(OmegaConf.load(CONFIG)))
                run(root)
            else:
                run(root, smoke=args.phase == "smoke")
    except Exception as error:
        target = root / "smoke" if args.phase == "smoke" else root
        target.mkdir(exist_ok=True)
        atomic_json(target / "failure.json", dict(pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
