"""Collect, fit and run the frozen singleton objective-by-feature experiment."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_filter import matched_random, objective
from cafl4ds.single_image_filter_factorial import (
    ARMS,
    MAIN,
    feature_vectors,
    fit_factorial,
    linear_readout,
    reference_gradient,
    update_features,
)
from cafl4ds.single_image_forecast import HEALTH, encode, fine_tune
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import performance64, seeded_step
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_nonlinear import health_read
from scripts.run_single_image_signal import fresh, load_stream, read

ROOT = Path(__file__).resolve().parents[1]
SIGNAL = ROOT / "outputs/single-image-signal/20261021-followup"
PILOT = ROOT / "outputs/single-image-filter/20261021-aggregate-pilot"
CONFIG = ROOT / "cafl4ds/configs/single_image_filter_factorial.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-filter-factorial-frozen.md"


def reset_rng(parent):
    torch.set_rng_state(parent["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([r.cpu() for r in parent["cuda_rng"]])


def panel_at(root, role):
    p = torch.load(root / "intake" / f"{role}-panel.pt", map_location="cpu", weights_only=True)
    for key in ("query_labels", "query_domains"):
        p[key] = p[key].numpy()
    return p


def environment(root):
    settings = OmegaConf.to_container(OmegaConf.load(CONFIG))
    config = OmegaConf.create(read(SIGNAL / "input_lock.json")["learner_config"])
    initial = _method(config).cuda()
    bank = torch.load(SIGNAL / "bank.pt", weights_only=True)
    warm = np.load(SIGNAL / "warm_bank.npy")
    development, evaluation = panel_at(root, "development"), panel_at(root, "evaluation")
    projection = np.random.default_rng(settings["projection_seed"]).normal(size=(warm.shape[1], 8)) / np.sqrt(
        warm.shape[1]
    )
    normalized = warm.astype(float) / np.maximum(np.linalg.norm(warm.astype(float), axis=1, keepdims=True), 1e-12)
    centroid = normalized.mean(0)
    clusters = KMeans(n_clusters=settings["clusters"], n_init=10, random_state=settings["projection_seed"]).fit_predict(
        normalized
    )
    reference = health_read(initial, bank, warm, settings["assay_seed"])[0]
    eval_warm = encode(initial, evaluation["query_images"], settings["assay_seed"])
    eval_ref = health_read(initial, evaluation["query_images"], eval_warm, settings["assay_seed"])[0]
    return (
        settings,
        config,
        initial,
        bank,
        warm,
        development,
        evaluation,
        projection,
        centroid,
        clusters,
        reference,
        eval_warm,
        eval_ref,
    )


def make_observation(model, bank, warm, development, settings, rich=False):
    health = health_read(model, bank, warm, settings["assay_seed"])
    if not rich:
        return health, None, None
    probe = linear_readout(model, development, settings["reader_draws"], settings["assay_seed"])
    gradient = reference_gradient(model, development, probe["heads"], settings["assay_seed"])
    return health, probe, gradient


def sources():
    return [
        CONFIG,
        PROTOCOL,
        Path(__file__).resolve(),
        ROOT / "cafl4ds/single_image_filter_factorial.py",
        ROOT / "scripts/prepare_single_image_filter_factorial.py",
        ROOT / "cafl4ds/single_image_filter.py",
        ROOT / "cafl4ds/single_image_forecast.py",
        ROOT / "cafl4ds/single_image_signal.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "scripts/run_single_image_nonlinear.py",
        ROOT / "scripts/run_single_image_signal.py",
        ROOT / "cafl4ds/ssl/mae.py",
        ROOT / "cafl4ds/ssl/augment.py",
        ROOT / "cafl4ds/models/vit.py",
    ]


def lock_inputs(root, settings):
    assert not (root / "input_lock.json").exists(), "Do not overwrite scientific collection"
    previous = list(
        dict.fromkeys(
            read(PILOT / "input_lock.json")["prior_studies"]
            + [str(PILOT), str(ROOT / "outputs/single-image-filter-review/20261021-objective-metrics")]
        )
    )
    prior_files = 0
    for path in previous:
        directory = Path(path)
        for name, expected in read(directory / "artifact_hashes.json")["sha256"].items():
            assert digest(directory / name) == expected, str(directory / name)
            prior_files += 1
    config = read(SIGNAL / "input_lock.json")["learner_config"]
    paths = sources() + [
        root / "intake" / name for name in ["selection.json", "development-panel.pt", "evaluation-panel.pt"]
    ]
    paths += [
        SIGNAL / "input_lock.json",
        SIGNAL / "bank.pt",
        SIGNAL / "warm_bank.npy",
        SIGNAL / "panel.pt",
        Path(config["well"]),
    ]
    paths += [Path(p) / "artifact_hashes.json" for p in previous]
    atomic_json(
        root / "input_lock.json",
        dict(
            settings=settings,
            hashes={str(p): digest(p) for p in paths},
            prior_studies=previous,
            prior_files_verified=prior_files,
            task_informed_live_features=True,
            evaluation_never_gate_input=True,
            exploratory=True,
        ),
    )
    for p in sources():
        destination = root / "source_snapshot" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)


def assert_inputs(root):
    for name, expected in read(root / "input_lock.json")["hashes"].items():
        assert digest(Path(name)) == expected, name


def preflight(root, env):
    settings, config, initial, bank, warm, dev, _, _, _, _, _, _, _ = env
    meta = next(v for k, v in read(SIGNAL / "input_lock.json")["histories"].items() if k.startswith("held_"))
    images = load_stream(meta)
    model, optim = fresh(initial, config, 5003)
    for t in range(3):
        seeded_step(model, optim, images[t : t + 1].cuda(), 86101 + t)
    parent = capture(model, optim)
    probe = linear_readout(model, dev, settings["reader_draws"], settings["assay_seed"])
    gradient = reference_gradient(model, dev, probe["heads"], settings["assay_seed"])
    paid, delta = update_features(model, optim, images[3:4], 86109, gradient)
    assert fingerprint(capture(model, optim)) == fingerprint(parent)
    before = torch.cat([p.detach().flatten().clone() for p in model.parameters() if p.requires_grad])
    train = seeded_step(model, optim, images[3:4].cuda(), 86109)
    actual = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad]) - before
    torch.testing.assert_close(delta, actual, rtol=2e-4, atol=1.2e-7)
    assert train[0] == paid[0]
    atomic_json(
        root / "preflight.json",
        dict(
            passed=True,
            discarded_updates=4,
            proposed_actual_max_error=float((delta - actual).abs().max()),
            exact_training_view=True,
            feature_observer_state_unchanged=True,
        ),
    )


def calibration(root, env, progress):
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, reference, _, _ = env
    lock_inputs(root, settings)
    np.savez_compressed(
        root / "references.npz", selector=reference, clusters=clusters, projection=projection, centroid=centroid
    )
    preflight(root, env)
    origin_settings = settings["development_origins"]
    origins = set(range(origin_settings["start"], origin_settings["stop"], origin_settings["stride"]))
    clocks = {t - lag for t in origins for lag in range(4)}
    rows = []
    histories = {k: v for k, v in read(SIGNAL / "input_lock.json")["histories"].items() if k.startswith("held_")}
    arrays = {k: [] for k in ["state", "base", "rich", "targets", "histories", "seeds", "origins"]}
    for history, meta in histories.items():
        arrivals = load_stream(meta)
        for seed in settings["trajectory_seeds"]:
            model, optim = fresh(initial, config, seed)
            observations, training = {}, []
            directory = root / "calibration" / history / f"seed_{seed}"
            directory.mkdir(parents=True)
            for t in range(max(origins) + 1):
                if t in clocks:
                    observations[t] = health_read(model, bank, warm, settings["assay_seed"])
                if t in origins:
                    parent = capture(model, optim)
                    before_probe = linear_readout(model, dev, settings["reader_draws"], settings["assay_seed"])
                    task_gradient = reference_gradient(model, dev, before_probe["heads"], settings["assay_seed"])
                    draw = view_seed(settings["branch_namespace"], history, t, seed * 10)
                    vectors = feature_vectors(
                        model,
                        optim,
                        arrivals,
                        t,
                        [observations[t - l] for l in (3, 2, 1, 0)],
                        centroid,
                        projection,
                        clusters,
                        settings,
                        draw,
                        task_gradient,
                        rich=True,
                    )
                    assert fingerprint(capture(model, optim)) == fingerprint(parent)
                    after_health, after_probs, targets, fingerprints, traces, draws = [], [], [], [], [], []
                    for view in range(settings["development_views"]):
                        branch, bo = restore(initial, config.optim, parent)
                        branch_seed = view_seed(settings["branch_namespace"], history, t, seed * 10 + view)
                        train = seeded_step(branch, bo, arrivals[t : t + 1].cuda(), branch_seed)
                        obs = health_read(branch, bank, warm, settings["assay_seed"])
                        after_probe = linear_readout(branch, dev, settings["reader_draws"], settings["assay_seed"])
                        after_health.append(obs[0])
                        after_probs.append(after_probe["probabilities"])
                        targets.append(
                            [
                                float(
                                    objective(obs[0], reference, settings)
                                    - objective(observations[t][0], reference, settings)
                                ),
                                after_probe["metrics"]["linear_ce_balanced"]
                                - before_probe["metrics"]["linear_ce_balanced"],
                            ]
                        )
                        fingerprints.append(fingerprint(capture(branch, bo)))
                        traces.append(train)
                        draws.append(branch_seed)
                        del branch, bo
                    reset_rng(parent)
                    assert fingerprint(capture(model, optim)) == fingerprint(parent)
                    raw = directory / f"{t}.npz"
                    np.savez_compressed(
                        raw,
                        **vectors,
                        targets=np.asarray(targets),
                        before=observations[t][0],
                        after=np.asarray(after_health),
                        before_probabilities=before_probe["probabilities"],
                        after_probabilities=np.asarray(after_probs),
                        heads=before_probe["heads"],
                        draws=np.asarray(draws),
                        training=np.asarray(traces),
                    )
                    row = dict(
                        history=history,
                        seed=seed,
                        origin=t,
                        raw=str(raw.relative_to(root)),
                        parent_fingerprint=fingerprint(parent),
                        branch_fingerprints=fingerprints,
                    )
                    rows.append(row)
                    for k in ("state", "base", "rich"):
                        arrays[k].append(vectors[k])
                    for k, value in dict(
                        targets=np.mean(targets, axis=0), histories=history, seeds=seed, origins=t
                    ).items():
                        arrays[k].append(value)
                    atomic_json(root / "calibration-collection.json", dict(records=rows))
                    progress(stage="calibration", history=history, seed=seed, origin=t, parents=len(rows))
                training.append(
                    seeded_step(
                        model,
                        optim,
                        arrivals[t : t + 1].cuda(),
                        view_seed(settings["stream_namespace"], history, t, seed),
                    )
                )
            np.save(directory / "training.npy", np.asarray(training))
            torch.save(capture(model, optim), directory / "final_state.pt")
            del model, optim, observations
        del arrivals
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(root / "calibration.npz", **arrays)
    models, selection = fit_factorial(arrays, settings)
    joblib.dump(models, root / "forecasters.joblib")
    atomic_json(root / "model-selection.json", selection)
    atomic_json(
        root / "calibration-verification.json",
        dict(
            passed=True,
            parents=len(rows),
            development_views=settings["development_views"],
            actual_branches=len(rows) * settings["development_views"],
            target_names=["J", "probe_CE"],
            feature_dimensions={k: arrays[k].shape[1] for k in ("state", "base", "rich")},
            no_test_outcome_used=True,
            frozen_forecasters=True,
        ),
    )
    assert_inputs(root)
    progress(stage="calibration_complete", parents=len(rows))


def evaluation_read(model, optim, env, directory, final=False):
    settings, _, _, bank, warm, dev, evaluation, _, _, _, reference, eval_warm, eval_ref = env
    directory.mkdir(parents=True, exist_ok=False)
    parent = fingerprint(capture(model, optim))
    selector = health_read(model, bank, warm, settings["assay_seed"])
    held = health_read(model, evaluation["query_images"], eval_warm, settings["assay_seed"])
    probe = linear_readout(model, evaluation, settings["reader_draws"], settings["assay_seed"])
    development = linear_readout(model, dev, settings["reader_draws"], settings["assay_seed"])
    metrics = dict(
        selector_objective=float(objective(selector[0], reference, settings)),
        evaluation_objective=float(objective(held[0], eval_ref, settings)),
        development_probe_ce=development["metrics"]["linear_ce_balanced"],
    )
    metrics.update({"selector_" + k: float(v) for k, v in zip(HEALTH, selector[0], strict=True)})
    metrics.update({"evaluation_" + k: float(v) for k, v in zip(HEALTH, held[0], strict=True)})
    metrics.update(probe["metrics"])
    panels = probe["panels"]
    ft = []
    if final:
        for p, draw in enumerate(settings["reader_draws"]):
            probabilities, _ = fine_tune(
                model,
                evaluation["support_images"][draw],
                evaluation["support_labels"][draw],
                evaluation["query_images"],
                settings["reader_steps"],
                draw,
            )
            ft.append(probabilities)
            panels[p].update(
                performance64(dict(finetune100=probabilities), evaluation["query_labels"], evaluation["query_domains"])
            )
        metrics.update({k: float(np.mean([r[k] for r in panels])) for k in panels[0] if k.startswith("finetune")})
    assert fingerprint(capture(model, optim)) == parent
    np.savez_compressed(
        directory / "raw.npz",
        selector_values=selector[0],
        selector_features=selector[1],
        selector_reconstruction=selector[2],
        evaluation_values=held[0],
        evaluation_features=held[1],
        evaluation_reconstruction=held[2],
        probabilities=probe["probabilities"],
        development_probabilities=development["probabilities"],
        finetune_probabilities=np.asarray(ft),
    )
    record = dict(metrics=metrics, panels=panels, fingerprint=parent, fine_tuning_measured=final)
    atomic_json(directory / "readout.json", record)
    return record


def diagnostic_positions(settings, history, seed):
    positions = []
    for clip in range(settings["start"] // 96, settings["stop"] // 96):
        rng = np.random.default_rng(view_seed(settings["audit_namespace"], history, clip, seed))
        positions.append(clip * 96 + int(rng.integers(0, 65)))
    return positions


def diagnostic(root, model, optim, arrivals, origin, history, seed, env, directory, row):
    settings, config, initial, bank, warm, dev, evaluation, _, _, _, reference, _, _ = env
    parent = capture(model, optim)

    def read_branch(branch):
        h = health_read(branch, bank, warm, settings["assay_seed"])
        d = linear_readout(branch, dev, settings["reader_draws"], settings["assay_seed"])
        e = linear_readout(branch, evaluation, settings["reader_draws"], settings["assay_seed"])
        return dict(
            health=h[0],
            selector_features=h[1],
            selector_reconstruction=h[2],
            development_probabilities=d["probabilities"],
            evaluation_probabilities=e["probabilities"],
        ), dict(
            J=float(objective(h[0], reference, settings)),
            probe=d["metrics"]["linear_ce_balanced"],
            evaluation_probe_ce=e["metrics"]["linear_ce_balanced"],
            evaluation_probe_accuracy=e["metrics"]["linear_accuracy_balanced"],
        )

    raw, measures, fingerprints, training = {}, {}, {}, {}
    observed, measures["before"] = read_branch(model)
    raw.update({"before_" + k: v for k, v in observed.items()})
    for accept, name in ((True, "accept"), (False, "skip")):
        branch, bo = restore(initial, config.optim, parent)
        trace = []
        if accept:
            trace.append(
                seeded_step(
                    branch,
                    bo,
                    arrivals[origin : origin + 1].cuda(),
                    view_seed(settings["stream_namespace"], history, origin, seed),
                )
            )
            observed, measures["H1"] = read_branch(branch)
            raw.update({"H1_" + k: v for k, v in observed.items()})
            fingerprints["H1"] = fingerprint(capture(branch, bo))
        for t in range(origin + 1, origin + settings["continuation_horizon"]):
            trace.append(
                seeded_step(
                    branch, bo, arrivals[t : t + 1].cuda(), view_seed(settings["stream_namespace"], history, t, seed)
                )
            )
        observed, measures["H32_" + name] = read_branch(branch)
        raw.update({"H32_" + name + "_" + k: v for k, v in observed.items()})
        fingerprints["H32_" + name] = fingerprint(capture(branch, bo))
        training[name] = trace
        del branch, bo
    reset_rng(parent)
    assert fingerprint(capture(model, optim)) == fingerprint(parent)
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / f"{origin}.npz", **raw)
    record = dict(
        origin=origin,
        accepted=row["accepted"],
        prediction=row["prediction"],
        predictions=row["predictions"],
        parent_fingerprint=fingerprint(parent),
        branch_fingerprints=fingerprints,
        measures=measures,
        training=training,
    )
    atomic_json(directory / f"{origin}.json", record)
    return record


def run_policies(root, env, progress):
    assert_inputs(root)
    settings, config, initial, bank, warm, dev, _, projection, centroid, clusters, reference, _, _ = env
    intake = read(root / "intake/intake.json")
    assert intake["complete"]
    atomic_json(
        root / "policy-lock.json",
        dict(
            intake_sha256=digest(root / "intake/intake.json"),
            forecaster_sha256=digest(root / "forecasters.joblib"),
            selection_sha256=digest(root / "model-selection.json"),
        ),
    )
    models, selection = joblib.load(root / "forecasters.joblib"), read(root / "model-selection.json")
    roster = []
    for history, meta in intake["histories"].items():
        arrivals = load_stream(meta)
        for seed in settings["trajectory_seeds"]:
            directory = root / "trajectories" / history / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            model, optim = fresh(initial, config, seed)
            observations, prefix_training = [], []
            for t in range(settings["start"] + 1):
                if t >= settings["start"] - 3:
                    observations.append(health_read(model, bank, warm, settings["assay_seed"]))
                if t == settings["start"]:
                    break
                prefix_training.append(
                    seeded_step(
                        model,
                        optim,
                        arrivals[t : t + 1].cuda(),
                        view_seed(settings["stream_namespace"], history, t, seed),
                    )
                )
            parent = capture(model, optim)
            torch.save(parent, directory / "parent.pt")
            np.save(directory / "prefix-training.npy", np.asarray(prefix_training))
            np.savez_compressed(directory / "parent-observations.npz", values=np.asarray([r[0] for r in observations]))
            common = evaluation_read(model, optim, env, directory / "common", final=True)
            del model, optim
            schedules = {}
            for arm in ARMS:
                branch_dir = directory / arm
                branch_dir.mkdir()
                model, optim = restore(initial, config.optim, parent)
                is_gate = arm not in ("full", "hold") and not arm.startswith("random_")
                design = arm.split("_")[1] if is_gate else None
                rich = design == "rich"
                current_obs = list(observations)
                costs = dict(features=0.0, prediction=0.0, observations=0.0, training=0.0)
                research_seconds = 0.0
                rows, diagnostics, checkpoints = [], [], {str(settings["start"]): common}
                tick = time.perf_counter()
                task = None
                if rich:
                    current, _, task = make_observation(model, bank, warm, dev, settings, rich=True)
                    current_obs[-1] = current
                costs["observations"] += time.perf_counter() - tick
                sample_positions = diagnostic_positions(settings, history, seed) if arm in MAIN else []
                for t in range(settings["start"], settings["stop"]):
                    row = dict(origin=t)
                    if is_gate:
                        tick = time.perf_counter()
                        vectors = feature_vectors(
                            model,
                            optim,
                            arrivals,
                            t,
                            current_obs[-4:],
                            centroid,
                            projection,
                            clusters,
                            settings,
                            view_seed(settings["stream_namespace"], history, t, seed),
                            task,
                            rich=rich,
                        )
                        x = vectors[design]
                        costs["features"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                        predictions = {
                            target: float(models[target + "_" + design][seed].predict(x)[0])
                            for target in ("J", "probe")
                        }
                        predicted = predictions[arm.split("_")[0]]
                        threshold = selection[arm]["threshold"]
                        accept = predicted < threshold
                        costs["prediction"] += time.perf_counter() - tick
                        row.update(
                            features=x.tolist(), predictions=predictions, prediction=predicted, threshold=threshold
                        )
                    else:
                        accept = arm == "full" or (
                            arm.startswith("random_")
                            and bool(schedules[arm.removeprefix("random_")][t - settings["start"]])
                        )
                    row["accepted"] = bool(accept)
                    if t in sample_positions:
                        tick = time.perf_counter()
                        diagnostics.append(
                            diagnostic(
                                root, model, optim, arrivals, t, history, seed, env, branch_dir / "diagnostics", row
                            )
                        )
                        research_seconds += time.perf_counter() - tick
                    tick = time.perf_counter()
                    train = (
                        seeded_step(
                            model,
                            optim,
                            arrivals[t : t + 1].cuda(),
                            view_seed(settings["stream_namespace"], history, t, seed),
                        )
                        if accept
                        else None
                    )
                    costs["training"] += time.perf_counter() - tick
                    row["training"] = None if train is None else list(train)
                    if is_gate:
                        tick = time.perf_counter()
                        if accept:
                            current, _, task = make_observation(model, bank, warm, dev, settings, rich=rich)
                        else:
                            current = current_obs[-1]
                        current_obs.append(current)
                        current_obs = current_obs[-4:]
                        costs["observations"] += time.perf_counter() - tick
                    rows.append(row)
                    if (t + 1) % settings["checkpoint_stride"] == 0:
                        tick = time.perf_counter()
                        checkpoints[str(t + 1)] = evaluation_read(
                            model, optim, env, branch_dir / f"checkpoint_{t + 1}", final=t + 1 == settings["stop"]
                        )
                        research_seconds += time.perf_counter() - tick
                        progress(
                            stage="policies",
                            history=history,
                            seed=seed,
                            arm=arm,
                            origin=t + 1,
                            accepted=sum(r["accepted"] for r in rows),
                            completed_arms=len(roster),
                        )
                final = capture(model, optim)
                if arm == "hold":
                    assert fingerprint(final) == fingerprint(parent)
                torch.save(final, branch_dir / "final_state.pt")
                record = dict(
                    records=rows,
                    diagnostics=diagnostics,
                    checkpoints=checkpoints,
                    costs=costs,
                    policy_seconds=sum(costs.values()),
                    research_seconds=research_seconds,
                    parent_fingerprint=fingerprint(parent),
                    final_fingerprint=fingerprint(final),
                    accepted=sum(r["accepted"] for r in rows),
                )
                atomic_json(branch_dir / "trajectory.json", record)
                if arm in MAIN:
                    schedules[arm] = matched_random(
                        np.array([r["accepted"] for r in rows]),
                        np.arange(settings["start"], settings["stop"]),
                        settings["frames_per_clip"],
                        view_seed(settings["random_namespace"] + "-" + arm, history, settings["start"], seed),
                    )
                roster.append(
                    dict(
                        history=history,
                        seed=seed,
                        arm=arm,
                        path=str(branch_dir.relative_to(root)),
                        accepted=record["accepted"],
                    )
                )
                atomic_json(root / "collection.json", dict(records=roster))
                del model, optim, final, task, current_obs
            del parent, observations
        del arrivals
    assert_inputs(root)
    atomic_json(
        root / "collection-verification.json",
        dict(
            passed=True,
            histories=6,
            seeds=2,
            arms=len(ARMS),
            trajectories=len(roster),
            batch_size=1,
            gradient_accumulation=False,
            policy_updates=sum(r["accepted"] for r in roster),
        ),
    )
    progress(stage="collection_complete", completed_arms=len(roster))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["calibration", "policies"], required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    setup()
    started = time.perf_counter()

    def progress(**kwargs):
        record = dict(seconds=time.perf_counter() - started, pid=os.getpid(), **kwargs)
        atomic_json(root / "status.json", record)
        print(record, flush=True)

    with threadpool_limits(limits=1):
        env = environment(root)
        if args.phase == "calibration":
            calibration(root, env, progress)
        else:
            run_policies(root, env, progress)


if __name__ == "__main__":
    main()
