"""Matched-budget accepted-batch coverage admission and exact source/learner replay."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import json
import shutil
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.return_actions import run_branch
from cafl4ds.short_buffer import capture, restore
from cafl4ds.coverage_admission import ARMS, CHECKPOINTS, run_admission
from cafl4ds.stopping_calibration import BUDGETS, SEEDS, DRAWS, METRICS, READERS, interval
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import evaluate_gpu
from scripts.run_short_buffer_experiment import digest
from scripts.run_scheduling_experiment import scores
from scripts.run_stopping_calibration import tensor

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/scheduling/20261013-gpu"
FRAMES = ROOT / "outputs/coverage-admission/20261013-frames"
INTAKE = ROOT / "outputs/coverage-admission/20261013-intake"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(out):
    previous = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "independent-verification.json")["passed"] or any(
        digest(Path(p)) != h for p, h in previous["hashes"].items()
    ):
        raise ValueError("Previous audited inputs changed")
    old_intake = read(ROOT / "outputs/scheduling/20261013-frames/intake.json")
    fresh = read(FRAMES / "intake.json")
    if not fresh["complete"] or len(fresh["clips"]) != 144:
        raise ValueError("Fresh144-clip intake incomplete")
    dev = [
        {**s, "name": "dev_" + s["name"], "old_name": s["name"]}
        for s in old_intake["streams"]
        if s["role"] == "development"
    ]
    streams = dev + fresh["streams"]
    clips = {c["clip_id"]: c for c in old_intake["clips"] + fresh["clips"]}
    ancestors = ROOT / "outputs/regime-windows/20261013-gpu"
    paths = [
        PRIOR / "input_lock.json",
        PRIOR / "independent-verification.json",
        PRIOR / "collection.json",
        ROOT / "outputs/scheduling/20261013-frames/intake.json",
        FRAMES / "intake.json",
        FRAMES / "allocation.json",
        INTAKE / "selection.json",
        INTAKE / "selection.pool.json",
        INTAKE / "download_manifest.json",
        ROOT / ".cache/stopping-known-videos.txt",
        ROOT / "docs/project-plan/slow-edge-coverage-admission-frozen.md",
        Path(__file__),
        ROOT / "scripts/verify_coverage_admission.py",
        ROOT / "scripts/select_coverage_clips.py",
        ROOT / "scripts/prepare_coverage_frames.py",
        ROOT / "scripts/predecode_coverage_frames.py",
        ROOT / "cafl4ds/coverage_admission.py",
    ]
    for c in fresh["clips"]:
        cache = Path(c["frames_path"])
        paths.extend(
            (
                cache,
                cache.parent / "manifest.json",
                cache.parent / "decoder.log",
                Path(c["video_path"]),
                Path(c["keyframe_path"]).parents[3] / "labels/train" / f"{c['clip_id']}.json",
            )
        )
    paths.append(ROOT / "cafl4ds/stopping_calibration.py")
    paths.extend(INTAKE.glob("amendment-*/*.json"))
    for reason in INTAKE.glob("amendment-*/reason.json"):
        identifier = read(reason)["failed_clip"]
        paths.extend((FRAMES / identifier / "decoder.log", INTAKE / "videos/train" / f"{identifier}.mov"))
    paths.extend(FRAMES.glob("*/predecode-failure.json"))
    for seed in SEEDS:
        paths.extend(PRIOR / "warm" / f"seed_{seed}" / p for p in ("state.pt", "readout.json", "predictions.npz"))
        for s in dev:
            base = PRIOR / "runs" / s["old_name"] / "1111" / f"seed_{seed}"
            paths.append(base / "training.json")
            for k in BUDGETS[2:]:
                paths.extend(base / f"at_{32 * k}" / p for p in ("state.pt", "readout.json", "predictions.npz"))
    image = ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt"
    with np.load(ancestors / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            torch.load(image, map_location="cpu", weights_only=True),
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in DRAWS},
            read(ancestors / "data-role-manifest.json"),
        )
    lock = json.loads(
        json.dumps(
            {
                "hashes": {**previous["hashes"], **{str(p): digest(p) for p in paths}},
                "config": previous["config"],
                "streams": streams,
                "budgets": BUDGETS,
                "scientific_new_ssl_updates": 29952,
                "new_primary_fits": 738,
                "reused_primary_fits": 81,
            }
        )
    )
    if (out / "input_lock.json").exists():
        raise ValueError("Use immutable evidence; no automatic scientific retry")
    atomic_json(out / "input_lock.json", lock)
    for p in paths:
        if p.suffix in (".py", ".md") and p.is_relative_to(ROOT):
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    return OmegaConf.create(previous["config"]), data, streams, clips, lock


def report(out, rows, labels, domains):
    fresh = [r for r in rows if r["role"] == "heldout"]
    names = sorted({r["stream"] for r in fresh})
    values = {(r["stream"], r["seed"], r["arm"]): scores(out, r["endpoints"][-1], labels, domains) for r in fresh}

    def metric(n, arm, reader, m):
        return statistics.fmean(values[n, seed, arm][reader][m] for seed in SEEDS)

    effects = {
        ref: {
            reader: {
                level: {
                    m: interval(
                        [metric(n, "coverage", reader, m) - metric(n, ref, reader, m) for n in names],
                        12 if level == "adjusted" else 1,
                    )
                    for m in METRICS
                }
                for level in ("ordinary", "adjusted")
            }
            for reader in READERS
        }
        for ref in ("uniform", "random", "early_stop", "full")
    }
    means = {
        arm: {
            reader: {m: statistics.fmean(metric(n, arm, reader, m) for n in names) for m in METRICS}
            for reader in READERS
        }
        for arm in (*ARMS, "full")
    }
    warm_scores = {seed: scores(out, f"warm/seed_{seed}", labels, domains) for seed in SEEDS}
    warm = {
        reader: {m: statistics.fmean(warm_scores[seed][reader][m] for seed in SEEDS) for m in METRICS}
        for reader in READERS
    }
    frozen_gain = interval(
        [metric(n, "coverage", "finetune", "balanced") - warm["finetune"]["balanced"] for n in names]
    )
    costs = {
        arm: statistics.fmean(r["processing_seconds"] for r in fresh if r["arm"] == arm) for arm in (*ARMS, "full")
    }
    saving = 1 - costs["coverage"] / costs["full"]
    superior = all(
        effects[ref]["finetune"]["adjusted"]["balanced"]["mean_pp"] >= 0.5
        and effects[ref]["finetune"]["adjusted"]["balanced"]["ci95_pp"][0] > 0
        for ref in ("uniform", "random")
    )
    guards = all(
        effects[ref]["finetune"]["adjusted"][m]["ci95_pp"][0] > -1 for ref in ("uniform", "random") for m in ("A", "B")
    ) and all(
        effects[ref]["finetune"]["adjusted"][m]["ci95_pp"][0] > -1 for ref in ("early_stop", "full") for m in METRICS
    )
    gates = {
        "beats_paced_controls": superior,
        "quality_guards": guards,
        "positive_frozen_context": frozen_gain["ci95_pp"][0] > 0,
        "cost_saving": saving >= 0.25,
    }
    result = {
        "heldout_means": means,
        "frozen_means": warm,
        "coverage_vs_references": effects,
        "coverage_vs_frozen_context": frozen_gain,
        "processing_seconds": costs,
        "coverage_saving_fraction": saving,
        "diagnostic_seconds": {
            arm: statistics.fmean(r["diagnostic_seconds"] for r in fresh if r["arm"] == arm) for arm in ARMS
        },
        "gates": gates,
        "decision": "coverage_content_lead_qualified" if all(gates.values()) else "no_coverage_content_lead_qualified",
        "scientific_ssl_updates": 29952,
        "new_primary_fits": 738,
        "controller_fit": False,
        "per_source_primary": {
            n: {a: {m: metric(n, a, "finetune", m) for m in METRICS} for a in (*ARMS, "full")} for n in names
        },
        "source_metadata": {r["stream"]: {"pattern": r["pattern"], "dwell": r["dwell"]} for r in fresh},
    }
    atomic_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = {"pid": os.getpid()}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - start)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if shutil.disk_usage(out).free < 10 * 1024**3 + 276 * 5063947 + 300 * 1024**2:
            raise ValueError("Storage reserve insufficient before collection")
        progress(stage="locking_inputs")
        config, data, streams, clips, lock = load_inputs(out)
        labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
        torch.manual_seed(11)
        initial = _method(config).cuda()
        reference = copy.deepcopy(initial.encoder).requires_grad_(False).eval()
        warm = {}
        for seed in SEEDS:
            old = PRIOR / "warm" / f"seed_{seed}"
            directory = out / "warm" / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("state.pt", "readout.json", "predictions.npz"):
                shutil.copyfile(old / name, directory / name)
            warm[seed] = torch.load(old / "state.pt", map_location="cpu", weights_only=True)
        sample = tensor(streams[0], clips)[:128]
        signatures = []
        for _ in range(2):
            model, optimizer = restore(initial, config.optim, warm[SEEDS[0]])
            run_admission(model, optimizer, sample, "coverage", reference, SEEDS[0], (128,))
            signatures.append(fingerprint(capture(model, optimizer)))
        if len(set(signatures)) != 1:
            raise ValueError("Preflight nondeterministic")
        evaluate_gpu(model, optimizer, data, out / "preflight")
        atomic_json(out / "preflight.json", {"passed": True, "discarded_ssl_updates": 4, "discarded_primary_fits": 3})
        rows = []
        for stream in streams:
            arrivals = tensor(stream, clips)
            for seed in SEEDS:
                for arm in (*ARMS, "full"):
                    progress(stage="collection", stream=stream["name"], seed=seed, arm=arm, completed=len(rows))
                    base = out / "runs" / stream["name"] / arm / f"seed_{seed}"
                    base.mkdir(parents=True, exist_ok=True)
                    if arm == "full" and stream["role"] == "development":
                        old = PRIOR / "runs" / stream["old_name"] / "1111" / f"seed_{seed}"
                        endpoints = []
                        for clock in CHECKPOINTS:
                            target = base / f"at_{clock}"
                            target.mkdir(exist_ok=True)
                            for name in ("state.pt", "readout.json", "predictions.npz"):
                                shutil.copyfile(old / f"at_{clock}" / name, target / name)
                            endpoints.append(str(target.relative_to(out)))
                        training = read(old / "training.json")
                        rows.append(
                            {
                                "stream": stream["name"],
                                "role": stream["role"],
                                "pattern": stream["pattern"],
                                "dwell": "development",
                                "seed": seed,
                                "arm": arm,
                                "reused": True,
                                "updates": 384,
                                "processing_seconds": sum(p["online_seconds"] for p in training["periods"]),
                                "diagnostic_seconds": 0,
                                "directory": str(base.relative_to(out)),
                                "endpoints": endpoints,
                            }
                        )
                        continue
                    model, optimizer = restore(initial, config.optim, warm[seed])
                    endpoints = []

                    def observe(clock, state):
                        target = base / f"at_{clock}"
                        target.mkdir(exist_ok=True)
                        torch.save(state, target / "state.pt")
                        if arm == "early_stop" and clock == 12288:
                            atomic_json(target / "readout-alias.json", {"directory": endpoints[0]})
                            if fingerprint(state["learner"]) != fingerprint(
                                torch.load(base / "at_6144/state.pt", map_location="cpu", weights_only=True)["learner"]
                            ):
                                raise ValueError("Early stop changed after budget exhaustion")
                        else:
                            evaluate_gpu(model, optimizer, data, target)
                        if arm == "early_stop" and stream["role"] == "development":
                            old_state = torch.load(
                                PRIOR / "runs" / stream["old_name"] / "1100" / f"seed_{seed}/at_{clock}/state.pt",
                                map_location="cpu",
                                weights_only=True,
                            )
                            if fingerprint(state["learner"]) != fingerprint(old_state["learner"]):
                                raise ValueError("Early-stop baseline differs from audited original")
                        endpoints.append(str(target.relative_to(out)))
                        return {"state_fingerprint": fingerprint(state), "updates": state["updates"]}

                    training = (
                        run_branch(model, optimizer, arrivals, "full", seed, CHECKPOINTS, observe)
                        if arm == "full"
                        else run_admission(model, optimizer, arrivals, arm, reference, seed, CHECKPOINTS, observe)
                    )
                    atomic_json(base / "training.json", training)
                    rows.append(
                        {
                            "stream": stream["name"],
                            "role": stream["role"],
                            "pattern": stream["pattern"],
                            "dwell": stream.get("dwell", "development"),
                            "seed": seed,
                            "arm": arm,
                            "reused": False,
                            "updates": training["updates"],
                            "processing_seconds": training.get("processing_seconds", training.get("online_seconds")),
                            "diagnostic_seconds": training.get("diagnostic_seconds", 0),
                            "directory": str(base.relative_to(out)),
                            "endpoints": endpoints,
                        }
                    )
                    atomic_json(out / "collection.json", {"records": rows, "complete": len(rows) == 150})
            del arrivals
        atomic_json(out / "collection.json", {"records": rows, "complete": True})
        replay_updates, audited = 0, 0
        for stream in streams:
            arrivals = tensor(stream, clips)
            for row in [r for r in rows if r["stream"] == stream["name"] and not r["reused"]]:
                seed, arm = row["seed"], row["arm"]
                progress(
                    stage="exact_replay",
                    stream=stream["name"],
                    seed=seed,
                    arm=arm,
                    audited=audited,
                    replay_updates=replay_updates,
                )
                model, optimizer = restore(initial, config.optim, warm[seed])

                def check(clock, state):
                    stored = torch.load(
                        out / row["directory"] / f"at_{clock}/state.pt", map_location="cpu", weights_only=True
                    )
                    if fingerprint(state) != fingerprint(stored):
                        raise ValueError("Complete learner/selector/memory replay mismatch")
                    return {"passed": True}

                repeated = (
                    run_branch(model, optimizer, arrivals, "full", seed, CHECKPOINTS, check)
                    if arm == "full"
                    else run_admission(model, optimizer, arrivals, arm, reference, seed, CHECKPOINTS, check)
                )
                original = read(out / row["directory"] / "training.json")
                fields = (
                    ("start", "end", "exposure_indices", "updates", "loss_gradients")
                    if arm == "full"
                    else (
                        "start",
                        "end",
                        "selected_batches",
                        "exposure_indices",
                        "updates",
                        "loss_gradients",
                        "centroids",
                        "novelty_rounds",
                        "incoming_losses",
                        "prior_training_loss",
                        "relative_losses",
                    )
                )
                for a, b in zip(original["blocks"], repeated["blocks"], strict=True):
                    if any(json.dumps(a[k]) != json.dumps(b[k]) for k in fields):
                        raise ValueError("Selection/diagnostic/exposure replay differs")
                replay_updates += repeated["updates"]
                audited += 1
            del arrivals
        if replay_updates != 29952 or any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
            raise ValueError("Final count/input lock failed")
        atomic_json(
            out / "training-replay-audit.json",
            {"passed": True, "new_trajectories": audited, "endpoint_states": 276, "audit_ssl_updates": replay_updates},
        )
        result = report(out, rows, labels, domains)
        progress(
            stage="complete", completed=150, scientific_ssl_updates=29952, primary_fits=738, decision=result["decision"]
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
