"""Calibrate absolute SSL stopping allowances and freeze before fresh dwell transfer."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
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
from cafl4ds.stopping_calibration import BUDGETS, DRAWS, METRICS, READERS, SEEDS, choose, gates, interval
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import evaluate_gpu
from scripts.run_short_buffer_experiment import digest
from scripts.run_scheduling_experiment import scores

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/scheduling/20261013-gpu"
FRAMES = ROOT / "outputs/stopping-calibration/20261013-frames"
INTAKE = ROOT / "outputs/stopping-calibration/20261013-intake"


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
        ROOT / "docs/project-plan/slow-edge-stopping-calibration-frozen.md",
        Path(__file__),
        ROOT / "scripts/verify_stopping_calibration.py",
        ROOT / "scripts/select_stopping_clips.py",
        ROOT / "scripts/prepare_stopping_frames.py",
        ROOT / "scripts/predecode_stopping_frames.py",
        ROOT / "cafl4ds/stopping_calibration.py",
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
                "scientific_new_ssl_updates": 7488,
                "new_primary_fits": 306,
                "reused_primary_fits": 153,
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


def tensor(stream, clips):
    array = np.concatenate(
        [np.load(clips[c]["frames_path"], allow_pickle=False) for group in stream["periods"] for c in group]
    )
    if array.shape != (12288, 64, 64, 3):
        raise ValueError("Whole-stream shape changed")
    return torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255


def copy_predictions(source, target):
    for name in ("readout.json", "predictions.npz"):
        shutil.copyfile(source / name, target / name)


def hold_check(initial, config, learner, arrivals, seed, budget):
    model, optimizer = restore(initial, config.optim, learner)
    before = fingerprint(capture(model, optimizer))
    suffix = arrivals[32 * budget :]
    held = run_branch(model, optimizer, suffix, "hold", seed, (len(suffix),))
    if before != fingerprint(capture(model, optimizer)) or held["updates"]:
        raise ValueError("Stopped learner changed on remaining arrivals")
    return held["online_seconds"]


def calibration_choice(out, records, labels, domains):
    means = {
        k: statistics.fmean(
            scores(out, r["directory"], labels, domains)["finetune"]["balanced"]
            for r in records
            if r["budget"] == k and r["role"] == "development"
        )
        for k in BUDGETS
    }
    result = {
        "budget": choose(means),
        "development_terminal_means": means,
        "locked_inputs_sha256": digest(out / "input_lock.json"),
        "frozen_unix": time.time(),
        "new_scientific_collection_started": False,
    }
    atomic_json(out / "fixed-budget-selection.json", result)
    return result


def report(out, records, labels, domains, selected):
    rows = [r for r in records if r["role"] == "heldout"]
    names = sorted({r["stream"] for r in rows})
    values = {(r["stream"], r["seed"], r["budget"]): scores(out, r["directory"], labels, domains) for r in rows}

    def metric(n, k, reader, m):
        return statistics.fmean(values[n, seed, k][reader][m] for seed in SEEDS)

    chosen = selected["budget"]
    contrasts = {
        ref: {
            reader: {
                level: {
                    m: interval(
                        [metric(n, chosen, reader, m) - metric(n, ref, reader, m) for n in names],
                        9 if level == "adjusted" else 1,
                    )
                    for m in METRICS
                }
                for level in ("ordinary", "adjusted")
            }
            for reader in READERS
        }
        for ref in (0, 192, 384)
    }
    means = {
        k: {
            reader: {m: statistics.fmean(metric(n, k, reader, m) for n in names) for m in METRICS} for reader in READERS
        }
        for k in BUDGETS
    }
    costs = {k: statistics.fmean(r["processing_seconds"] for r in rows if r["budget"] == k) for k in BUDGETS}
    saving = 1 - costs[chosen] / costs[384]
    qualification = gates(chosen, {ref: contrasts[ref]["finetune"]["adjusted"] for ref in contrasts}, saving)
    per_source = {n: {k: {m: metric(n, k, "finetune", m) for m in METRICS} for k in BUDGETS} for n in names}
    winners = {n: choose({k: metric(n, k, "finetune", "balanced") for k in BUDGETS}) for n in names}
    gaps = {n: metric(n, winners[n], "finetune", "balanced") - metric(n, chosen, "finetune", "balanced") for n in names}
    shortfalls = {
        n: {
            m: metric(n, chosen, "finetune", m) - metric(n, 384, "finetune", m)
            for m in METRICS
            if round(metric(n, chosen, "finetune", m) - metric(n, 384, "finetune", m), 12) < -1
        }
        for n in names
    }
    meta = {r["stream"]: {"pattern": r["pattern"], "dwell": r["dwell"]} for r in rows}
    result = {
        "selected_budget": chosen,
        "heldout_means": means,
        "selected_vs_references": contrasts,
        "transfer_gates": qualification,
        "per_source_primary_curves": per_source,
        "source_metadata": meta,
        "diagnostic_shortfalls_vs_full": shortfalls,
        "optimistic_budget_reference": {
            "winners": winners,
            "gaps_pp": gaps,
            "mean_gap_pp": statistics.fmean(gaps.values()),
            "outcome_selected_not_a_policy": True,
        },
        "prepared_arrival_seconds": costs,
        "observed_selected_savings_fraction": saving,
        "scientific_new_ssl_updates": 7488,
        "audit_ssl_updates": 8064,
        "new_primary_fits": 306,
        "reused_primary_fits": 153,
        "decision": "fixed_budget_transfer_qualified"
        if qualification["combined"]
        else "fixed_budget_transfer_not_qualified",
        "controller_fit": False,
        "uncertainty": "paired t across six fresh clip-disjoint synthetic streams, seed/panel means within source, reused warm/reader panels; two sources per dwell scale descriptive only",
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
        progress(stage="locking_inputs")
        config, data, streams, clips, lock = load_inputs(out)
        labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
        torch.manual_seed(11)
        initial = _method(config).cuda()
        warm = {}
        for seed in SEEDS:
            old = PRIOR / "warm" / f"seed_{seed}"
            state = torch.load(old / "state.pt", map_location="cpu", weights_only=True)
            if fingerprint(initial.state_dict()) != fingerprint(state["model"]):
                raise ValueError("Warm weights changed")
            directory = out / "warm" / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old / "state.pt", directory / "state.pt")
            copy_predictions(old, directory)
            warm[seed] = state
        sample = tensor(streams[0], clips)
        signatures = []
        for _ in range(2):
            model, optimizer = restore(initial, config.optim, warm[SEEDS[0]])
            run_branch(model, optimizer, sample[:64], "full", SEEDS[0], (64,))
            before = fingerprint(capture(model, optimizer))
            run_branch(model, optimizer, sample[64:128], "hold", SEEDS[0], (64,))
            if fingerprint(capture(model, optimizer)) != before:
                raise ValueError("Preflight hold mutated learner")
            signatures.append(before)
        if len(set(signatures)) != 1:
            raise ValueError("Preflight nondeterministic")
        evaluate_gpu(model, optimizer, data, out / "preflight")
        atomic_json(out / "preflight.json", {"passed": True, "discarded_ssl_updates": 4, "discarded_primary_fits": 3})
        del sample
        records, trainings = [], []
        selected = None
        for stream in streams:
            if stream["role"] == "heldout" and selected is None:
                selected = calibration_choice(out, records, labels, domains)
                progress(stage="budget_frozen", selected_budget=selected["budget"])
            arrivals = tensor(stream, clips)
            for seed in SEEDS:
                progress(
                    stage="scientific_collection",
                    stream=stream["name"],
                    role=stream["role"],
                    seed=seed,
                    completed_cases=len(records),
                )
                states = {0: warm[seed]}
                base = out / "curves" / stream["name"] / f"seed_{seed}"
                base.mkdir(parents=True, exist_ok=True)
                started = time.time()
                model, optimizer = restore(initial, config.optim, warm[seed])
                fresh = stream["role"] == "heldout"
                checkpoints = tuple(32 * k for k in BUDGETS[1:]) if fresh else (1536,)

                def observe(clock, state):
                    k = clock // 32
                    states[k] = state["learner"]
                    directory = base / f"k_{k}"
                    directory.mkdir(exist_ok=True)
                    evaluate_gpu(model, optimizer, data, directory)
                    return {"learner_fingerprint": fingerprint(states[k]), "updates": k}

                trained = run_branch(
                    model, optimizer, arrivals if fresh else arrivals[:1536], "full", seed, checkpoints, observe
                )
                atomic_json(base / "training.json", trained)
                trainings.append(
                    {
                        "stream": stream["name"],
                        "seed": seed,
                        "updates": trained["updates"],
                        "directory": str(base.relative_to(out)),
                        "role": stream["role"],
                    }
                )
                if not fresh:
                    old_base = PRIOR / "runs" / stream["old_name"] / "1111" / f"seed_{seed}"
                    old_training = read(old_base / "training.json")
                    old_blocks = [b for period in old_training["periods"] for b in period["blocks"]]
                    for k in BUDGETS[2:]:
                        old = old_base / f"at_{32 * k}"
                        states[k] = torch.load(old / "state.pt", map_location="cpu", weights_only=True)["learner"]
                        directory = base / f"k_{k}"
                        directory.mkdir(exist_ok=True)
                        copy_predictions(old, directory)
                else:
                    old_blocks = trained["blocks"]
                for k in BUDGETS:
                    directory = base / f"k_{k}"
                    directory.mkdir(exist_ok=True)
                    if k == 0:
                        copy_predictions(out / "warm" / f"seed_{seed}", directory)
                    learner = states[k]
                    torch.save(
                        {"learner": learner, "stop_updates": k, "stop_arrivals": 32 * k, "terminal_arrivals": 12288},
                        directory / "state.pt",
                    )
                    held_seconds = hold_check(initial, config, learner, arrivals, seed, k)
                    training_blocks = trained["blocks"] if k == 48 else old_blocks
                    prefix_seconds = sum(b["online_seconds"] for b in training_blocks if b["end"] <= 32 * k)
                    records.append(
                        {
                            "stream": stream["name"],
                            "role": stream["role"],
                            "pattern": stream["pattern"],
                            "dwell": stream.get("dwell", "development"),
                            "seed": seed,
                            "budget": k,
                            "directory": str(directory.relative_to(out)),
                            "processing_seconds": prefix_seconds + held_seconds,
                            "held_suffix_seconds": held_seconds,
                            "readout_reused": not fresh and k != 48 or k == 0,
                            "started_unix": started,
                        }
                    )
                atomic_json(
                    out / "collection.json",
                    {"records": records, "trainings": trainings, "complete": len(records) == 180},
                )
            del arrivals
        audited, replay_updates, hold_cases = 0, 0, 0
        for stream in streams:
            arrivals = tensor(stream, clips)
            for seed in SEEDS:
                progress(
                    stage="independent_replay",
                    stream=stream["name"],
                    seed=seed,
                    audited_parents=audited,
                    replay_updates=replay_updates,
                )
                model, optimizer = restore(initial, config.optim, warm[seed])
                base = out / "curves" / stream["name"] / f"seed_{seed}"
                fresh = stream["role"] == "heldout"
                clocks = tuple(32 * k for k in BUDGETS[1:]) if fresh else (1536, 3072)

                def check(clock, state):
                    learner = state["learner"]
                    k = clock // 32
                    stored = torch.load(base / f"k_{k}/state.pt", map_location="cpu", weights_only=True)
                    if fingerprint(learner) != fingerprint(stored["learner"]):
                        raise ValueError("Fresh prefix or original development96 checkpoint replay differs")
                    return {"passed": True}

                replay = run_branch(
                    model, optimizer, arrivals if fresh else arrivals[:3072], "full", seed, clocks, check
                )
                original = read(base / "training.json")
                compared = replay["blocks"] if fresh else replay["blocks"][:48]
                for a, b in zip(original["blocks"], compared, strict=True):
                    for key in (
                        "start",
                        "end",
                        "selected_indices",
                        "exposure_indices",
                        "queued_indices",
                        "updates",
                        "loss_gradients",
                    ):
                        if json.dumps(a[key]) != json.dumps(b[key]):
                            raise ValueError("Independent exposure/loss replay changed")
                for k in BUDGETS:
                    state = torch.load(base / f"k_{k}/state.pt", map_location="cpu", weights_only=True)
                    hold_check(initial, config, state["learner"], arrivals, seed, k)
                    hold_cases += 1
                replay_updates += replay["updates"]
                audited += 1
            del arrivals
        if replay_updates != 8064 or any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
            raise ValueError("Final replay count/input lock failed")
        atomic_json(
            out / "training-replay-audit.json",
            {
                "passed": True,
                "parents": audited,
                "audit_ssl_updates": replay_updates,
                "hold_counterfactuals": hold_cases,
                "new_checkpoints_replayed": 102,
                "original_development96_states_reproduced": 12,
            },
        )
        result = report(out, records, labels, domains, selected)
        progress(
            stage="complete",
            completed_cases=180,
            scientific_new_ssl_updates=7488,
            new_primary_fits=306,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
