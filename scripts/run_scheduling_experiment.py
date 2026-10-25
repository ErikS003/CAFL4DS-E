"""Budget-matched complete stream schedules, frozen selection, and exact replay."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.return_actions import run_branch
from cafl4ds.scheduling import DRAWS, HALF, METRICS, PERIOD, READERS, SCHEDULES, SEEDS, choose, interval, qualifies
from cafl4ds.short_buffer import capture, restore
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import evaluate_gpu
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/regime-windows/20261013-gpu"
FRAMES = ROOT / "outputs/scheduling/20261013-frames"
INTAKE = ROOT / "outputs/scheduling/20261013-intake"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scores(out, directory, labels, domains):
    directory = out / directory
    if (directory / "readout-alias.json").exists():
        directory = out / read(directory / "readout-alias.json")["directory"]
    with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
        return {
            r: {
                m: statistics.fmean(100 * float((predictions[f"{d}_{r}"][mask] == labels[mask]).mean()) for d in DRAWS)
                for m, mask in {"balanced": np.ones(len(labels), bool), "A": domains == 0, "B": domains == 1}.items()
            }
            for r in READERS
        }


def load_inputs(out):
    old = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "independent-verification.json")["passed"] or any(
        digest(Path(p)) != h for p, h in old["hashes"].items()
    ):
        raise ValueError("Audited ancestor inputs changed")
    intake = read(FRAMES / "intake.json")
    if not intake["complete"] or len(intake["clips"]) != 240:
        raise ValueError("Incomplete sources")
    paths = [
        FRAMES / "intake.json",
        FRAMES / "allocation.json",
        INTAKE / "selection.json",
        INTAKE / "selection.pool.json",
        INTAKE / "download_manifest.json",
        ROOT / ".cache/scheduling-known-videos.txt",
        ROOT / "docs/project-plan/slow-edge-scheduling-frozen.md",
        ROOT / "scripts/select_scheduling_clips.py",
        ROOT / "scripts/prepare_scheduling_frames.py",
        ROOT / "scripts/predecode_scheduling_frames.py",
        ROOT / "scripts/verify_scheduling_experiment.py",
        Path(__file__),
        ROOT / "cafl4ds/scheduling.py",
        ROOT / "scripts/decode_history_video.py",
        ROOT / "scripts/fetch_cross_clips.py",
        PRIOR / "evaluation_arrays.npz",
        PRIOR / "data-role-manifest.json",
        PRIOR / "independent-verification.json",
        Path(intake["ffmpeg_path"]),
    ]
    for clip in intake["clips"]:
        frame = Path(clip["frames_path"])
        paths.extend(
            (
                frame,
                Path(clip["video_path"]),
                frame.parent / "manifest.json",
                frame.parent / "decoder.log",
                Path(clip["keyframe_path"]).parents[3] / "labels/train" / f"{clip['clip_id']}.json",
            )
        )
        if digest(frame) != clip["frames_sha256"]:
            raise ValueError("Pixel cache changed")
    paths.extend(INTAKE.glob("amendment-*/*.json"))
    for failure in FRAMES.glob("*/predecode-failure.json"):
        paths.extend((failure, failure.parent / "decoder.log", INTAKE / "videos/train" / f"{failure.parent.name}.mov"))
    for reason in INTAKE.glob("amendment-*/reason.json"):
        failed_id = read(reason)["failed_clip"]
        paths.extend((FRAMES / failed_id / "decoder.log", INTAKE / "videos/train" / f"{failed_id}.mov"))
    image = ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt"
    paths.append(image)
    with np.load(PRIOR / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            torch.load(image, map_location="cpu", weights_only=True),
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in DRAWS},
            read(PRIOR / "data-role-manifest.json"),
        )
    lock = {
        "hashes": {**old["hashes"], **{str(p): digest(p) for p in paths}},
        "config": old["config"],
        "streams": intake["streams"],
        "seeds": SEEDS,
        "schedules": SCHEDULES,
        "period": PERIOD,
        "scientific_ssl_updates": 46080,
        "primary_fits": 1449,
    }
    lock = json.loads(json.dumps(lock))
    if (out / "input_lock.json").exists() and read(out / "input_lock.json") != lock:
        raise ValueError("Resume inputs differ")
    atomic_json(out / "input_lock.json", lock)
    for p in paths:
        if p.suffix in (".py", ".md") and p.is_relative_to(ROOT):
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    return OmegaConf.create(old["config"]), data, intake, lock


def tensors(stream, clips):
    result = []
    for period in stream["periods"]:
        array = np.concatenate([np.load(clips[c]["frames_path"], allow_pickle=False) for c in period])
        if array.shape != (PERIOD, 64, 64, 3):
            raise ValueError("Incorrect period size")
        result.append(torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255)
    return result


def train_period(model, optimizer, arrivals, active, seed, period):
    trace = run_branch(model, optimizer, arrivals, "full" if active else "hold", seed, (PERIOD,))
    for block in trace["blocks"]:
        for key in ("start", "end"):
            block[key] += period * PERIOD
        block["selected_indices"] = [k + period * PERIOD for k in block["selected_indices"]]
        block["exposure_indices"] = [[k + period * PERIOD for k in batch] for batch in block["exposure_indices"]]
    return trace


def selection(out, records, labels, domains, lock):
    means = {
        s: statistics.fmean(
            scores(out, row["endpoints"][-1], labels, domains)["finetune"]["balanced"]
            for row in records
            if row["role"] == "development" and row["schedule"] == s
        )
        for s in HALF
    }
    by_pattern = {
        pattern: choose(
            {
                s: statistics.fmean(
                    scores(out, row["endpoints"][-1], labels, domains)["finetune"]["balanced"]
                    for row in records
                    if row["role"] == "development" and row["pattern"] == pattern and row["schedule"] == s
                )
                for s in HALF
            }
        )
        for pattern in {row["pattern"] for row in records}
    }
    result = {
        "schedule": choose(means),
        "development_terminal_means": means,
        "pattern_specific_schedules": by_pattern,
        "locked_inputs_sha256": digest(out / "input_lock.json"),
        "development_streams": [s["name"] for s in lock["streams"] if s["role"] == "development"],
        "heldout_collection_started": False,
    }
    if (out / "development-selection.json").exists():
        original = read(out / "development-selection.json")
        if any(original[k] != v for k, v in result.items()):
            raise ValueError("Frozen development choice changed")
        return original
    result["frozen_unix"] = time.time()
    atomic_json(out / "development-selection.json", result)
    return result


def report(out, records, labels, domains, chosen):
    values = {
        (r["stream"], r["seed"], r["schedule"]): [scores(out, p, labels, domains) for p in r["endpoints"]]
        for r in records
    }
    streams = sorted({r["stream"] for r in records if r["role"] == "heldout"})

    def metric(stream, schedule, reader, m, endpoint=-1):
        return statistics.fmean(values[stream, seed, schedule][endpoint][reader][m] for seed in SEEDS)

    fixed = chosen["schedule"]
    contrasts = {
        s: {
            r: {
                level: {
                    m: interval(
                        [metric(n, s, r, m) - metric(n, fixed, r, m) for n in streams], 15 if level == "adjusted" else 1
                    )
                    for m in METRICS
                }
                for level in ("ordinary", "adjusted")
            }
            for r in READERS
        }
        for s in HALF
        if s != fixed
    }
    qualified = [s for s in contrasts if qualifies(contrasts[s]["finetune"]["adjusted"])]
    means = {
        s: {r: {m: statistics.fmean(metric(n, s, r, m) for n in streams) for m in METRICS} for r in READERS}
        for s in SCHEDULES
    }
    winners = {n: choose({s: metric(n, s, "finetune", "balanced") for s in HALF}) for n in streams}
    gaps = {
        n: metric(n, winners[n], "finetune", "balanced") - metric(n, fixed, "finetune", "balanced") for n in streams
    }
    patterns = {r["stream"]: r["pattern"] for r in records}
    pattern_diagnostic = {
        n: metric(n, chosen["pattern_specific_schedules"][patterns[n]], "finetune", "balanced")
        - metric(n, fixed, "finetune", "balanced")
        for n in streams
    }
    replicate = {}
    for n in streams:
        others = [k for k in streams if k != n and patterns[k] == patterns[n]]
        if others:
            s = choose({s: statistics.fmean(metric(k, s, "finetune", "balanced") for k in others) for s in HALF})
            replicate[n] = {
                "schedule_selected_on_other_replicate": s,
                "gap_pp": metric(n, s, "finetune", "balanced") - metric(n, fixed, "finetune", "balanced"),
            }
    controls = {
        ref: {
            r: {m: interval([metric(n, fixed, r, m) - metric(n, ref, r, m) for n in streams]) for m in METRICS}
            for r in READERS
        }
        for ref in ("1111", "0000")
    }
    trajectory = {
        s: {
            r: {m: [statistics.fmean(metric(n, s, r, m, p) for n in streams) for p in range(4)] for m in METRICS}
            for r in READERS
        }
        for s in SCHEDULES
    }
    timing = {
        s: {
            "mean_online_seconds": statistics.fmean(
                r["online_seconds"] for r in records if r["role"] == "heldout" and r["schedule"] == s
            ),
            "stream_seed_seconds": [
                r["online_seconds"] for r in records if r["role"] == "heldout" and r["schedule"] == s
            ],
        }
        for s in SCHEDULES
    }
    result = {
        "selected_fixed_schedule": fixed,
        "heldout_terminal_means": means,
        "half_vs_frozen_fixed": contrasts,
        "qualified_static_alternatives": qualified,
        "fixed_vs_controls_secondary": controls,
        "optimistic_six_schedule_reference": {
            "winners": winners,
            "stream_gaps_pp": gaps,
            "mean_gap_pp": statistics.fmean(gaps.values()),
            "outcome_selected_not_a_policy": True,
        },
        "development_pattern_specific_secondary_gaps_pp": pattern_diagnostic,
        "other_replicate_selected_secondary": replicate,
        "trajectory_means": trajectory,
        "timing": timing,
        "scientific_ssl_updates": 46080,
        "primary_fits": 1449,
        "decision": "static_baseline_transfer_needs_recalibration" if qualified else "no_static_alternative_qualified",
        "adaptive_controller_qualified": False,
        "uncertainty": "paired intervals across six held-out clip-disjoint synthetic streams, seeds/panels averaged within stream; training/readout population conditional, capture sessions unknown",
    }
    atomic_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    status = {"pid": os.getpid(), "total_trajectories": 240}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - tick)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config, data, intake, lock = load_inputs(out)
        labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
        clips = {c["clip_id"]: c for c in intake["clips"]}
        torch.manual_seed(11)
        initial = _method(config).cuda()
        warm = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            optimizer = instantiate(config.optim, params=list(initial.parameters()))
            state = capture(initial, optimizer)
            directory = out / "warm" / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "state.pt").exists():
                if fingerprint(
                    torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
                ) != fingerprint(state):
                    raise ValueError("Warm state changed")
            else:
                torch.save(state, directory / "state.pt")
            warm[seed] = state
            evaluate_gpu(initial, optimizer, data, directory)
        first = tensors(intake["streams"][0], clips)[0]
        signatures = []
        for _ in range(2):
            model, optimizer = restore(initial, config.optim, warm[SEEDS[0]])
            run_branch(model, optimizer, first[:64], "full", SEEDS[0], (64,))
            before = fingerprint(capture(model, optimizer))
            run_branch(model, optimizer, first[:64], "hold", SEEDS[0], (64,))
            if fingerprint(capture(model, optimizer)) != before:
                raise ValueError("Hold mutates learner")
            signatures.append(before)
        if len(set(signatures)) != 1:
            raise ValueError("Preflight determinism failed")
        evaluate_gpu(model, optimizer, data, out / "preflight")
        atomic_json(out / "preflight.json", {"passed": True, "discarded_ssl_updates": 4, "discarded_primary_fits": 3})
        del first
        records = []
        chosen = None
        prefix_checks = 0
        for stream in intake["streams"]:
            if stream["role"] == "heldout" and chosen is None:
                chosen = selection(out, records, labels, domains, lock)
                progress(
                    stage="fixed_schedule_frozen",
                    selected_schedule=chosen["schedule"],
                    completed_trajectories=len(records),
                )
            periods = tensors(stream, clips)
            prefixes = {}
            for seed in SEEDS:
                for schedule in SCHEDULES:
                    progress(
                        stage="scientific_collection",
                        stream=stream["name"],
                        role=stream["role"],
                        seed=seed,
                        schedule=schedule,
                        completed_trajectories=len(records),
                    )
                    model, optimizer = restore(initial, config.optim, warm[seed])
                    directory = out / "runs" / stream["name"] / schedule / f"seed_{seed}"
                    directory.mkdir(parents=True, exist_ok=True)
                    previous = f"warm/seed_{seed}"
                    endpoints, traces, updates = [], [], 0
                    started = time.time()
                    for p, bit in enumerate(schedule):
                        before = fingerprint(capture(model, optimizer))
                        trace = train_period(model, optimizer, periods[p], bit == "1", seed, p)
                        updates += trace["updates"]
                        checkpoint = directory / f"at_{(p + 1) * PERIOD}"
                        checkpoint.mkdir(exist_ok=True)
                        state = {
                            "learner": capture(model, optimizer),
                            "updates": updates,
                            "arrivals": (p + 1) * PERIOD,
                            "schedule": schedule,
                            "period": p,
                        }
                        torch.save(state, checkpoint / "state.pt")
                        signature = fingerprint(state["learner"])
                        if bit == "0" and signature != before:
                            raise ValueError("Pause changes learner state")
                        key = (seed, schedule[: p + 1])
                        if key in prefixes:
                            if prefixes[key] != signature:
                                raise ValueError("Future schedule affects prefix")
                            prefix_checks += 1
                        prefixes[key] = signature
                        if bit == "1":
                            evaluate_gpu(model, optimizer, data, checkpoint)
                            previous = str(checkpoint.relative_to(out))
                        else:
                            atomic_json(checkpoint / "readout-alias.json", {"directory": previous})
                        endpoints.append(str(checkpoint.relative_to(out)))
                        traces.append(trace)
                    if updates != schedule.count("1") * PERIOD // 32:
                        raise ValueError("Schedule budget mismatch")
                    atomic_json(directory / "training.json", {"periods": traces, "updates": updates})
                    records.append(
                        {
                            "stream": stream["name"],
                            "role": stream["role"],
                            "pattern": stream["pattern"],
                            "seed": seed,
                            "schedule": schedule,
                            "updates": updates,
                            "online_seconds": sum(t["online_seconds"] for t in traces),
                            "endpoints": endpoints,
                            "directory": str(directory.relative_to(out)),
                            "started_unix": started,
                        }
                    )
                    atomic_json(out / "collection.json", {"records": records, "complete": len(records) == 240})
            del periods
        replay_updates = 0
        for stream in intake["streams"]:
            periods = tensors(stream, clips)
            for row in [r for r in records if r["stream"] == stream["name"]]:
                progress(
                    stage="independent_training_replay",
                    stream=stream["name"],
                    seed=row["seed"],
                    schedule=row["schedule"],
                    replay_updates=replay_updates,
                )
                model, optimizer = restore(initial, config.optim, warm[row["seed"]])
                original = read(out / row["directory"] / "training.json")
                for p, bit in enumerate(row["schedule"]):
                    repeated = train_period(model, optimizer, periods[p], bit == "1", row["seed"], p)
                    state = torch.load(out / row["endpoints"][p] / "state.pt", map_location="cpu", weights_only=True)
                    if fingerprint(capture(model, optimizer)) != fingerprint(state["learner"]):
                        raise ValueError("Independent complete learner replay differs")
                    for a, b in zip(original["periods"][p]["blocks"], repeated["blocks"], strict=True):
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
                                raise ValueError("Replay exposure/loss gradient mismatch")
                    replay_updates += repeated["updates"]
            del periods
        if replay_updates != 46080 or any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
            raise ValueError("Replay budget or final input lock failed")
        atomic_json(
            out / "training-replay-audit.json",
            {
                "passed": True,
                "trajectories": 240,
                "complete_states": 960,
                "audit_ssl_updates": replay_updates,
                "causal_prefix_checks": prefix_checks,
            },
        )
        result = report(out, records, labels, domains, chosen)
        progress(
            stage="complete",
            completed_trajectories=240,
            scientific_ssl_updates=46080,
            primary_fits=1449,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
