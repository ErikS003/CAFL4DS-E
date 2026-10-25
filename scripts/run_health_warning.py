"""Run compact CKA warning trajectories and replay all complete online states."""

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
from cafl4ds.health_calibration import changes, embeddings
from cafl4ds.health_warning import LANDMARKS, MODELS, fit, remaining_effect, report
from cafl4ds.recurrence import Evaluation
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import CONTROL, cases, learner, reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import DRAWS, evaluate_gpu
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_scheduling_experiment import scores
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/health-calibration/20261013-gpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def alias(case, landmark):
    if landmark == 0:
        return Path(case["hold"]).parent
    if (case["role"] == "six_stream" and landmark == 16) or (case["role"] == "control" and landmark == 32):
        return Path(case["actual"]).parent
    return None


def readout_source(directory):
    if (directory / "readout-alias.json").exists():
        return CONTROL / read(directory / "readout-alias.json")["directory"]
    return directory


def load(out):
    old = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "verification.json")["passed"] or not read(PRIOR / "independent-verification.json")["passed"]:
        raise ValueError("Prior health audit failed")
    hashes = dict(old["hashes"])
    selected = cases()
    paths = [
        Path(__file__),
        ROOT / "scripts/verify_health_warning.py",
        ROOT / "cafl4ds/health_warning.py",
        ROOT / "tests/unit/test_health_warning.py",
        ROOT / "docs/project-plan/slow-edge-health-warning-frozen.md",
        PRIOR / "verification.json",
        PRIOR / "independent-verification.json",
    ]
    for case in selected:
        for j in LANDMARKS:
            directory = alias(case, j)
            if directory:
                paths.append(directory / "state.pt")
                if (directory / "readout-alias.json").exists():
                    paths.append(directory / "readout-alias.json")
                paths.extend(readout_source(directory) / name for name in ("predictions.npz", "readout.json"))
    for path in paths:
        hashes[str(path)] = digest(path)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Inherited locked source changed")
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    original = read(ROOT / "outputs/regime-windows/20261013-gpu/data-role-manifest.json")
    config = OmegaConf.create(old["config"])
    ancestor = ROOT / "outputs/regime-windows/20261013-gpu"
    with np.load(ancestor / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            torch.load(
                ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt", weights_only=True
            ),
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in DRAWS},
            read(ancestor / "data-role-manifest.json"),
        )
    lock = {
        "hashes": hashes,
        "config": old["config"],
        "reference": old["reference"],
        "cases": selected,
        "landmarks": LANDMARKS,
        "primary_landmark": 8,
        "data_role": "reused_development",
    }
    atomic_json(out / "input_lock.json", lock)
    for path in paths:
        if path.suffix in (".py", ".md"):
            dest = out / "source_snapshot" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
    return config, data, intake, original, lock


def arrivals(case, intake, original):
    indices = list(range(case["point"], case["point"] + 1024))
    if case["role"] == "six_stream":
        stream = next(s for s in intake["streams"] if s["name"] == case["stream"])
        if indices[-1] >= len(stream["periods"]) * 1024:
            raise ValueError("Continuation exceeds available history")
        return gather(stream, {c["clip_id"]: c for c in intake["clips"]}, indices)
    clips = original["clips"]
    arrays = {
        k: np.load(clips[k]["frames_path"], mmap_mode="r", allow_pickle=False) for k in {i // 528 for i in indices}
    }
    images = np.stack([arrays[i // 528][i % 528] for i in indices])
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255


def summarize(rows):
    forecasts = {}
    for j in (4, 8, 16):
        points = []
        groups = sorted({(r["stream"], r["checkpoint"]) for r in rows if r["role"] == "six_stream"})
        for stream, checkpoint in groups:
            selected = [r for r in rows if r["stream"] == stream and r["checkpoint"] == checkpoint]
            mean = lambda key, landmark: statistics.fmean(
                r["landmarks"][str(landmark)]["scores"]["finetune"][key] for r in selected
            )
            effect = remaining_effect(
                {k: mean(k, j) for k in ("balanced", "A", "B")}, {k: mean(k, 32) for k in ("balanced", "A", "B")}
            )
            already = remaining_effect(
                {k: mean(k, 0) for k in ("balanced", "A", "B")}, {k: mean(k, j) for k in ("balanced", "A", "B")}
            )
            point = {
                "stream": stream,
                "checkpoint": checkpoint,
                **effect,
                "already_damage": already["damage"],
                "updates": statistics.fmean(r["parent_updates"] + j for r in selected),
                "recent_loss": statistics.fmean(r["landmarks"][str(j)]["recent_loss"] for r in selected),
                "cka_drift": statistics.fmean(r["landmarks"][str(j)]["geometry"]["cka_drift"] for r in selected),
            }
            points.append(point)
        forecasts[str(j)] = report(points)
        forecasts[str(j)]["already_incurred_diagnostic"] = {
            name: fit(points, fields, "already_damage") for name, fields in MODELS.items()
        }
        forecasts[str(j)]["balanced_damage_diagnostic"] = {
            name: fit(points, fields, "balanced_damage") for name, fields in MODELS.items()
        }
    control_rows = [r for r in rows if r["role"] == "control"]
    return {
        "primary_landmark": 8,
        "forecasts": forecasts,
        "qualified": forecasts["8"]["qualified"],
        "controls": control_rows,
        "controller_trained": False,
        "independent_histories_confirmed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable run; no automatic scientific retry")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 768 * 2**20:
        raise ValueError("Insufficient storage reserve")
    start = time.perf_counter()

    def progress(**fields):
        status = {**fields, "elapsed_seconds": time.perf_counter() - start, "pid": os.getpid()}
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    setup()
    progress(stage="locking")
    config, data, intake, original, lock = load(out)
    bank = reference_images(lock["reference"])
    torch.manual_seed(11)
    initial = _method(config).cuda()
    sample = arrivals(lock["cases"][0], intake, original)
    signatures = []
    for _ in range(2):
        model, optimizer = restore(initial, config.optim, learner(Path(lock["cases"][0]["parent"])))
        for batch in sample[:128].split(32):
            optimize_current(model, optimizer, batch.cuda(), True)
        signatures.append(fingerprint(capture(model, optimizer)))
    if signatures[0] != signatures[1]:
        raise ValueError("Preflight nondeterministic")
    atomic_json(out / "preflight.json", {"passed": True, "discarded_updates": 8})
    labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
    rows, clone_times, observer_times, update_times, geometry_times = [], [], [], [], []
    for case in lock["cases"]:
        batch_images = arrivals(case, intake, original)
        base = out / "runs" / case["role"] / case["stream"] / case["checkpoint"] / f"seed_{case['seed']}"
        base.mkdir(parents=True, exist_ok=True)
        tick = time.perf_counter()
        model, optimizer = restore(initial, config.optim, learner(Path(case["parent"])))
        synchronize(torch.device("cuda"))
        clone_times.append(time.perf_counter() - tick)
        trace, checkpoints, arrays = [], {}, {}
        baseline = None
        for j in range(33):
            if j in LANDMARKS:
                if shutil.disk_usage(out).free < 256 * 2**20:
                    raise ValueError("Storage reserve threatened before checkpoint write")
                marker = fingerprint(capture(model, optimizer))
                tick = time.perf_counter()
                z = embeddings(model, bank)
                synchronize(torch.device("cuda"))
                observer_times.append(time.perf_counter() - tick)
                if fingerprint(capture(model, optimizer)) != marker:
                    raise ValueError("Geometry observer changed complete state")
                if j == 0:
                    baseline = z
                arrays[f"at_{j}"] = z
                tick = time.perf_counter()
                metrics = changes(baseline, z)
                geometry_times.append(time.perf_counter() - tick)
                source = alias(case, j)
                if source:
                    if fingerprint(learner(source / "state.pt")) != marker:
                        raise ValueError("Aliased readout state differs")
                    source = readout_source(source)
                else:
                    source = base / f"at_{j}"
                    evaluate_gpu(model, optimizer, data, source)
                if fingerprint(capture(model, optimizer)) != marker:
                    raise ValueError("Readout changed online state")
                checkpoints[str(j)] = {
                    "fingerprint": marker,
                    "geometry": metrics,
                    "scores": scores(source, Path("."), labels, domains),
                    "readout": str(source),
                    "aliased": alias(case, j) is not None,
                    "recent_loss": statistics.fmean(t[0] for t in trace[-4:]) if trace else None,
                }
            if j < 32:
                tick = time.perf_counter()
                outcome = optimize_current(model, optimizer, batch_images[j * 32 : (j + 1) * 32].cuda(), True)
                synchronize(torch.device("cuda"))
                update_times.append(time.perf_counter() - tick)
                trace.append(list(outcome))
        expected = read(Path(case["training"]))
        expected_trace = (
            [r["loss_gradient"] for r in expected["records"][:16]]
            if case["role"] == "six_stream"
            else [v for block in expected["blocks"] for v in block["loss_gradients"]][:32]
        )
        if trace[: len(expected_trace)] != expected_trace:
            raise ValueError("Trajectory disagrees with audited accepted updates")
        np.savez_compressed(base / "embeddings.npz", **arrays)
        # Replay excludes offline readouts and reconstructs every landmark independently.
        model, optimizer = restore(initial, config.optim, learner(Path(case["parent"])))
        for j in range(33):
            if j in LANDMARKS:
                if fingerprint(capture(model, optimizer)) != checkpoints[str(j)]["fingerprint"]:
                    raise ValueError("Complete landmark replay differs")
                if not np.array_equal(embeddings(model, bank), arrays[f"at_{j}"]):
                    raise ValueError("Landmark embedding replay differs")
            if j < 32:
                if (
                    list(optimize_current(model, optimizer, batch_images[j * 32 : (j + 1) * 32].cuda(), True))
                    != trace[j]
                ):
                    raise ValueError("Loss/gradient replay differs")
        row = {
            **case,
            "parent_updates": case["point"] // (64 if case["role"] == "six_stream" else 32),
            "landmarks": checkpoints,
            "trace": trace,
            "embeddings_path": str(base / "embeddings.npz"),
            "exact_replay": True,
        }
        rows.append(row)
        atomic_json(out / "collection.json", {"complete": len(rows) == 63, "records": rows})
        if shutil.disk_usage(out).free < 256 * 2**20:
            raise ValueError("Storage reserve threatened")
        progress(stage="collection_and_replay", completed=len(rows), total=63)
    result = summarize(rows)
    result["costs"] = {
        "restore_seconds_per_trajectory": statistics.fmean(clone_times),
        "reference_forward_seconds": statistics.fmean(observer_times),
        "geometry_seconds": statistics.fmean(geometry_times),
        "ssl_update_seconds": statistics.fmean(update_times),
        "offline_readout_seconds_total": sum(
            read(Path(p["readout"]) / "readout.json")["readout_seconds"]
            for r in rows
            for p in r["landmarks"].values()
            if not p["aliased"]
        ),
    }
    atomic_json(out / "summary.json", result)
    progress(stage="final_hash_audit", completed=63)
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Locked inputs changed")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 128 * 2**20:
        raise ValueError("Output storage cap exceeded")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "scientific_updates": 2016,
            "replay_updates": 2016,
            "preflight_updates": 8,
            "new_primary_fits": 567,
            "reused_primary_fits": 378,
            "complete_state_replays": 315,
            "embedding_replays": 315,
            "observer_invariance": True,
            "locked_inputs": len(lock["hashes"]),
            "output_mib": size / 2**20,
        },
    )
    progress(stage="complete", qualified=result["qualified"])


if __name__ == "__main__":
    main()
