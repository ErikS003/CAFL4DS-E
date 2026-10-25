"""Collect matched-dose temporal grouping trajectories with resumable exact replay."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import embeddings
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.recurrence import Evaluation
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.temporal_batch_grouping import ARMS, SEEDS, batches, manipulation, summarize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import learner, reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import DRAWS, evaluate_gpu
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_scheduling_experiment import scores
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/adaptive-mixture/20261014-gpu"
COVER = ROOT / "outputs/coverage-admission/20261013-gpu"
TIME = ROOT / "outputs/prediction-health-timeseries/20261014-pilot"


def read(path: Path) -> Any:
    """Read finite research artifacts."""
    return json.loads(path.read_text(encoding="utf-8"))


def committed(directory: Path) -> dict | None:
    """A trajectory can be reused only after completed replay and file-hash checks."""
    path = directory / "trajectory.json"
    if not path.exists():
        return None
    row = read(path)
    if not row.get("complete") or not row.get("exact_replay"):
        raise ValueError("Incomplete trajectory cannot be reused")
    if any(digest(Path(p)) != h for p, h in row["artifact_hashes"].items()):
        raise ValueError("Committed trajectory artifact changed")
    return row


def inputs(out: Path) -> tuple[Any, Any, dict, dict, dict]:
    """Pin the protocol, all helpers, audited controls and fixed health bank."""
    previous = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "independent-verification.json")["passed"]:
        raise ValueError("Mixture control audit failed")
    intake_path = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"
    intake = read(intake_path)
    reference = read(TIME / "input_lock.json")["reference"]
    old_rows = [read(Path(p)) for p in read(PRIOR / "collection.json")["records"]]
    controls = {
        (r["stream"], r["seed"], "chronological" if r["arm"] == "fixed" else "current"): r
        for r in old_rows
        if r["arm"] in ("fixed", "current")
    }
    if not (out / "input_lock.json").exists():
        paths = [
            Path(__file__),
            ROOT / "scripts/verify_temporal_batch_grouping.py",
            ROOT / "tests/unit/test_temporal_batch_grouping.py",
            ROOT / "docs/project-plan/temporal-batch-grouping-frozen.md",
            intake_path,
            PRIOR / "collection.json",
            PRIOR / "independent-verification.json",
            PRIOR / "dependency-verification.json",
            TIME / "input_lock.json",
        ]
        paths += [
            ROOT / p
            for p in (
                "cafl4ds/temporal_batch_grouping.py",
                "cafl4ds/adaptive_mixture.py",
                "cafl4ds/short_buffer.py",
                "cafl4ds/update_filtering.py",
                "cafl4ds/health_calibration.py",
                "cafl4ds/prediction_benchmark.py",
                "cafl4ds/artifacts.py",
                "scripts/run_adaptation_experiment.py",
                "scripts/run_health_measurement.py",
                "scripts/run_paired_update_response.py",
                "scripts/run_regime_windows_experiment.py",
                "scripts/run_retention_anchor_signal.py",
                "scripts/run_scheduling_experiment.py",
                "scripts/run_short_buffer_experiment.py",
                "scripts/verify_adaptive_mixture.py",
                "scripts/verify_prediction_pilots.py",
            )
        ]
        paths += [
            Path(p) for p in read(PRIOR / "collection.json")["records"] if read(Path(p))["arm"] in ("fixed", "current")
        ]
        for r in controls.values():
            for endpoint in r["readouts"].values():
                paths += [Path(endpoint["directory"]) / name for name in ("predictions.npz", "readout.json")]
        paths += [Path(row["path"]) for row in reference["images"]]
        hashes = {**previous["hashes"], **{str(p): digest(p) for p in paths}}
        lock = {
            "hashes": hashes,
            "config": previous["config"],
            "streams": intake["streams"],
            "reference": reference,
            "role": "reused_source_mechanism_development",
            "seeds": SEEDS,
            "arms": ARMS,
        }
        for p in paths:
            if p.suffix in (".py", ".md"):
                dest = out / "source_snapshot" / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
        atomic_json(out / "input_lock.json", lock)
    lock = read(out / "input_lock.json")
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Frozen experiment inputs changed")
    for row in reference["images"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Reference image changed")
    ancestor = ROOT / "outputs/regime-windows/20261013-gpu"
    with np.load(ancestor / "evaluation_arrays.npz") as a:
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
    return OmegaConf.create(lock["config"]), data, intake, lock, controls


def main() -> None:
    """Never change the frozen policy or recompute a committed trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 512 * 2**20:
        raise ValueError("Launch reserve")
    setup()
    start = time.perf_counter()

    def progress(**fields: Any) -> None:
        atomic_json(
            out / "status.json",
            {**fields, "elapsed_seconds_this_process": time.perf_counter() - start, "pid": os.getpid()},
        )
        print(json.dumps(fields), flush=True)

    progress(stage="locking")
    config, data, intake, lock, controls = inputs(out)
    if (out / "verification.json").exists():
        raise ValueError("Completed experiment is immutable")
    clips = {c["clip_id"]: c for c in intake["clips"]}
    bank = reference_images(lock["reference"])
    labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
    torch.manual_seed(11)
    initial = _method(config).cuda()
    warm_states = {s: learner(COVER / "warm" / f"seed_{s}/state.pt") for s in SEEDS}
    warm_scores = {str(s): scores(COVER / "warm" / f"seed_{s}", Path("."), labels, domains) for s in SEEDS}
    atomic_json(out / "warm_scores.json", warm_scores)
    model, optimizer = restore(initial, config.optim, warm_states[SEEDS[0]])
    marker = fingerprint(capture(model, optimizer))
    warm_z = embeddings(model, bank)
    if marker != fingerprint(capture(model, optimizer)):
        raise ValueError("Warm health observer changed learner")
    warm_path = out / "warm_bank.npz"
    if warm_path.exists():
        with np.load(warm_path) as a:
            if not np.array_equal(a["embeddings"], warm_z):
                raise ValueError("Recovered warm bank differs")
    else:
        np.savez_compressed(warm_path, embeddings=warm_z)
    if not (out / "preflight.json").exists():
        markers = []
        for _ in range(2):
            model, optimizer = restore(initial, config.optim, warm_states[SEEDS[0]])
            for r in batches(intake["streams"][0]["name"], SEEDS[0], "current")[:8]:
                optimize_current(model, optimizer, gather(intake["streams"][0], clips, r["current"]).cuda(), True)
            markers.append(fingerprint(capture(model, optimizer)))
        if markers[0] != markers[1]:
            raise ValueError("Preflight replay differs")
        atomic_json(out / "preflight.json", {"passed": True, "updates": 16})
    rows, calibration = [], {}
    for stream in intake["streams"]:
        changes = [i * 1024 for i in range(1, 12) if stream["pattern"][i] != stream["pattern"][i - 1]]
        clocks = {changes[1]: "before_return", 12288: "terminal"}
        for seed in SEEDS:
            for arm in ARMS:
                base = out / "runs" / stream["name"] / f"seed_{seed}" / arm
                existing = committed(base)
                if existing:
                    rows.append(existing)
                    calibration[stream["name"], seed] = existing["calibration_fingerprint"]
                    progress(stage="reused_committed", completed=len(rows), total=72)
                    continue
                if base.exists():
                    target = out / "interrupted" / f"{stream['name']}_{seed}_{arm}_{time.time_ns()}"
                    target.parent.mkdir(exist_ok=True)
                    if not base.resolve().is_relative_to(out) or not target.resolve().is_relative_to(out):
                        raise ValueError("Recovery path escapes output")
                    base.rename(target)
                    atomic_json(
                        target / "recovery_note.json",
                        {
                            "reason": "Interrupted trajectory preserved; restarted from unchanged warm parent",
                            "additional_updates_upper_bound": 384,
                        },
                    )
                base.mkdir(parents=True)
                if shutil.disk_usage(out).free < 256 * 2**20:
                    raise ValueError("Trajectory storage reserve")
                progress(stage="training", stream=stream["name"], seed=seed, arm=arm, completed=len(rows), total=72)
                tick = time.perf_counter()
                plan = batches(stream["name"], seed, arm)
                costs = {
                    "plan_seconds": time.perf_counter() - tick,
                    "gather_seconds": 0.0,
                    "batch_seconds": 0.0,
                    "ssl_seconds": 0.0,
                    "health_seconds": 0.0,
                    "restore_seconds": 0.0,
                    "offline_readout_seconds": 0.0,
                }
                tick = time.perf_counter()
                model, optimizer = restore(initial, config.optim, warm_states[seed])
                synchronize(torch.device("cuda"))
                costs["restore_seconds"] += time.perf_counter() - tick
                trace, health, readouts, endpoints = [], {}, {}, {}
                current_pool, pool_start = None, -1
                for r in plan:
                    step, clock = r["step"], (r["step"] + 1) * 64
                    if step % 8 == 0:
                        pool_start = step * 64
                        tick = time.perf_counter()
                        current_pool = gather(stream, clips, list(range(pool_start, pool_start + 512)))
                        costs["gather_seconds"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    current = current_pool[np.array(r["current"]) - pool_start]
                    if r["old"]:
                        old = gather(stream, clips, r["old"])
                        selected = torch.cat((old, current))
                    else:
                        selected = current
                    costs["batch_seconds"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    outcome = list(optimize_current(model, optimizer, selected.cuda(), True))
                    synchronize(torch.device("cuda"))
                    costs["ssl_seconds"] += time.perf_counter() - tick
                    trace.append({**r, "loss_gradient": outcome, "manipulation": manipulation(r)})
                    if (
                        arm in ("current", "chronological")
                        and outcome != controls[stream["name"], seed, arm]["trace"][step]["loss_gradient"]
                    ):
                        raise ValueError("Cached control training differs")
                    if step == 7:
                        value = fingerprint(capture(model, optimizer))
                        key = stream["name"], seed
                        if key in calibration and value != calibration[key]:
                            raise ValueError("Common calibration parent differs")
                        calibration[key] = value
                    if clock in (*clocks, 512):
                        name = "calibration" if clock == 512 else clocks[clock]
                        marker = fingerprint(capture(model, optimizer))
                        tick = time.perf_counter()
                        z = embeddings(model, bank)
                        costs["health_seconds"] += time.perf_counter() - tick
                        if fingerprint(capture(model, optimizer)) != marker:
                            raise ValueError("Health observer changed learner")
                        path = base / f"{name}_bank.npz"
                        np.savez_compressed(path, embeddings=z)
                        health[name] = {
                            "path": str(path),
                            "values": health_values(z, warm_z),
                            "fingerprint": marker,
                            "clock": clock,
                        }
                        endpoints[clock] = marker
                        if clock in clocks:
                            if arm in ("current", "chronological"):
                                previous = controls[stream["name"], seed, arm]["readouts"][name]
                                if marker != previous["fingerprint"]:
                                    raise ValueError("Cached control complete state differs")
                                directory, aliased = Path(previous["directory"]), True
                            else:
                                directory, aliased = base / name, False
                                tick = time.perf_counter()
                                evaluate_gpu(model, optimizer, data, directory)
                                costs["offline_readout_seconds"] += time.perf_counter() - tick
                            readouts[name] = {
                                "directory": str(directory),
                                "aliased": aliased,
                                "fingerprint": marker,
                                "scores": scores(directory, Path("."), labels, domains),
                            }
                # Complete state replay has no intervening observations or readouts.
                repeated, opt = restore(initial, config.optim, warm_states[seed])
                for r in plan:
                    if r["step"] % 8 == 0:
                        atomic_json(
                            out / "status.json",
                            {
                                "stage": "replay",
                                "completed": len(rows),
                                "total": 72,
                                "stream": stream["name"],
                                "seed": seed,
                                "arm": arm,
                                "step": r["step"],
                                "elapsed_seconds_this_process": time.perf_counter() - start,
                            },
                        )
                    actual = list(
                        optimize_current(repeated, opt, gather(stream, clips, r["old"] + r["current"]).cuda(), True)
                    )
                    if actual != trace[r["step"]]["loss_gradient"]:
                        raise ValueError("Exact replay training differs")
                    clock = (r["step"] + 1) * 64
                    if clock in endpoints:
                        if fingerprint(capture(repeated, opt)) != endpoints[clock]:
                            raise ValueError("Complete replay state differs")
                        name = "calibration" if clock == 512 else clocks[clock]
                        with np.load(health[name]["path"]) as saved:
                            if not np.array_equal(embeddings(repeated, bank), saved["embeddings"]):
                                raise ValueError("Replay bank embeddings differ")
                paths = [Path(h["path"]) for h in health.values()]
                for r in readouts.values():
                    paths += [Path(r["directory"]) / f for f in ("predictions.npz", "readout.json")]
                row = {
                    "complete": True,
                    "stream": stream["name"],
                    "seed": seed,
                    "arm": arm,
                    "trace": trace,
                    "health": health,
                    "readouts": readouts,
                    "costs": costs,
                    "calibration_fingerprint": calibration[stream["name"], seed],
                    "exact_replay": True,
                    "artifact_hashes": {str(p): digest(p) for p in paths},
                }
                atomic_json(base / "trajectory.json", row)
                rows.append(row)
                atomic_json(
                    out / "collection.json",
                    {
                        "complete": len(rows) == 72,
                        "records": [
                            str(out / "runs" / r["stream"] / f"seed_{r['seed']}" / r["arm"] / "trajectory.json")
                            for r in rows
                        ],
                    },
                )
                progress(stage="committed", completed=len(rows), total=72)
    summary = summarize(rows, warm_scores)
    atomic_json(out / "summary.json", summary)
    progress(stage="final_hash_audit", completed=72, total=72)
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Frozen sources changed during collection")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 96 * 2**20:
        raise ValueError("Output cap")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "trajectories": 72,
            "new_grouping_updates": 6912,
            "cached_control_reconstruction_updates": 6912,
            "replay_updates": 13824,
            "preflight_updates": 16,
            "planned_optimizer_updates": 27664,
            "new_primary_fits": 216,
            "cached_control_fits": 216,
            "unique_cached_warm_fits": 9,
            "process_seconds": time.perf_counter() - start,
            "output_mib": size / 2**20,
            "interrupted_attempts": len(list((out / "interrupted").glob("*"))) if (out / "interrupted").exists() else 0,
        },
    )
    progress(stage="complete", structured_filter_qualified=summary["structured_filter_qualified"])


if __name__ == "__main__":
    main()
