"""Run adaptive replay timing with exact dose controls and independent GPU replay."""

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

from cafl4ds.adaptive_mixture import ARMS, SEEDS, Mixture, interval, memory_plan, stream_seed
from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import embeddings
from cafl4ds.recurrence import Evaluation
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import learner
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import DRAWS, evaluate_gpu
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_scheduling_experiment import scores
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/coverage-admission/20261013-gpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(out):
    old = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "independent-verification.json")["passed"]:
        raise ValueError("Coverage audit failed")
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    paths = [
        Path(__file__),
        ROOT / "scripts/verify_adaptive_mixture.py",
        ROOT / "cafl4ds/adaptive_mixture.py",
        ROOT / "tests/unit/test_adaptive_mixture.py",
        ROOT / "docs/project-plan/slow-edge-adaptive-mixture-frozen.md",
        PRIOR / "independent-verification.json",
    ]
    for seed in SEEDS:
        paths.extend(PRIOR / "warm" / f"seed_{seed}" / name for name in ("state.pt", "predictions.npz", "readout.json"))
    for stream in intake["streams"]:
        for seed in SEEDS:
            base = PRIOR / "runs" / stream["name"] / "uniform" / f"seed_{seed}"
            paths.append(base / "training.json")
            paths.extend(base / "at_12288" / name for name in ("state.pt", "predictions.npz", "readout.json"))
    hashes = {**old["hashes"], **{str(p): digest(p) for p in paths}}
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Inherited sources changed")
    lock = {
        "hashes": hashes,
        "config": old["config"],
        "streams": intake["streams"],
        "seeds": SEEDS,
        "arms": ARMS,
        "role": "reused_development",
        "arrivals": 12288,
        "updates_per_trajectory": 192,
    }
    atomic_json(out / "input_lock.json", lock)
    for path in paths[:5]:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
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
    return OmegaConf.create(old["config"]), data, intake, lock


def summarize(rows, warm):
    names = sorted({r["stream"] for r in rows})
    horizons = {}
    for horizon in ("before_return", "terminal"):
        means, contrasts = {}, {}
        for arm in (*ARMS, "warm"):
            means[arm] = {
                reader: {
                    metric: statistics.fmean(
                        warm[seed][reader][metric]
                        if arm == "warm"
                        else next(r for r in rows if r["stream"] == n and r["seed"] == seed and r["arm"] == arm)[
                            "readouts"
                        ][horizon]["scores"][reader][metric]
                        for n in names
                        for seed in SEEDS
                    )
                    for metric in ("balanced", "A", "B")
                }
                for reader in ("finetune", "linear", "knn")
            }
        for comparator in ("current", "fixed", "random", "warm"):
            contrasts[comparator] = {}
            for reader in ("finetune", "linear", "knn"):
                contrasts[comparator][reader] = {}
                for metric in ("balanced", "A", "B"):
                    delta = []
                    for name in names:

                        def mean(arm):
                            return statistics.fmean(
                                warm[seed][reader][metric]
                                if arm == "warm"
                                else next(
                                    r for r in rows if r["stream"] == name and r["seed"] == seed and r["arm"] == arm
                                )["readouts"][horizon]["scores"][reader][metric]
                                for seed in SEEDS
                            )

                        delta.append(mean("adaptive") - mean(comparator))
                    contrasts[comparator][reader][metric] = {
                        "ordinary": interval(delta, 1),
                        "adjusted": interval(delta, 12),
                    }
        horizons[horizon] = {"means": means, "contrasts": contrasts}
    terminal = horizons["terminal"]["contrasts"]
    gates = {
        "adaptive_content_lead": all(
            terminal[a]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.5
            and terminal[a]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0
            for a in ("current", "fixed", "random")
        ),
        "useful_ssl": terminal["warm"]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.1
        and terminal["warm"]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0,
        "domain_retention": all(
            terminal[a]["finetune"][m]["adjusted"]["ci95_pp"][0] >= -1
            for a in ("current", "fixed", "random", "warm")
            for m in ("A", "B")
        ),
    }
    costs = {
        arm: {key: statistics.fmean(r["costs"][key] for r in rows if r["arm"] == arm) for key in rows[0]["costs"]}
        for arm in ARMS
    }
    return {
        "horizons": horizons,
        "gates": gates,
        "adaptive_mixture_qualified": all(gates.values()),
        "costs_per_trajectory": costs,
        "history_groups": 6,
        "trajectories": len(rows),
        "fresh_confirmation": False,
        "deployment_efficiency_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable run; no automatic scientific retry")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 768 * 2**20:
        raise ValueError("Insufficient launch reserve")
    start = time.perf_counter()

    def progress(**fields):
        status = {**fields, "elapsed_seconds": time.perf_counter() - start, "pid": os.getpid()}
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    setup()
    progress(stage="locking")
    config, data, intake, lock = load(out)
    clips = {c["clip_id"]: c for c in intake["clips"]}
    torch.manual_seed(11)
    initial = _method(config).cuda()
    reference = _method(config).cuda().eval().requires_grad_(False)
    reference_id = fingerprint({k: v for k, v in reference.state_dict().items() if k.startswith("encoder.")})
    warm_states, warm_scores = {}, {}
    labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
    for seed in SEEDS:
        state = learner(PRIOR / "warm" / f"seed_{seed}/state.pt")
        if fingerprint({k: v for k, v in state["model"].items() if k.startswith("encoder.")}) != reference_id:
            raise ValueError("Frozen reference differs from warm encoder")
        warm_states[seed] = state
        warm_scores[seed] = scores(PRIOR / "warm" / f"seed_{seed}", Path("."), labels, domains)
    sample = intake["streams"][0]
    signatures = []
    for _ in range(2):
        model, optimizer = restore(initial, config.optim, warm_states[SEEDS[0]])
        for j in range(8):
            optimize_current(model, optimizer, gather(sample, clips, list(range(j * 64, j * 64 + 32))).cuda(), True)
        signatures.append(fingerprint(capture(model, optimizer)))
    if len(set(signatures)) != 1:
        raise ValueError("Preflight nondeterministic")
    atomic_json(out / "preflight.json", {"passed": True, "discarded_updates": 16})
    rows, calibration_states = [], {}
    for stream in intake["streams"]:
        pattern = stream["pattern"]
        transitions = [i * 1024 for i in range(1, len(pattern)) if pattern[i] != pattern[i - 1]]
        before_return = transitions[1]
        feature_path = out / "reference_features" / f"{stream['name']}.npy"
        saved_features = None
        for seed in SEEDS:
            tick = time.perf_counter()
            plans = memory_plan(stream["name"], seed)
            plan_seconds = time.perf_counter() - tick
            for arm in ARMS:
                progress(stage="collection", stream=stream["name"], seed=seed, arm=arm, completed=len(rows), total=72)
                base = out / "runs" / stream["name"] / arm / f"seed_{seed}"
                base.mkdir(parents=True, exist_ok=True)
                tick = time.perf_counter()
                model, optimizer = restore(initial, config.optim, warm_states[seed])
                synchronize(torch.device("cuda"))
                restore_seconds = time.perf_counter() - tick
                selector = Mixture(arm, plans, stream_seed(stream["name"], seed))
                features = np.full((12288, 96), np.nan, dtype=np.float32) if arm == "adaptive" else None
                trace, readouts, endpoints = [], {}, {}
                costs = {
                    "gather_seconds": 0.0,
                    "reference_seconds": 0.0,
                    "decision_seconds": 0.0,
                    "ssl_seconds": 0.0,
                    "reservoir_plan_seconds": plan_seconds if arm != "current" else 0.0,
                    "restore_seconds": restore_seconds,
                    "offline_readout_seconds": 0.0,
                }
                for step in range(192):
                    offset = step * 64
                    if arm == "adaptive":
                        marker = fingerprint(capture(model, optimizer))
                        tick = time.perf_counter()
                        incoming = gather(stream, clips, list(range(offset, offset + 64)))
                        costs["gather_seconds"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                        features[offset : offset + 64] = embeddings(reference, incoming)
                        synchronize(torch.device("cuda"))
                        costs["reference_seconds"] += time.perf_counter() - tick
                        if fingerprint(capture(model, optimizer)) != marker:
                            raise ValueError("Context observer changed complete learner")
                    tick = time.perf_counter()
                    choice = selector.choose(step, features)
                    costs["decision_seconds"] += time.perf_counter() - tick
                    indices = choice["old"] + choice["current"]
                    tick = time.perf_counter()
                    images = gather(stream, clips, indices)
                    costs["gather_seconds"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    outcome = optimize_current(model, optimizer, images.cuda(), True)
                    synchronize(torch.device("cuda"))
                    costs["ssl_seconds"] += time.perf_counter() - tick
                    trace.append({**choice, "loss_gradient": list(outcome)})
                    if step == 7:
                        marker = fingerprint(capture(model, optimizer))
                        key = (stream["name"], seed)
                        if key in calibration_states and marker != calibration_states[key]:
                            raise ValueError("Common calibration state differs across arms")
                        calibration_states[key] = marker
                    clock = (step + 1) * 64
                    if clock in (before_return, 12288):
                        if shutil.disk_usage(out).free < 256 * 2**20:
                            raise ValueError("Storage reserve threatened")
                        horizon = "terminal" if clock == 12288 else "before_return"
                        marker = fingerprint(capture(model, optimizer))
                        endpoints[clock] = marker
                        directory = base / f"at_{clock}"
                        aliased = False
                        if arm == "current" and clock == 12288:
                            directory = PRIOR / "runs" / stream["name"] / "uniform" / f"seed_{seed}/at_12288"
                            if marker != fingerprint(learner(directory / "state.pt")):
                                raise ValueError("Current trajectory differs from uniform reference")
                            aliased = True
                        else:
                            tick = time.perf_counter()
                            evaluate_gpu(model, optimizer, data, directory)
                            costs["offline_readout_seconds"] += time.perf_counter() - tick
                        if fingerprint(capture(model, optimizer)) != marker:
                            raise ValueError("Readout changed complete learner")
                        readouts[horizon] = {
                            "clock": clock,
                            "directory": str(directory),
                            "aliased": aliased,
                            "fingerprint": marker,
                            "scores": scores(directory, Path("."), labels, domains),
                        }
                if arm == "current":
                    old = read(PRIOR / "runs" / stream["name"] / "uniform" / f"seed_{seed}/training.json")
                    old_trace = [v for block in old["blocks"] for v in block["loss_gradients"]]
                    if old_trace != [r["loss_gradient"] for r in trace]:
                        raise ValueError("Current trace differs from audited uniform input sequence")
                if features is not None:
                    if saved_features is None:
                        saved_features = features.copy()
                        feature_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(feature_path, features)
                    elif not np.array_equal(saved_features, features):
                        raise ValueError("Warm reference embeddings differ across seeds")
                # Replay exposure decisions and optimization without offline readouts.
                model, optimizer = restore(initial, config.optim, warm_states[seed])
                replay = Mixture(arm, plans, stream_seed(stream["name"], seed))
                for step in range(192):
                    chosen = replay.choose(step, features)
                    expected = {k: v for k, v in trace[step].items() if k != "loss_gradient"}
                    if chosen != expected:
                        raise ValueError("Replay selector differs")
                    outcome = optimize_current(
                        model, optimizer, gather(stream, clips, chosen["old"] + chosen["current"]).cuda(), True
                    )
                    if list(outcome) != trace[step]["loss_gradient"]:
                        raise ValueError("Replay loss/gradient differs")
                    clock = (step + 1) * 64
                    if clock in endpoints and fingerprint(capture(model, optimizer)) != endpoints[clock]:
                        raise ValueError("Complete endpoint replay differs")
                if (
                    fingerprint({k: v for k, v in reference.state_dict().items() if k.startswith("encoder.")})
                    != reference_id
                ):
                    raise ValueError("Frozen context encoder mutated")
                row = {
                    "stream": stream["name"],
                    "pattern": pattern,
                    "seed": seed,
                    "arm": arm,
                    "trace": trace,
                    "plans": plans,
                    "readouts": readouts,
                    "feature_path": str(feature_path) if arm == "adaptive" else None,
                    "costs": costs,
                    "exact_replay": True,
                    "common_calibration_fingerprint": calibration_states[(stream["name"], seed)],
                }
                rows.append(row)
                atomic_json(base / "trajectory.json", row)
                atomic_json(
                    out / "collection.json",
                    {
                        "complete": len(rows) == 72,
                        "records": [
                            str(out / "runs" / r["stream"] / r["arm"] / f"seed_{r['seed']}/trajectory.json")
                            for r in rows
                        ],
                    },
                )
                progress(stage="collection_and_replay", completed=len(rows), total=72)
    result = summarize(rows, warm_scores)
    atomic_json(out / "summary.json", result)
    progress(stage="final_hash_audit", completed=72)
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Locked inputs changed")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 128 * 2**20:
        raise ValueError("Output cap exceeded")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "trajectories": 72,
            "scientific_updates": 13824,
            "replay_updates": 13824,
            "preflight_updates": 16,
            "new_primary_fits": 378,
            "reused_primary_fits": 63,
            "complete_endpoint_replays": 144,
            "context_observer_checks": 3456,
            "locked_inputs": len(lock["hashes"]),
            "output_mib": size / 2**20,
        },
    )
    progress(stage="complete", adaptive_mixture_qualified=result["adaptive_mixture_qualified"])


if __name__ == "__main__":
    main()
