"""Fixed two-stage precision budget for the unchanged canonical downstream learner."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.functional_retention import plain
from cafl4ds.target_alignment_measurement import metric_means
from cafl4ds.target_precision_budget import query_precision_gate
from scripts.run_collapse_risk_regression import digest, read
from scripts.run_regime_windows_experiment import fine_tune_gpu
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import encode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_precision_budget.yaml"


def write(path, value):
    atomic_json(path, plain(value))


def atomic_predictions(path, probabilities):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, probabilities=probabilities)
    temporary.replace(path)


def fit_and_score(encoder, images, arrays, seed):
    """Keep the old readout batch layout for exact pilot reproduction."""
    selected = arrays[f"support_{seed}"]
    tuned, head, audit = fine_tune_gpu(
        encoder, images[selected], torch.from_numpy(arrays["labels"][selected]), seed + 25
    )

    def score(indices):
        features = encode(tuned, images[indices])
        with torch.no_grad():
            return head(features.cuda()).softmax(-1).cpu().numpy()

    original = score(np.concatenate([arrays["pilot_query"], arrays["legacy_query"]]))
    additional = score(arrays["additional_query"])
    return np.concatenate([original[:1024], additional, original[1024:]]), audit


def roster(source):
    return [
        (u["source"], int(step))
        for u in read(source / "collection.json")["units"]
        for step in sorted(u["bindings"], key=int)
    ]


def worker(index, stage):
    setup()
    config = OmegaConf.load(CONFIG)
    out, source = ROOT / config.output, ROOT / config.source
    lock, manifest = read(out / "input_lock.json"), read(out / "data-role-manifest.json")
    assert all(
        digest(p) == lock["hashes"][str(p)]
        for p in (CONFIG, Path(__file__), ROOT / "cafl4ds/target_precision_budget.py")
    )
    images = torch.load(out / "images.pt", weights_only=True)
    with np.load(out / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    seeds = manifest["support_seeds"][:stage]
    jobs = roster(source)[index :: lock["workers"]]
    journal_dir = out / "attempts"
    journal_dir.mkdir(exist_ok=True)
    attempt = len(list(journal_dir.glob(f"stage_{stage}_worker_{index}_*.json")))
    journal = journal_dir / f"stage_{stage}_worker_{index}_{attempt}.json"
    state = {
        "stage": stage,
        "worker": index,
        "pid": os.getpid(),
        "started_fits": 0,
        "completed_fits": 0,
        "reused_fits": 0,
        "complete": False,
    }
    write(journal, state)
    encoder = instantiate(OmegaConf.create(lock["model_config"]).encoder).cuda()
    for name, step in jobs:
        directory = out / "runs" / name / f"at_{step}"
        directory.mkdir(parents=True, exist_ok=True)
        source_encoder = source / "runs" / name / f"at_{step}/encoder.pt"
        encoder.load_state_dict(torch.load(source_encoder, weights_only=True))
        for position, seed in enumerate(seeds):
            path = directory / f"panel_{seed}.npz"
            audit_path = directory / f"panel_{seed}.json"
            if path.exists() and audit_path.exists():
                audit = read(audit_path)
                assert (
                    digest(path) == audit["sha256"] and audit["encoder_sha256"] == lock["hashes"][str(source_encoder)]
                )
                state["reused_fits"] += 1
                continue
            state["started_fits"] += 1
            write(journal, state)
            probability, fitted = fit_and_score(encoder, images, arrays, seed)
            if position < 12:
                old_path = source / "runs" / name / f"at_{step}/panel_{seed}.npz"
                with np.load(old_path) as original:
                    expected = original["probabilities"]
                    inherited = np.concatenate([probability[:1024], probability[-400:]])
                    assert np.array_equal(inherited, expected), (name, step, seed, "pilot replay")
            if path.exists():
                with np.load(path) as recovered:
                    assert np.array_equal(probability, recovered["probabilities"]), "Partial fit did not replay"
            else:
                atomic_predictions(path, probability)
            write(
                audit_path,
                {**fitted, "draw": seed, "encoder_sha256": lock["hashes"][str(source_encoder)], "sha256": digest(path)},
            )
            state["completed_fits"] += 1
            write(journal, state)
            if (position + 1) % 16 == 0:
                print(
                    {"worker": index, "source": name, "step": step, "stage": stage, "panels_complete": position + 1},
                    flush=True,
                )
        write(
            directory / f"complete_{stage}.json",
            {
                "stage": stage,
                "source": name,
                "step": step,
                "seeds": seeds,
                "hashes": {
                    str(directory / f"panel_{s}{suffix}"): digest(directory / f"panel_{s}{suffix}")
                    for s in seeds
                    for suffix in (".npz", ".json")
                },
            },
        )
    state["complete"] = True
    write(journal, state)


def collect_stage(config, stage):
    source, out = ROOT / config.source, ROOT / config.output
    manifest = read(out / "data-role-manifest.json")
    seeds = manifest["support_seeds"][:stage]
    with np.load(out / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    with np.load(out / f"resamples_{stage}.npz") as data:
        panel_counts, query_counts = data["panel"], data["query"]
    predictions = {}
    for name, step in roster(source):
        directory = out / "runs" / name / f"at_{step}"
        assert read(directory / f"complete_{stage}.json")["seeds"] == seeds
        values = []
        for seed in seeds:
            with np.load(directory / f"panel_{seed}.npz") as saved:
                values.append(saved["probabilities"])
        probability = np.array(values)
        atomic_predictions(directory / f"predictions_{stage}.npz", probability)
        predictions[name, step] = probability
    measurement = instantiate(config.measurement)
    records = []
    for old in read(source / "collection.json")["records"]:
        a, b = (predictions[old["source"], old["age"] + k] for k in (0, 32))
        q = arrays["query"]
        n = len(q)
        measured = measurement.compare(
            a[:, :n], b[:, :n], arrays["labels"][q], arrays["domains"][q], panel_counts, query_counts
        )
        secondary = {}
        for role, selected, part in (
            ("pilot", arrays["pilot_query"], slice(0, 1024)),
            ("legacy", arrays["legacy_query"], slice(n, None)),
        ):
            y, d = arrays["labels"][selected], arrays["domains"][selected]
            effect = 100 * ((b[:, part].argmax(-1) == y).astype(float) - (a[:, part].argmax(-1) == y))
            secondary[role + "_mean_gain_pp"] = metric_means(effect, y, d).mean(0)
        records.append(
            {
                "source": old["source"],
                "context": old["context"],
                "age": old["age"],
                "horizon": 32,
                "fresh": measured,
                **secondary,
            }
        )
    result = {
        "panels": stage,
        "records": records,
        "summary": measurement.summarize(records),
        "interim_query_gate": query_precision_gate(records),
    }
    write(out / f"stage_{stage}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--stage", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.stage)
        return
    config = OmegaConf.load(CONFIG)
    source, out = ROOT / config.source, ROOT / config.output
    assert not (out / "collection.json").exists(), "Completed science is immutable"
    assert shutil.disk_usage(out).free > 3 * 2**30
    source_lock = read(source / "input_lock.json")
    source_audit = read(source / "independent-verification.json")
    assert source_audit["passed"]
    manifest = read(out / "data-role-manifest.json")
    preflight = read(ROOT / config.preflight / "summary.json")
    assert preflight["passed"] and preflight["selected_workers"] in (1, 4)
    helper = read(ROOT / config.preflight / "helper-replay.json")
    assert helper["passed"] and helper["runner_sha256"] == digest(__file__)
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/prepare_target_precision_budget.py",
        ROOT / "scripts/verify_target_precision_budget.py",
        ROOT / "cafl4ds/target_precision_budget.py",
        ROOT / "tests/unit/test_target_precision_budget.py",
        ROOT / "docs/project-plan/target-precision-budget-frozen.md",
        out / "data-role-manifest.json",
        source / "input_lock.json",
        source / "independent-verification.json",
        ROOT / config.preflight / "summary.json",
        ROOT / config.preflight / "helper-replay.json",
        ROOT / "scripts/benchmark_target_precision_workers.py",
    ]
    hashes = {
        **source_lock["hashes"],
        **source_audit["hashes"],
        **manifest["hashes"],
        **{str(p): digest(p) for p in files},
    }
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    lock = {
        "hashes": hashes,
        "config": OmegaConf.to_container(config, resolve=True),
        "model_config": source_lock["model_config"],
        "workers": preflight["selected_workers"],
        "scope": "expanded_conditional_measurement; pilot-informed_budget_no_new_SSL_states",
    }
    if (out / "input_lock.json").exists():
        assert read(out / "input_lock.json") == plain(lock)
    else:
        write(out / "input_lock.json", lock)
        for path in files:
            if path.suffix in (".py", ".md", ".yaml"):
                target = out / "source_snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    started = time.perf_counter()
    selected_stage = None
    for stage in (config.interim_panels, config.final_panels):
        stage_file = out / f"stage_{stage}.json"
        if stage_file.exists():
            result = read(stage_file)
        else:
            handles, processes = [], []
            for index in range(lock["workers"]):
                handle = (out / f"stage_{stage}_worker_{index}.log").open("a")
                handles.append(handle)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.run_target_precision_budget",
                            "--worker",
                            str(index),
                            "--stage",
                            str(stage),
                        ],
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                )
            while any(p.poll() is None for p in processes):
                write(
                    out / "status.json",
                    {
                        "stage": stage,
                        "worker_pids": [p.pid for p in processes],
                        "workers_running": sum(p.poll() is None for p in processes),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                time.sleep(5)
            for handle in handles:
                handle.close()
            assert all(p.returncode == 0 for p in processes), "Worker failed; preserve outputs before recovery"
            result = collect_stage(config, stage)
        print(
            {"stage_complete": stage, "query_gate": result["interim_query_gate"], "gates": result["summary"]["gates"]},
            flush=True,
        )
        selected_stage = result
        if stage == config.interim_panels and not result["interim_query_gate"]["continue_to_final_budget"]:
            break
    panels = selected_stage["panels"]
    qualified = panels == config.final_panels and selected_stage["summary"]["measurement_qualified"]
    summary = {
        **selected_stage["summary"],
        "measurement_qualified": qualified,
        "panels": panels,
        "interim_query_gate": read(out / "stage_64.json")["interim_query_gate"],
        "decision": "prepare_separate_health_bridge_protocol" if qualified else "stop_before_health_fitting",
        "elapsed_seconds": time.perf_counter() - started,
        "unique_classifier_fits": 36 * panels,
        "supervised_steps": 3600 * panels,
        "preflight_replay_fits": 40,
        "helper_replay_fits": helper["fits"],
        "ssl_updates": 0,
        "query_count": 3663,
    }
    write(out / "collection.json", {"complete": True, "panels": panels, "records": selected_stage["records"]})
    write(out / "summary.json", summary)
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts) < 2 * 2**30
    write(out / "status.json", {"complete": True, "panels": panels, "decision": summary["decision"]})
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
