"""Time canonical learner replays in independent processes before a larger study."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from scripts.run_collapse_risk_regression import digest, read
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_alignment_measurement import fit_probabilities

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/target-alignment-measurement/20261015-gpu"
OUT = ROOT / "outputs/target-precision-budget/20261015-preflight"


def worker(index):
    setup()
    images = torch.load(SOURCE / "images.pt", weights_only=True)
    with np.load(SOURCE / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    lock = read(SOURCE / "input_lock.json")
    encoder = instantiate(OmegaConf.create(lock["model_config"])["encoder"]).cuda()
    base = SOURCE / "runs/fresh_00/at_128"
    encoder.load_state_dict(torch.load(base / "encoder.pt", weights_only=True))
    seeds = lock["config"]["support_seeds"][:4] * 2
    query = np.concatenate([arrays["query"], arrays["legacy_query"]])
    start = time.perf_counter()
    for number, seed in enumerate(seeds):
        prediction, _ = fit_probabilities(
            encoder, images, torch.from_numpy(arrays["labels"]), arrays[f"support_{seed}"], query, seed
        )
        with np.load(base / f"panel_{seed}.npz") as expected:
            assert np.array_equal(prediction, expected["probabilities"])
        print({"worker": index, "exact_fit": number + 1}, flush=True)
    result = {
        "worker": index,
        "fits": 8,
        "exact_replays": True,
        "compute_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(OUT / f"worker_{index}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        worker(args.worker)
        return
    assert not (OUT / "summary.json").exists()
    paths = [
        Path(__file__),
        SOURCE / "input_lock.json",
        SOURCE / "independent-verification.json",
        SOURCE / "images.pt",
        SOURCE / "data.npz",
        SOURCE / "runs/fresh_00/at_128/encoder.pt",
    ]
    paths += sorted((SOURCE / "runs/fresh_00/at_128").glob("panel_*.npz"))
    atomic_json(
        OUT / "input-lock.json",
        {
            "hashes": {str(p): digest(p) for p in paths},
            "design": "8 serial plus 32 four-process exact old-fit replays; choose concurrency by compute throughput",
        },
    )
    serial = worker(-1)
    processes, handles = [], []
    for index in range(4):
        handle = (OUT / f"worker_{index}.log").open("w")
        handles.append(handle)
        processes.append(
            subprocess.Popen(
                [sys.executable, "-u", "-m", "scripts.benchmark_target_precision_workers", "--worker", str(index)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        )
    codes = [p.wait() for p in processes]
    for handle in handles:
        handle.close()
    assert codes == [0] * 4, "Preflight failed; no new outcomes may launch"
    parallel = [read(OUT / f"worker_{i}.json") for i in range(4)]
    speedup = 4 * serial["compute_seconds"] / max(p["compute_seconds"] for p in parallel)
    result = {
        "passed": True,
        "serial": serial,
        "parallel": parallel,
        "warm_throughput_speedup": speedup,
        "selected_workers": 4 if speedup >= 1.2 else 1,
        "replayed_classifier_fits": 40,
        "supervised_steps": 4000,
        "new_outcomes": False,
    }
    atomic_json(OUT / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
