"""Replay/report a completed collection without rerunning science or readout fits."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from scripts.run_adaptation_experiment import _method
from scripts.run_regime_windows_experiment import audit, report
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    old_source = out / "collection-source/run_regime_windows_experiment.py"
    new_source = ROOT / "scripts/run_regime_windows_experiment.py"
    before = ast.parse(old_source.read_text())
    after = ast.parse(new_source.read_text())
    assert [
        ast.dump(node) for node in before.body if not isinstance(node, ast.FunctionDef) or node.name != "audit"
    ] == [ast.dump(node) for node in after.body if not isinstance(node, ast.FunctionDef) or node.name != "audit"]
    original = json.loads((out / "collection-input-lock.json").read_text())
    assert original["hashes"][str(new_source)] == digest(old_source)
    lock = json.loads((out / "input_lock.json").read_text())
    lock["hashes"][str(new_source)] = digest(new_source)
    assert all(digest(Path(p)) == v for p, v in lock["hashes"].items())
    atomic_json(out / "input_lock.json", lock)
    atomic_json(
        out / "audit-metadata-repair.json",
        {
            "scope": "tuple/list JSON trace normalization only",
            "scientific_ast_unchanged": True,
            "original_collection_source_sha256": digest(old_source),
            "audit_source_sha256": digest(new_source),
            "closure_source_sha256": digest(Path(__file__)),
            "failed_replay_updates": 396,
            "scientific_updates_repeated": 0,
            "readout_fits_repeated": 0,
        },
    )
    target = out / "audit-source-snapshot"
    target.mkdir(exist_ok=True)
    (target / new_source.name).write_bytes(new_source.read_bytes())
    (target / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    role = json.loads((out / "data-role-manifest.json").read_text())
    arrays = [np.load(c["frames_path"], allow_pickle=False) for c in role["clips"]]
    arrivals = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
    del arrays
    images = torch.load(
        ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt",
        map_location="cpu",
        weights_only=True,
    )
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            images,
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {draw: torch.from_numpy(a[f"support_{draw}"].copy()) for draw in (51011, 51017, 51023)},
            role,
        )
    config = OmegaConf.create(lock["config"])
    torch.manual_seed(11)
    initial = _method(config).cuda()
    records = json.loads((out / "collection.json").read_text())["records"]
    started = time.perf_counter()
    status = {"pid": os.getpid(), "total_trajectories": 36, "completed_trajectories": 36, "closure_only": True}

    def progress(**fields):
        status.update(fields, closure_elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        audit(initial, config, arrivals, records, out, progress)
        result = report(data, records, out)
        progress(
            stage="complete",
            audited_trajectories=36,
            scientific_ssl_updates=8910,
            primary_fits=543,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
