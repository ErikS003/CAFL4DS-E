"""Matched image/draw/reader counterfactuals with shared single-image futures."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import restore
from cafl4ds.single_image_signal import seeded_step, training_seed, zero_step
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_signal import load_panel, load_stream, measure, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    out = root / "diagnostic"
    out.mkdir(exist_ok=False)
    lock = read(root / "input_lock.json")
    settings = OmegaConf.create(lock["settings"])
    setup()
    config = OmegaConf.create(lock["learner_config"])
    initial = _method(config).cuda()
    panel = load_panel(root)
    bank = torch.load(root / "bank.pt", weights_only=True)
    warm = np.load(root / "warm_bank.npy")
    records = []
    work = {"actual_first_steps": 0, "zero_steps": 0, "future_ssl_steps": 0}
    started = time.perf_counter()
    origin = settings.diagnostic_origin
    for name, meta in lock["histories"].items():
        if meta["role"] != "held":
            continue
        directory = out / name
        directory.mkdir()
        arrivals = load_stream(meta, settings.frames_per_period)
        parent_path = root / "trajectories" / name / "parents" / f"{origin}.pt"
        previous_path = parent_path.with_name(f"{origin - 1}.pt")
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        previous = torch.load(previous_path, map_location="cpu", weights_only=True)
        model, optim = restore(initial, config.optim, previous)
        measure(model, optim, bank, warm, panel, directory / "previous", list(settings.reader_draws), (20, 100))
        model, optim = restore(initial, config.optim, parent)
        before = measure(model, optim, bank, warm, panel, directory / "before", list(settings.reader_draws), (20, 100))
        # Check exact identity of the established main reference reader.
        main = read(root / "trajectories" / name / "states" / str(origin) / "readout.json")
        reference = str(settings.reader_draws[0])
        if any(before["metrics"][reference][k] != v for k, v in main["metrics"][reference].items()):
            raise ValueError("Main and diagnostic reference readers differ")
        continuation = {}
        hold_model, hold_optim = restore(initial, config.optim, parent)
        continuation["hold"] = (hold_model, hold_optim)
        records.append(
            {
                "history": name,
                "arm": "hold",
                "horizon": 1,
                "directory": str((directory / "before").relative_to(root)),
                "fingerprint": fingerprint(parent),
            }
        )
        zero_model, zero_optim = restore(initial, config.optim, parent)
        zero_step(zero_model, zero_optim)
        work["zero_steps"] += 1
        zero = measure(
            zero_model, zero_optim, bank, warm, panel, directory / "zero" / "1", list(settings.reader_draws), (20, 100)
        )
        continuation["zero"] = (zero_model, zero_optim)
        records.append(
            {
                "history": name,
                "arm": "zero",
                "horizon": 1,
                "directory": str((directory / "zero" / "1").relative_to(root)),
                "fingerprint": zero["fingerprint"],
            }
        )
        for candidate, offset in enumerate(settings.candidate_offsets):
            index = origin + offset
            for draw_id, draw in enumerate(settings.diagnostic_draws):
                model, optim = restore(initial, config.optim, parent)
                train = seeded_step(model, optim, arrivals[index : index + 1].cuda(), draw)
                work["actual_first_steps"] += 1
                arm = f"image_{candidate}_draw_{draw_id}"
                branch = measure(
                    model, optim, bank, warm, panel, directory / arm / "1", list(settings.reader_draws), (20, 100)
                )
                records.append(
                    {
                        "history": name,
                        "arm": arm,
                        "candidate": candidate,
                        "index": index,
                        "draw": draw,
                        "draw_id": draw_id,
                        "horizon": 1,
                        "training": list(train),
                        "directory": str((directory / arm / "1").relative_to(root)),
                        "fingerprint": branch["fingerprint"],
                    }
                )
                if candidate == 0:
                    continuation[arm] = (model, optim)
                else:
                    del model, optim
                atomic_json(
                    out / "status.json",
                    {
                        "stage": "immediate",
                        "history": name,
                        "arm": arm,
                        "records": len(records),
                        "work": work,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
        for arm, (model, optim) in continuation.items():
            for t in range(origin + 1, origin + 32):
                seeded_step(model, optim, arrivals[t : t + 1].cuda(), training_seed(name, t))
                work["future_ssl_steps"] += 1
                horizon = t - origin + 1
                if horizon in (8, 32):
                    point = measure(
                        model,
                        optim,
                        bank,
                        warm,
                        panel,
                        directory / arm / str(horizon),
                        list(settings.reader_draws),
                        (100,),
                    )
                    records.append(
                        {
                            "history": name,
                            "arm": arm,
                            "horizon": horizon,
                            "fingerprint": point["fingerprint"],
                            "directory": str((directory / arm / str(horizon)).relative_to(root)),
                        }
                    )
            del model, optim
        atomic_json(out / "collection.json", {"records": records, "work": work})
        atomic_json(
            out / "status.json",
            {
                "stage": "collection",
                "completed_histories": len({r["history"] for r in records}),
                "history": name,
                "records": len(records),
                "work": work,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps({"completed_history": name, "records": len(records), "work": work}), flush=True)
        del continuation, arrivals
    for path, value in lock["hashes"].items():
        if digest(Path(path)) != value:
            raise ValueError(f"Scientific source/input changed: {path}")
    atomic_json(
        out / "status.json",
        {
            "stage": "collection_complete",
            "records": len(records),
            "work": work,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
