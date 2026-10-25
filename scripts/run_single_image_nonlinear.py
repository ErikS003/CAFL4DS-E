"""Collect six-view expected health effects on the unchanged singleton stream."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import capture, restore
from cafl4ds.single_image_forecast import HEALTH, encode, fixed_reconstruction
from cafl4ds.single_image_nonlinear import view_seed
from cafl4ds.single_image_signal import seeded_step, training_seed, zero_step
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_signal import fresh, load_stream, read

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/single-image-signal/20261021-followup"
CONFIG = ROOT / "cafl4ds/configs/single_image_nonlinear.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-nonlinear-frozen.md"


def health_read(model, bank, warm, seed):
    z = encode(model, bank, seed)
    reconstruction = fixed_reconstruction(model, bank, seed)
    values = health_values(z, warm)
    values["feature_norm"] = float(np.linalg.norm(z.astype(float), axis=1).mean())
    values["bank_reconstruction"] = float(reconstruction.mean())
    return np.array([values[k] for k in HEALTH]), z, reconstruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "stage": "inputs", "parents": 0}

    def progress(**kwargs):
        status.update(kwargs, seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        logger.info("{}", status)

    try:
        setup()
        settings = OmegaConf.to_container(OmegaConf.load(CONFIG))
        prior_lock = read(PRIOR / "input_lock.json")
        histories = {g: m for g, m in prior_lock["histories"].items() if m["role"] == "held"}
        earlier = [
            PRIOR,
            ROOT / "outputs/single-image-impact/20261021-direct-effect",
            ROOT / "outputs/single-image-expectation/20261021-views-readers",
        ]
        for root in earlier:
            for name, expected in read(root / "artifact_hashes.json")["sha256"].items():
                if digest(root / name) != expected:
                    raise ValueError(f"Earlier artifact changed: {root / name}")
        for p, expected in prior_lock["hashes"].items():
            if digest(Path(p)) != expected:
                raise ValueError(f"Frozen original input changed: {p}")
        sources = [CONFIG, PROTOCOL, Path(__file__).resolve(), ROOT / "cafl4ds/single_image_nonlinear.py"]
        inputs = [PRIOR / "input_lock.json", PRIOR / "bank.pt", PRIOR / "warm_bank.npy"]
        inputs += [root / "artifact_hashes.json" for root in earlier]
        inputs += [
            PRIOR / "trajectories" / g / f for g in histories for f in ("series.json", "training.npy", "final_state.pt")
        ]
        lock = {
            "settings": settings,
            "prior": str(PRIOR),
            "prior_lock": prior_lock,
            "histories": histories,
            "hashes": {str(p): digest(p) for p in sources + inputs},
            "prior_studies": [str(p) for p in earlier],
            "earlier_sparse_averaged_labels_inspected": True,
        }
        atomic_json(out / "input_lock.json", lock)
        for p in sources:
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        config = OmegaConf.create(prior_lock["learner_config"])
        initial = _method(config).cuda()
        bank = torch.load(PRIOR / "bank.pt", weights_only=True)
        warm = np.load(PRIOR / "warm_bank.npy")
        origins = set(range(settings["origin_start"], settings["origin_stop"] + 1, settings["origin_stride"]))
        records, replays = [], {}
        for g, meta in histories.items():
            series = read(PRIOR / "trajectories" / g / "series.json")
            trace = np.load(PRIOR / "trajectories" / g / "training.npy")
            arrivals = load_stream(meta, 96)
            model, optimizer = fresh(initial, config, 5003)
            checkpoints = 0
            for t in range(len(arrivals) + 1):
                if str(t) in series["fingerprints"]:
                    assert fingerprint(capture(model, optimizer)) == series["fingerprints"][str(t)]
                    checkpoints += 1
                if t in origins:
                    parent = capture(model, optimizer)
                    values, features, reconstructions, branch_fps, trains, seeds = [], [], [], [], [], []
                    for d in range(-1, settings["views"]):
                        branch, optim = restore(initial, config.optim, parent)
                        if d == -1:
                            zero_step(branch, optim)
                        else:
                            seed = view_seed(settings["view_seed_namespace"], g, t, d)
                            seeds.append(seed)
                            trains.append(seeded_step(branch, optim, arrivals[t : t + 1].cuda(), seed))
                        before = fingerprint(capture(branch, optim))
                        value, z, reconstruction = health_read(branch, bank, warm, settings["assay_seed"])
                        assert fingerprint(capture(branch, optim)) == before
                        values.append(value)
                        features.append(z)
                        reconstructions.append(reconstruction)
                        branch_fps.append(before)
                        del branch, optim
                    torch.set_rng_state(parent["cpu_rng"].cpu())
                    torch.cuda.set_rng_state_all([v.cpu() for v in parent["cuda_rng"]])
                    assert fingerprint(capture(model, optimizer)) == fingerprint(parent)
                    target = out / "raw" / g / f"{t}.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        target,
                        values=np.array(values),
                        bank_features=np.array(features),
                        bank_reconstruction=np.array(reconstructions),
                        training=np.array(trains),
                        seeds=np.array(seeds),
                    )
                    records.append(
                        {
                            "history": g,
                            "origin": t,
                            "raw": str(target.relative_to(out)),
                            "parent_fingerprint": fingerprint(parent),
                            "branch_fingerprints": branch_fps,
                        }
                    )
                    atomic_json(out / "collection.json", {"records": records, "replays": replays})
                    progress(stage="collection", history=g, origin=t, parents=len(records))
                    del parent
                if t == len(arrivals):
                    break
                actual = seeded_step(model, optimizer, arrivals[t : t + 1].cuda(), training_seed(g, t))
                np.testing.assert_array_equal(actual, trace[t])
            final = torch.load(PRIOR / "trajectories" / g / "final_state.pt", map_location="cpu", weights_only=True)
            assert fingerprint(capture(model, optimizer)) == fingerprint(final)
            replays[g] = {"updates": len(arrivals), "checkpoints": checkpoints, "final_matches": True}
            atomic_json(out / "collection.json", {"records": records, "replays": replays})
            del model, optimizer, arrivals
        for p, expected in lock["hashes"].items():
            assert digest(Path(p)) == expected, p
        atomic_json(
            out / "collection-verification.json",
            {
                "passed": True,
                "parents": len(records),
                "actual_branches": len(records) * settings["views"],
                "zero_branches": len(records),
                "replays": replays,
            },
        )
        progress(stage="collection_complete")
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
