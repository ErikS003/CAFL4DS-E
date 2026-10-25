"""Longer singleton trajectories with original-reader, causally timed forecast observations."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture
from cafl4ds.single_image_forecast import encode, image_features
from cafl4ds.single_image_signal import (
    gradient_features,
    optimizer_features,
    read_state,
    seeded_step,
    training_seed,
    zero_step,
)
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_forecast import local_image

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/single-image-forecast/20261021-batch1"
READER = ROOT / "outputs/temporal-state-opportunity/20261011-development-v1/experiment"
CONFIG = ROOT / "cafl4ds/configs/single_image_signal.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-signal-frozen.md"
INTAKE = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(out: Path, settings) -> tuple:
    prior = read(PRIOR / "input_lock.json")
    if not read(PRIOR / "delivery.json")["passed"]:
        raise ValueError("Prior study is not audited")
    config = OmegaConf.create(prior["learner_config"])
    config.batch_size, config.stream.batch_size = 1, 1
    images = torch.load(READER / "evaluation_images.pt", map_location="cpu", weights_only=True)
    manifest = read(READER / "reader-allocation.json")
    with np.load(READER / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        labels, domains = arrays["labels"].copy(), arrays["domains"].copy()
        supports = {d: arrays[f"support_{d}"].copy() for d in settings.reader_draws}
    supports = {d: ids[labels[ids] < 2] for d, ids in supports.items()}
    query = np.array(prior["query_indices"])
    panel = {
        "support_images": {d: images[ids] for d, ids in supports.items()},
        "support_labels": {d: torch.from_numpy(labels[ids]) for d, ids in supports.items()},
        "query_images": images[query],
        "query_labels": labels[query],
        "query_domains": domains[query],
    }
    paths = [
        CONFIG,
        PROTOCOL,
        PRIOR / "input_lock.json",
        PRIOR / "delivery.json",
        INTAKE,
        Path(config.well),
        READER / "reader-allocation.json",
        READER / "evaluation_images.pt",
        READER / "evaluation_arrays.npz",
    ]
    reader_ids = set()
    for index in sorted(set(query.tolist() + np.concatenate(list(supports.values())).tolist())):
        row = manifest["selected"][index]
        path = local_image(row["image_path"])
        if digest(path) != row["image_sha256"]:
            raise ValueError("Reader image hash differs")
        with Image.open(path) as opened:
            pixels = torch.from_numpy(np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        if not torch.equal(pixels.permute(2, 0, 1).float() / 255, images[index]):
            raise ValueError("Reader pixels differ")
        attrs = row["keyframe_attributes"]
        if labels[index] != ("city street", "highway").index(attrs["scene"]) or domains[index] != (
            attrs["timeofday"] == "night"
        ):
            raise ValueError("Reader labels differ")
        reader_ids.add(row["clip_id"])
        paths.append(path)
    if len(query) != 400 or any(len(ids) != 50 for ids in supports.values()):
        raise ValueError("Unexpected reader membership")
    if any(set(ids) & set(query) for ids in supports.values()):
        raise ValueError("Support/query overlap")
    reference_path = ROOT / "outputs/health-calibration/20261013-gpu/reference.json"
    reference = read(reference_path)
    paths.append(reference_path)
    bank = []
    for row in reference["images"]:
        path = local_image(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Health source differs")
        with Image.open(path) as opened:
            bank.append(np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        paths.append(path)
    bank = torch.from_numpy(np.stack(bank)).permute(0, 3, 1, 2).float() / 255
    intake = read(INTAKE)
    clips = {c["clip_id"]: c for c in intake["clips"]}
    histories = {}
    occupied = set()
    old_ids = {r["clip_id"] for s in prior["selected"].values() for r in s["frames"]}
    for role, member in (("development", 0), ("held", 1)):
        for stream in intake["streams"]:
            ids = [p[member] for p in stream["periods"]]
            if set(ids) & (occupied | reader_ids):
                raise ValueError("Clip-role overlap")
            if role == "held" and set(ids) & old_ids:
                raise ValueError("Held source entered the prior batch-one pilot")
            occupied.update(ids)
            for clip_id in ids:
                path = Path(clips[clip_id]["frames_path"])
                if digest(path) != clips[clip_id]["frames_sha256"]:
                    raise ValueError("Frame cache hash differs")
                paths.append(path)
            histories[f"{role}_{stream['name']}"] = {
                "role": role,
                "pattern": stream["pattern"],
                "clip_ids": ids,
                "paths": [clips[i]["frames_path"] for i in ids],
            }
    scientific = [
        Path(__file__).resolve(),
        ROOT / "cafl4ds/single_image_signal.py",
        ROOT / "scripts/run_single_image_signal_diagnostic.py",
        ROOT / "cafl4ds/single_image_forecast.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "cafl4ds/ssl/mae.py",
        ROOT / "cafl4ds/ssl/augment.py",
        ROOT / "cafl4ds/models/vit.py",
        ROOT / "scripts/run_adaptation_experiment.py",
        ROOT / "scripts/run_paired_update_response.py",
    ]
    paths.extend(scientific)
    lock = {
        "settings": OmegaConf.to_container(settings),
        "learner_config": OmegaConf.to_container(config),
        "histories": histories,
        "support_indices": {str(d): ids.tolist() for d, ids in supports.items()},
        "query_indices": query.tolist(),
        "hashes": {str(p): digest(p) for p in paths},
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(),
        "prior_clip_overlap_held": 0,
        "all_stream_clips": len(occupied),
        "source_independence_established": False,
    }
    if digest(Path(config.well)) != prior["hashes"][str(config.well)]:
        raise ValueError("Warm model differs")
    atomic_json(out / "input_lock.json", lock)
    torch.save({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in panel.items()}, out / "panel.pt")
    torch.save(bank, out / "bank.pt")
    for path in scientific + [CONFIG, PROTOCOL]:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return config, panel, bank, lock


def load_panel(out: Path) -> dict:
    panel = torch.load(out / "panel.pt", map_location="cpu", weights_only=True)
    for key in ("query_labels", "query_domains"):
        panel[key] = panel[key].numpy()
    return panel


def load_stream(meta: dict, frames: int = 96) -> torch.Tensor:
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False)[:frames].copy() for path in meta["paths"]]
    if any(a.shape != (frames, 64, 64, 3) for a in arrays):
        raise ValueError("Unexpected cached frame shape")
    return torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255


def measure(model, optimizer, bank, warm, panel, directory: Path, draws: list[int], steps=(100,)) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    marker = fingerprint(capture(model, optimizer))
    tick = time.perf_counter()
    metrics, raw = read_state(model, bank, warm, panel, draws, steps)
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Readout mutated complete learner/RNG")
    np.savez_compressed(
        directory / "raw.npz",
        **raw["probabilities"],
        bank_features=raw["bank_features"],
        bank_reconstruction=raw["bank_reconstruction"],
    )
    result = {"metrics": metrics, "fingerprint": marker, "seconds": time.perf_counter() - tick}
    atomic_json(directory / "readout.json", result)
    return result


def fresh(initial, config, seed: int):
    model = copy.deepcopy(initial)
    torch.manual_seed(seed)
    optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
    return model, optimizer


def preflight(initial, config, arrivals, bank, warm, panel, settings, out):
    model, optimizer = fresh(initial, config, 81991)
    seed = training_seed("preflight", 0)
    marker = fingerprint(capture(model, optimizer))
    probe = gradient_features(model, optimizer, arrivals[:1].cuda(), seed)
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Gradient probe changed the learner")
    actual = seeded_step(model, optimizer, arrivals[:1].cuda(), seed)
    if actual[0] != probe[0]:
        raise ValueError("Paid gradient probe did not use actual training view")
    before = fingerprint(capture(model, optimizer))
    values, raw = read_state(model, bank, warm, panel, [settings.reader_draws[0]], (100,))
    if fingerprint(capture(model, optimizer)) != before:
        raise ValueError("Observer impurity")
    baseline = copy.deepcopy(model.state_dict())
    zero_step(model, optimizer)
    if all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.items()):
        raise ValueError("Zero step failed to advance nonempty AdamW")
    atomic_json(
        out / "preflight.json",
        {
            "passed": True,
            "discarded_gradient_updates": 1,
            "discarded_zero_steps": 1,
            "probe_loss_equals_training_loss": True,
            "observer_invariance": True,
        },
    )


def collect(initial, config, arrivals, meta, name, bank, warm, panel, settings, out, progress):
    directory = out / "trajectories" / name
    directory.mkdir(parents=True, exist_ok=False)
    origins = list(range(settings.origin_start, settings.origin_stop + 1, settings.origin_stride))
    clocks = {0}
    for t in origins:
        clocks.update([t - 1, t, t + 1, t + 8, t + 32])
    model, optimizer = fresh(initial, config, settings.seed)
    initial_marker = fingerprint(capture(model, optimizer))
    observations = {}
    features = {}
    training = []
    marker = {}
    projection = np.random.default_rng(settings.image_projection_seed).normal(size=(warm.shape[1], 8)) / np.sqrt(
        warm.shape[1]
    )
    centroid = (warm / np.maximum(np.linalg.norm(warm, axis=1, keepdims=True), 1e-12)).mean(0)
    for t in range(len(arrivals) + 1):
        if t in clocks:
            record = measure(
                model, optimizer, bank, warm, panel, directory / "states" / str(t), [settings.reader_draws[0]]
            )
            observations[str(t)] = record["metrics"][str(settings.reader_draws[0])]
            marker[str(t)] = record["fingerprint"]
        if meta["role"] == "held" and t in (settings.diagnostic_origin - 1, settings.diagnostic_origin):
            folder = directory / "parents"
            folder.mkdir(exist_ok=True)
            torch.save(capture(model, optimizer), folder / f"{t}.pt")
        if t in origins:
            start = time.perf_counter()
            m = fingerprint(capture(model, optimizer))
            image, loss = image_features(
                model, arrivals[t : t + 1], arrivals[t - 1 : t], centroid, projection, settings.observer_seed
            )
            opt = optimizer_features(model, optimizer)
            gradient = gradient_features(model, optimizer, arrivals[t : t + 1].cuda(), training_seed(name, t))
            if fingerprint(capture(model, optimizer)) != m:
                raise ValueError("Pre-update predictors changed the learner")
            features[str(t)] = {
                "image": image.tolist(),
                "optimizer": opt.tolist(),
                "gradient": gradient.tolist(),
                "seconds": time.perf_counter() - start,
                "incoming_loss": loss,
            }
        if t == len(arrivals):
            break
        loss = seeded_step(model, optimizer, arrivals[t : t + 1].cuda(), training_seed(name, t))
        if t in origins and loss[0] != features[str(t)]["gradient"][0]:
            raise ValueError("Gradient feature differs from the actual view")
        training.append(loss)
        if (t + 1) % 64 == 0:
            progress(history=name, arrivals=t + 1, states=len(observations))
    torch.save(capture(model, optimizer), directory / "final_state.pt")
    np.save(directory / "training.npy", np.array(training))
    atomic_json(
        directory / "series.json",
        {
            "history": name,
            "role": meta["role"],
            "states": observations,
            "features": features,
            "initial_fingerprint": initial_marker,
            "fingerprints": marker,
            "final_fingerprint": fingerprint(capture(model, optimizer)),
            "updates": len(training),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "stage": "inputs", "completed_trajectories": 0}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        logger.info("{}", status)

    try:
        setup()
        settings = OmegaConf.load(CONFIG)
        config, panel, bank, lock = load_inputs(out, settings)
        initial = _method(config).cuda()
        warm = encode(initial, bank, settings.observer_seed)
        np.save(out / "warm_bank.npy", warm)
        progress(stage="preflight")
        first = load_stream(next(iter(lock["histories"].values())), settings.frames_per_period)
        preflight(initial, config, first, bank, warm, panel, settings, out)
        del first
        for name, meta in lock["histories"].items():
            progress(stage="collection", history=name, arrivals=0)
            arrivals = load_stream(meta, settings.frames_per_period)
            collect(initial, config, arrivals, meta, name, bank, warm, panel, settings, out, progress)
            progress(completed_trajectories=status["completed_trajectories"] + 1)
            del arrivals
        for path, value in lock["hashes"].items():
            if digest(Path(path)) != value:
                raise ValueError(f"Locked input/source changed: {path}")
        progress(stage="collection_complete", scientific_updates=12 * settings.stream_images)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
