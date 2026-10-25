"""Collect dense batch-one MAE time series and causal offline/online forecasts."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
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
from cafl4ds.single_image_forecast import (
    encode,
    fixed_reconstruction,
    forecasts,
    image_features,
    observe_state,
    single_step,
)
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest, remap

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_forecast.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-forecast-frozen.md"
READER = ROOT / "outputs/temporal-state-opportunity/20261011-development-v1/experiment"
INTAKE = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"
REFERENCE = ROOT / "outputs/health-calibration/20261013-gpu/reference.json"
ORIGINAL = ROOT / "outputs/paired-update-response/20261010-states/input_lock.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_image(value: str) -> Path:
    """Resolve archived BDD paths on this machine without changing old records."""
    path = Path(value)
    if path.exists():
        return path
    text = value.replace("\\", "/")
    if "/bdd100k/" in text:
        return ROOT.parents[1] / "datasets/bdd100k" / text.split("/bdd100k/", 1)[1]
    return remap(value)


def load_inputs(out: Path, settings):
    old = read(ORIGINAL)
    config = OmegaConf.create(old["config"])
    warm = remap(config.well)
    if digest(warm) != old["hashes"][str(config.well)]:
        raise ValueError("Warm checkpoint changed")
    config.well, config.device = str(warm), "cuda"
    config.batch_size, config.stream.batch_size = 1, 1
    config.optim.lr, config.optim.weight_decay = settings.ssl_lr, settings.ssl_weight_decay
    intake, reference = read(INTAKE), read(REFERENCE)
    manifest = read(READER / "reader-allocation.json")
    all_images = torch.load(READER / "evaluation_images.pt", weights_only=True, map_location="cpu")
    with np.load(READER / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        labels, domains = arrays["labels"].copy(), arrays["domains"].copy()
        support = arrays[f"support_{settings.probe_draw}"].copy()
        support = support[labels[support] < 2]
        old_query = arrays["query"].copy()
    query = np.concatenate(
        [
            old_query[(labels[old_query] == label) & (domains[old_query] == domain)][:100]
            for domain in (0, 1)
            for label in (0, 1)
        ]
    )
    if len(support) != 50 or len(query) != 400 or set(support) & set(query):
        raise ValueError("Invalid fixed support/query membership")
    paths = [
        CONFIG,
        PROTOCOL,
        ORIGINAL,
        warm,
        INTAKE,
        REFERENCE,
        READER / "evaluation_arrays.npz",
        READER / "evaluation_images.pt",
        READER / "reader-allocation.json",
    ]
    reader_ids = set()
    for index in np.r_[support, query]:
        row = manifest["selected"][index]
        if row["role"] == "sealed_final_query":
            raise ValueError("Sealed query used")
        path = local_image(row["image_path"])
        if digest(path) != row["image_sha256"]:
            raise ValueError("Reader source hash differs")
        with Image.open(path) as image:
            x = torch.from_numpy(np.array(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        if not torch.equal(x.permute(2, 0, 1).float() / 255, all_images[index]):
            raise ValueError("Reader cache pixel mismatch")
        attrs = row["keyframe_attributes"]
        if labels[index] != ("city street", "highway").index(attrs["scene"]) or domains[index] != (
            attrs["timeofday"] == "night"
        ):
            raise ValueError("Reader labels differ from archived attributes")
        reader_ids.add(row["clip_id"])
        paths.append(path)
    panel = {
        "support_images": all_images[support],
        "support_labels": torch.from_numpy(labels[support]),
        "query_images": all_images[query],
        "query_labels": labels[query],
        "query_domains": domains[query],
    }
    bank = []
    for row in reference["images"]:
        path = local_image(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Health bank source changed")
        with Image.open(path) as image:
            bank.append(np.array(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        paths.append(path)
    bank = torch.from_numpy(np.stack(bank)).permute(0, 3, 1, 2).float() / 255
    by_clip = {c["clip_id"]: c for c in intake["clips"]}
    by_stream = {s["name"]: s for s in intake["streams"]}
    streams, selected, all_ids = {}, {}, set()
    for name in settings.history_names:
        stream = by_stream[name]
        images, rows = [], []
        for period, members in enumerate(stream["periods"]):
            clip_id = members[0]
            if clip_id in all_ids or clip_id in reader_ids:
                raise ValueError("Repeated clip or reader/stream overlap")
            all_ids.add(clip_id)
            clip = by_clip[clip_id]
            path = Path(clip["frames_path"])
            if digest(path) != clip["frames_sha256"]:
                raise ValueError("Frame cache hash differs")
            frames = np.load(path, allow_pickle=False)
            if frames.shape != (512, 64, 64, 3) or frames.dtype != np.uint8:
                raise ValueError("Unexpected frame cache")
            images.append(frames[: settings.frames_per_period].copy())
            for slot in range(settings.frames_per_period):
                rows.append(
                    {
                        "clip_id": clip_id,
                        "period": period,
                        "frame": slot,
                        "rgb_sha256": hashlib.sha256(frames[slot].tobytes()).hexdigest(),
                        "domain": stream["pattern"][period],
                    }
                )
            paths.append(path)
        streams[name] = torch.from_numpy(np.concatenate(images)).permute(0, 3, 1, 2).float() / 255
        selected[name] = {"pattern": stream["pattern"], "frames": rows}
        if len(rows) != 192:
            raise ValueError("Expected 192 arrivals")
    source_paths = [
        Path(__file__),
        ROOT / "scripts/report_single_image_forecast.py",
        ROOT / "scripts/verify_single_image_forecast.py",
        ROOT / "cafl4ds/single_image_forecast.py",
        ROOT / "cafl4ds/ssl/mae.py",
        ROOT / "cafl4ds/models/vit.py",
        ROOT / "scripts/run_adaptation_experiment.py",
        ROOT / "scripts/run_paired_update_response.py",
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
    ]
    paths.extend(source_paths)
    lock = {
        "settings": OmegaConf.to_container(settings),
        "learner_config": OmegaConf.to_container(config),
        "hashes": {str(p): digest(p) for p in paths},
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "gpu": torch.cuda.get_device_name(),
        "selected": selected,
        "support_indices": support.tolist(),
        "query_indices": query.tolist(),
        "capture_sessions_known": False,
        "development_only": True,
        "sealed_final_scored": False,
        "stream_clip_count": len(all_ids),
        "batch_size": 1,
        "accumulation_steps": 1,
    }
    atomic_json(out / "input_lock.json", lock)
    np.savez_compressed(
        out / "panel.npz",
        support_labels=panel["support_labels"].numpy(),
        query_labels=panel["query_labels"],
        query_domains=panel["query_domains"],
    )
    for path in source_paths + [PROTOCOL, CONFIG]:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return config, streams, bank, panel, lock


def initial_state(initial, config, seed):
    model = copy.deepcopy(initial)
    torch.manual_seed(seed)
    optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
    return model, optimizer


def measure(model, optimizer, bank, warm, panel, settings, full=False):
    marker = fingerprint(capture(model, optimizer))
    start = time.perf_counter()
    values, raw = observe_state(model, bank, warm, panel, settings.observer_seed, full)
    torch.cuda.synchronize()
    if marker != fingerprint(capture(model, optimizer)):
        raise ValueError("Observer mutated model, optimizer, modes or RNG")
    return values, raw, time.perf_counter() - start


def run_preflight(initial, config, images, bank, warm, panel, settings, out):
    model, optimizer = initial_state(initial, config, 81991)
    before, first, seconds = measure(model, optimizer, bank, warm, panel, settings, True)
    again, second, _ = measure(model, optimizer, bank, warm, panel, settings, True)
    if before != again or any(
        not np.array_equal(first["probabilities"][k], second["probabilities"][k]) for k in first["probabilities"]
    ):
        raise ValueError("Repeated observer differs")
    trace = []
    for image in images[:3]:
        trace.append(single_step(model, optimizer, image[None].cuda()))
        measure(model, optimizer, bank, warm, panel, settings)
    final = fingerprint(capture(model, optimizer))
    replay, replay_optimizer = initial_state(initial, config, 81991)
    retrace = [single_step(replay, replay_optimizer, image[None].cuda()) for image in images[:3]]
    if trace != retrace or final != fingerprint(capture(replay, replay_optimizer)):
        raise ValueError("Observers changed single-image training trajectory")
    atomic_json(
        out / "preflight.json",
        {
            "passed": True,
            "discarded_ssl_updates": 6,
            "repeated_observation_exact": True,
            "training_with_and_without_observers_exact": True,
            "full_observation_seconds": seconds,
            "fine_tune20_reconstruction_and_all_readers_finite": True,
        },
    )


def collect(initial, config, images, bank, warm, panel, settings, directory, progress):
    directory.mkdir(parents=True, exist_ok=False)
    seed = int(directory.name.split("_")[-1])
    model, optimizer = initial_state(initial, config, seed)
    initial_marker = fingerprint(capture(model, optimizer))
    current, raw, elapsed = measure(model, optimizer, bank, warm, panel, settings, True)
    metrics = list(current) + ["incoming_reconstruction"]
    before, after, features, training = [], [], [], []
    states, readouts, embeddings = (
        [current],
        [{k: v for k, v in raw["probabilities"].items() if k != "finetune100"}],
        [raw["bank_features"]],
    )
    full_states = [
        {"state": 0, "metrics": raw["full_metrics"], "probabilities": raw["probabilities"]["finetune100"].tolist()}
    ]
    observer_seconds, image_seconds, update_seconds = elapsed, 0.0, 0.0
    generator = np.random.default_rng(settings.image_projection_seed)
    projection = generator.normal(size=(warm.shape[1], 8)) / np.sqrt(warm.shape[1])
    centroid = (warm / np.maximum(np.linalg.norm(warm, axis=1, keepdims=True), 1e-12)).mean(0)
    image_rows, checkpoints = [], {}
    for t, image in enumerate(images):
        tick = time.perf_counter()
        marker = fingerprint(capture(model, optimizer))
        feat, pre_loss = image_features(
            model, image[None], None if t == 0 else images[t - 1][None], centroid, projection, settings.observer_seed
        )
        if marker != fingerprint(capture(model, optimizer)):
            raise ValueError("Image features changed learner")
        torch.cuda.synchronize()
        image_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        train = single_step(model, optimizer, image[None].cuda())
        torch.cuda.synchronize()
        update_seconds += time.perf_counter() - tick
        full = (t + 1) % settings.full_fine_tune_every == 0
        following, raw, seconds = measure(model, optimizer, bank, warm, panel, settings, full)
        observer_seconds += seconds
        post_loss = float(fixed_reconstruction(model, image[None], settings.observer_seed + 1)[0])
        a, b = dict(current, incoming_reconstruction=pre_loss), dict(following, incoming_reconstruction=post_loss)
        before.append([a[k] for k in metrics])
        after.append([b[k] for k in metrics])
        features.append(feat)
        training.append(train)
        states.append(following)
        readouts.append({k: v for k, v in raw["probabilities"].items() if k != "finetune100"})
        embeddings.append(raw["bank_features"])
        image_rows.append(
            {
                "step": t + 1,
                "pre_loss": pre_loss,
                "post_loss": post_loss,
                "training_loss": train[0],
                "gradient_norm": train[1],
                "observer_seconds": seconds,
            }
        )
        if full:
            full_states.append(
                {
                    "state": t + 1,
                    "metrics": raw["full_metrics"],
                    "probabilities": raw["probabilities"]["finetune100"].tolist(),
                }
            )
            checkpoints[str(t + 1)] = fingerprint(capture(model, optimizer))
        current = following
        if (t + 1) % 16 == 0:
            atomic_json(directory / "progress.json", {"completed_images": t + 1, "total_images": len(images)})
            progress(step=t + 1)
    before, after, features, training = map(np.array, (before, after, features, training))
    targets, predicted = forecasts(
        before, after, features, metrics, training, settings.prefix_images, settings.ridge_penalty
    )
    np.savez_compressed(
        directory / "series.npz",
        before=before,
        after=after,
        image=features,
        training=training,
        bank_features=np.array(embeddings),
        metrics=np.array(metrics),
    )
    np.savez_compressed(directory / "forecasts.npz", targets=targets, **predicted)
    np.savez_compressed(directory / "readouts.npz", **{k: np.array([r[k] for r in readouts]) for k in readouts[0]})
    torch.save(capture(model, optimizer), directory / "final_state.pt")
    atomic_json(
        directory / "observations.json",
        {
            "metrics": metrics,
            "states": states,
            "images": image_rows,
            "full_finetune": full_states,
            "initial_fingerprint": initial_marker,
            "checkpoints": checkpoints,
            "costs": {
                "observers_seconds": observer_seconds,
                "image_features_seconds": image_seconds,
                "ssl_updates_seconds": update_seconds,
            },
            "single_image_updates": len(images),
            "observer_invariance_checks": 2 * len(images) + 1,
        },
    )
    logger.info("collected {}", directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    status = {"pid": os.getpid(), "stage": "inputs", "completed_trajectories": 0, "total_trajectories": 12}

    def progress(**values):
        status.update(values, elapsed_seconds=time.perf_counter() - start)
        atomic_json(out / "status.json", status)
        logger.info("{}", status)

    try:
        setup()
        settings = OmegaConf.load(CONFIG)
        if settings.batch_size != 1 or settings.fine_tune_steps != 20:
            raise ValueError("Frozen protocol differs")
        config, streams, bank, panel, lock = load_inputs(out, settings)
        initial = _method(config).cuda()
        warm = encode(initial, bank, settings.observer_seed)
        np.save(out / "warm_bank.npy", warm)
        progress(stage="preflight")
        run_preflight(initial, config, next(iter(streams.values())), bank, warm, panel, settings, out)
        for name, images in streams.items():
            for seed in settings.seeds:
                progress(stage="collection", history=name, seed=seed, step=0)
                directory = out / "trajectories" / name / f"seed_{seed}"
                collect(initial, config, images, bank, warm, panel, settings, directory, progress)
                progress(completed_trajectories=status["completed_trajectories"] + 1)
        if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
            raise ValueError("Frozen sources/inputs changed during collection")
        progress(stage="collection_complete", scientific_updates=2304)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
