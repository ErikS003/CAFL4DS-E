"""Collect a locked SimSiam collapse-warning screen; no runtime health gradients."""

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
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.collapse_risk_regression import ARMS, crossfit, geometry, ridge_fit, summarize
from cafl4ds.short_buffer import capture, restore
from cafl4ds.ssl.augment import TwoView, make_ssl_augment
from scripts.run_paired_update_response import fingerprint

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"
EVAL = ROOT / "outputs/content-filter-bdd/20261011-coverage-half"
ARRAYS = ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz"
CONFIG = ROOT / "cafl4ds/configs/collapse_risk_regression.yaml"
PROTOCOL = ROOT / "docs/project-plan/collapse-risk-regression-frozen.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def cpu_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {k: cpu_state(v) for k, v in state.items()}
    if isinstance(state, list | tuple):
        return [cpu_state(v) for v in state]
    return state


def observe(model, optimizer, images, projector=True):
    """Read embeddings with every parameter, BN buffer, mode, optimizer and RNG invariant."""
    before = fingerprint(capture(model, optimizer))
    modes = {n: m.training for n, m in model.named_modules()}
    device = next(model.parameters()).device
    result = []
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
            for batch in images.split(128):
                z = model.encoder.embed(batch.to(device))
                result.append((model.projector(z) if projector else z).cpu())
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if fingerprint(capture(model, optimizer)) != before:
        raise ValueError("Observer changed complete learner state")
    return torch.cat(result).numpy()


def update(model, optimizer, batch, seed):
    """Explicit update-index RNG gives matched draws even across differing objectives."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        loss = model.training_step(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
    return float(loss.detach())


def role_manifest(intake):
    selected = []
    seen = set()
    clips = {c["clip_id"]: c for c in intake["clips"]}
    for stream in intake["streams"]:
        ids = [c for period in stream["periods"] for c in period]
        if len(ids) != 24 or seen.intersection(ids):
            raise ValueError("Disjoint 24-clip cohorts required")
        seen.update(ids)
        selected.append({"group": stream["name"], "warm": ids[:8], "continuation": ids[8:20], "bank": ids[20:]})
    val = Path("C:/CAFL-E/datasets/bdd100k/images/100k/val")
    validation_ids = sorted(p.stem for p in val.glob("*.jpg"))
    if not validation_ids or set(validation_ids).intersection(seen) or any(clips[c]["split"] != "train" for c in seen):
        raise ValueError("Training clips overlap validation identities or inventory is missing")
    return {
        "cohorts": selected,
        "role": "reused_development",
        "capture_sessions_known": False,
        "validation_identity_overlap": 0,
        "validation_inventory_count": len(validation_ids),
        "validation_inventory_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
        "evaluation_cache_role": "inherited validation split, no query/support overlap",
    }


def lock(out, config, manifest, intake, resume):
    scientific = [
        Path(__file__),
        ROOT / "scripts/verify_collapse_risk_regression.py",
        ROOT / "cafl4ds/collapse_risk_regression.py",
        ROOT / "tests/unit/test_collapse_risk_regression.py",
        CONFIG,
        PROTOCOL,
        ROOT / "cafl4ds/artifacts.py",
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "scripts/run_paired_update_response.py",
        ROOT / "cafl4ds/ssl/simsiam.py",
        ROOT / "cafl4ds/ssl/factory.py",
        ROOT / "cafl4ds/ssl/augment.py",
        ROOT / "cafl4ds/ssl/base.py",
        ROOT / "cafl4ds/models/vit.py",
        ROOT / "cafl4ds/models/heads.py",
    ]
    paths = scientific + [
        INTAKE,
        ARRAYS,
        EVAL / "evaluation_images.pt",
        EVAL / "data_lock.json",
        EVAL / "input_lock.json",
    ]
    paths += [Path(c["frames_path"]) for c in intake["clips"]]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    for clip in intake["clips"]:
        if hashes[str(Path(clip["frames_path"]).resolve())] != clip["frames_sha256"]:
            raise ValueError("Cached frame source changed")
    payload = {
        "hashes": hashes,
        "config": OmegaConf.to_container(config, resolve=True),
        "roles": manifest,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    if resume:
        if read(out / "input_lock.json") != payload:
            raise ValueError("Resume requires unchanged code/config/data/environment lock")
    else:
        atomic_json(out / "input_lock.json", payload)
        atomic_json(out / "data-role-manifest.json", manifest)
        for p in scientific:
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())


def pool(ids, clips, stride):
    arrays = [np.load(clips[c]["frames_path"], mmap_mode="r")[::stride] for c in ids]
    x = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float().cuda() / 255
    return F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False, antialias=True)


def configure_arm(model, arm):
    model.collapse_alpha = {"blend005": 0.05, "blend015": 0.15, "blend100": 1.0}.get(arm, 0.0)
    model.stopgrad_beta = 1 - {"soft015": 0.15, "soft050": 0.5, "soft100": 1.0}.get(arm, 0.0)
    if arm == "heavy":
        model.two_view = TwoView(make_ssl_augment(32, min_scale=0.1, jitter_strength=1.5))


def probe(model, optimizer, evaluation, arrays):
    features = observe(model, optimizer, evaluation, projector=False)
    labels, query = arrays["labels"], arrays["query"]
    targets = np.eye(int(labels.max()) + 1)[labels]
    predictions, scores = {}, []
    for draw in (51011, 51017, 51023):
        support = arrays[f"support_{draw}"]
        if np.intersect1d(query, support).size:
            raise ValueError("Probe support/query overlap")
        values, _ = ridge_fit(features[support], targets[support], features[query])
        predicted = values.argmax(1)
        predictions[str(draw)] = predicted
        scores.append(100 * float(np.mean(predicted == labels[query])))
    return features, predictions, scores


def warm(config, images, evaluation, arrays, seed):
    torch.manual_seed(seed)
    model = instantiate(config.ssl, encoder=instantiate(config.encoder)).cuda()
    optimizer = instantiate(config.optim, params=model.parameters())
    initial, initial_predictions, initial_scores = probe(model, optimizer, evaluation, arrays)
    generator = np.random.default_rng(seed + 100003)
    losses = []
    for step in range(config.warm_updates):
        indices = generator.integers(len(images), size=config.batch_size)
        losses.append(update(model, optimizer, images[indices], seed * 100000 + step))
    features, predictions, scores = probe(model, optimizer, evaluation, arrays)
    return (
        model,
        capture(model, optimizer),
        losses,
        {
            "initial": initial,
            "warm": features,
            **{f"initial_{k}": v for k, v in initial_predictions.items()},
            **{f"warm_{k}": v for k, v in predictions.items()},
        },
        initial_scores,
        scores,
    )


def branch(initial, state, config, images, bank, evaluation, arrays, seed, arm):
    model, optimizer = restore(initial, config.optim, state)
    configure_arm(model, arm)
    generator = np.random.default_rng(seed + 200003)
    source = images[:64] if arm == "lowdiv" else images
    readings = [observe(model, optimizer, bank)]
    losses = []
    for step in range(config.continuation_updates):
        indices = generator.integers(len(source), size=config.batch_size)
        losses.append(update(model, optimizer, source[indices], seed * 100000 + config.warm_updates + step))
        if (step + 1) % config.cadence == 0:
            readings.append(observe(model, optimizer, bank))
    features, predictions, scores = probe(model, optimizer, evaluation, arrays)
    return (
        {
            "projector": np.stack(readings),
            "evaluation": features,
            **{f"prediction_{k}": v for k, v in predictions.items()},
        },
        losses,
        scores,
        fingerprint(capture(model, optimizer)),
    )


def saved_record(path):
    row = read(path)
    for p, expected in row["artifact_hashes"].items():
        if digest(p) != expected:
            raise ValueError("Completed record artifact changed")
    if not row["exact_replay"]:
        raise ValueError("Unaudited completed record")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG)
    out = (args.output or ROOT / config.output).resolve()
    if args.resume:
        if not (out / "input_lock.json").exists():
            raise ValueError("No locked run to resume")
    else:
        out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 2 * 2**30:
        raise ValueError("Require 2GiB reserve")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    status = {"pid": os.getpid(), "stage": "locking", "completed_arms": 0}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    progress()
    intake = read(INTAKE)
    manifest = role_manifest(intake)
    lock(out, config, manifest, intake, args.resume)
    with np.load(ARRAYS, allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files}
    evaluation = torch.load(EVAL / "evaluation_images.pt", weights_only=True)
    evaluation = F.interpolate(evaluation.cuda(), size=(32, 32), mode="bilinear", align_corners=False, antialias=True)
    np.savez_compressed(out / "evaluation_arrays.npz", **arrays)
    clips = {c["clip_id"]: c for c in intake["clips"]}
    parents, records = [], []
    for cohort in manifest["cohorts"]:
        group = cohort["group"]
        warm_images, images, bank = (
            pool(cohort["warm"], clips, 4),
            pool(cohort["continuation"], clips, 8),
            pool(cohort["bank"], clips, 8),
        )
        for seed in config.seeds:
            directory = out / "runs" / group / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            parent_path = directory / "parent.json"
            if parent_path.exists():
                parent = saved_record(parent_path)
                state = torch.load(directory / "warm.pt", map_location="cpu", weights_only=True)
                initial = instantiate(config.ssl, encoder=instantiate(config.encoder)).cuda()
            else:
                progress(stage="warm_and_replay", group=group, seed=seed)
                model, state, losses, parent_arrays, initial_scores, warm_scores = warm(
                    config, warm_images, evaluation, arrays, seed
                )
                replay_model, replay_state, replay_losses, replay_arrays, _, _ = warm(
                    config, warm_images, evaluation, arrays, seed
                )
                if (
                    fingerprint(state) != fingerprint(replay_state)
                    or losses != replay_losses
                    or any(not np.array_equal(parent_arrays[k], replay_arrays[k]) for k in parent_arrays)
                ):
                    raise ValueError("Warm exact replay failed")
                torch.save(cpu_state(state), directory / "warm.pt")
                np.savez_compressed(directory / "parent_arrays.npz", **parent_arrays)
                parent = {
                    "group": group,
                    "seed": seed,
                    "initial_scores": initial_scores,
                    "warm_scores": warm_scores,
                    "losses": losses,
                    "fingerprint": fingerprint(state),
                    "exact_replay": True,
                    "artifact_hashes": {
                        str(directory / n): digest(directory / n) for n in ("warm.pt", "parent_arrays.npz")
                    },
                }
                atomic_json(parent_path, parent)
                initial = copy.deepcopy(model)
                del replay_model, replay_state, model
            parents.append(parent)
            for arm in ARMS:
                record_path = directory / f"{arm}.json"
                if record_path.exists():
                    row = saved_record(record_path)
                else:
                    progress(stage="arm_and_replay", group=group, seed=seed, arm=arm)
                    result, losses, scores, marker = branch(
                        initial, state, config, images, bank, evaluation, arrays, seed, arm
                    )
                    replay, replay_losses, _, replay_marker = branch(
                        initial, state, config, images, bank, evaluation, arrays, seed, arm
                    )
                    if (
                        marker != replay_marker
                        or losses != replay_losses
                        or any(not np.array_equal(result[k], replay[k]) for k in result)
                    ):
                        raise ValueError("Arm exact replay failed")
                    path = directory / f"{arm}.npz"
                    np.savez_compressed(path, **result)
                    readings = result["projector"]
                    row = {
                        "group": group,
                        "seed": seed,
                        "arm": arm,
                        "losses": losses,
                        "scores": scores,
                        "health": [geometry(z, readings[0]) for z in readings],
                        "fingerprint": marker,
                        "exact_replay": True,
                        "arrays": str(path),
                        "artifact_hashes": {str(path): digest(path)},
                    }
                    atomic_json(record_path, row)
                records.append(row)
                status["completed_arms"] = len(records)
                atomic_json(out / "collection.json", {"parents": parents, "records": records, "complete": False})
                if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 512 * 2**20:
                    raise ValueError("Output budget exceeded; preserve completed artifacts")
            del initial, state
        del warm_images, images, bank
    atomic_json(out / "collection.json", {"parents": parents, "records": records, "complete": True})
    progress(stage="crossfit")
    fitted = crossfit(records)
    atomic_json(out / "regressions.json", fitted)
    result = summarize(records, parents, fitted)
    result["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(out / "summary.json", result)
    progress(stage="complete", ready=result["ready_for_separate_controller_experiment"])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
