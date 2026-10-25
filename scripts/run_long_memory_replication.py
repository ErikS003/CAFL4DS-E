"""Execute the sealed three-source long-horizon memory comparison."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.filters.base import FilterContext
from cafl4ds.health_calibration import changes, embeddings, geometry
from cafl4ds.historical_loss_screen import bank_losses
from cafl4ds.long_memory_replication import reader_assignments
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.reservoir_action_replication import capture, restore
from cafl4ds.update_filtering import optimize_current
from scripts.natural_health_view import accuracy_panel, frozen_predictions, write
from scripts.prepare_long_memory_replication import CONFIG, PROTOCOL, block_config
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_precision_budget import fit_and_score

_OUTPUT_CHECKS = {}


def reserve(study):
    out = ROOT / study.study_root
    if shutil.disk_usage(out).free < study.minimum_free_mib * 2**20:
        raise RuntimeError("Storage reserve reached; preserve completed work")
    if time.monotonic() - _OUTPUT_CHECKS.get(str(out), 0) < 60:
        return
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not any("-frames" in x for x in p.parts))
    if used > study.max_output_mib * 2**20:
        raise RuntimeError(f"Scientific output cap reached: {used}")
    _OUTPUT_CHECKS[str(out)] = time.monotonic()


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if not path.name.endswith("resume.pt"):
        lossless_compress(path)


def lossless_compress(path):
    """Reduce physical storage while retaining the exact scientific file bytes."""
    before = digest(path)
    if os.name == "nt":
        subprocess.run(
            [
                str(Path(os.environ["SystemRoot"]) / "System32/compact.exe"),
                "/C",
                "/EXE:LZX",
                "/I",
                "/Q",
                str(path.resolve()),
            ],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
        )
    assert digest(path) == before


def freeze(study):
    out = ROOT / study.study_root
    target = out / "input_lock.json"
    if target.exists():
        check(read(target)["hashes"])
        return
    prior = ROOT / study.prior
    old = read(prior / "input_lock.json")
    initial = prior / "initial_state.pt"
    assert digest(initial) == "e372f80d86788e70850bfca407bbd93a2a8a26fcce18f23fffe316e5199d686d"
    assert read(prior / "delivery-verification.json")["passed"]
    write(
        out / "execution-environment.json",
        {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scikit-learn", "Pillow", "omegaconf")
            },
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    )
    paths = [
        CONFIG,
        PROTOCOL,
        ROOT / study.archive_index,
        ROOT / study.selection_source,
        Path(__file__),
        prior / "input_lock.json",
        initial,
        prior / "delivery-verification.json",
        ROOT / study.evaluation / "images.pt",
        ROOT / study.evaluation / "data.npz",
        ROOT / study.evaluation / "data-role-manifest.json",
        ROOT / study.evaluation / "planning.json",
        ROOT / study.evaluation / "summary.json",
        out / "reader-planning.json",
        out / "preflight.json",
        out / "engineering-timing.json",
        out / "tests.log",
        out / "quality.log",
        out / "engineering-reader-verification.json",
        out / "execution-environment.json",
        ROOT / "outputs/long-memory-engineering/20261018-reader/reader-worker-0.json",
        ROOT / "outputs/long-memory-engineering/20261018-reader-compact/reader-worker-0.json",
        ROOT / "scripts/prepare_long_memory_replication.py",
        ROOT / "scripts/verify_long_memory_replication.py",
        ROOT / "scripts/report_long_memory_replication.py",
        ROOT / "scripts/preflight_long_memory_replication.py",
        ROOT / "scripts/verify_long_memory_delivery.py",
        ROOT / "cafl4ds/long_memory_replication.py",
        ROOT / "tests/unit/test_long_memory_replication.py",
        ROOT / "uv.lock",
    ]
    # Pin every imported project helper that actually executes in this process.
    paths += [
        Path(m.__file__)
        for name, m in sys.modules.items()
        if name.startswith(("cafl4ds.", "scripts.")) and (getattr(m, "__file__", "") or "").endswith(".py")
    ]
    paths += list(out.glob("*compression*.json"))
    for bank in (old["monitor_reference"], old["audit_reference"]):
        paths.extend(Path(row["path"]) for row in bank["images"])
    seen = set()
    for block in study.blocks:
        config = block_config(study, block)
        intake, selection = read(ROOT / config.frames / "intake.json"), read(ROOT / config.intake / "selection.json")
        assert intake["frames_used"] == 131072 and intake["batches"] == 4096
        ids = {c["clip_id"] for c in selection["clips"]}
        assert len(ids) == 384 and not ids.intersection(seen)
        seen.update(ids)
        for suffix in (
            "preparation-lock.json",
            "selection.json",
            "prior-exclusions.json",
            "compression-verification.json",
        ):
            paths.append(ROOT / config.intake / suffix)
        check(read(ROOT / config.intake / "preparation-lock.json")["hashes"])
        paths.append(ROOT / config.frames / "intake.json")
        source_audit = ROOT / config.intake / "source-preflight.json"
        assert read(source_audit)["passed"]
        paths.append(source_audit)
        for clip in intake["clips"]:
            p = Path(clip["frames_path"])
            assert digest(p) == clip["frames_sha256"]
            paths.extend([p, p.parent / "source.json", p.parent / "manifest.json", p.parent / "decoder.log"])
    assert read(out / "preflight.json")["passed"]
    assert read(out / "reader-planning.json")["selected_endpoint_draws"] == study.endpoint_draws
    hashes = {str(p.resolve()): digest(p) for p in set(paths)}
    snapshot = out / "source_snapshot"
    for path in set(paths):
        if path.suffix in (".py", ".md", ".yaml"):
            dest = snapshot / path.resolve().relative_to(ROOT.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    write(
        target,
        {
            "hashes": hashes,
            "config": OmegaConf.to_container(study, resolve=True),
            "model_config": old["model_config"],
            "monitor_reference": old["monitor_reference"],
            "audit_reference": old["audit_reference"],
            "frozen_unix": time.time(),
            "before_scientific_SSL": True,
            "initial_state": str(initial),
            "points": [0, *range(128, 4097, 128)],
            "max_SSL_starts": 221184,
        },
    )
    print({"scientific_design_frozen": True, "sources": 3, "training_draws": 3, "trained_arms": 3}, flush=True)


def memory_for(study, arm):
    if arm == "current":
        return None
    specification = {
        "_target_": "cafl4ds.filters.window_replay.WindowReplay",
        "capacity": study.capacity,
        "replay_batch": 16,
        "seed": study.sample_seed,
    }
    if arm == "reservoir":
        specification = {
            "_target_": "cafl4ds.filters.historical_reservoir.HistoricalReservoir",
            "capacity": study.capacity,
            "sample_seed": study.sample_seed,
            "admission_seed": study.admission_seed,
        }
    return instantiate(OmegaConf.create(specification))


def state(model, optimizer, memory):
    return {"learner": capture(model, optimizer), "memory": None if memory is None else memory.state_dict()}


def observe(model, optimizer, memory, banks, reference, step, losses):
    before = fingerprint(state(model, optimizer, memory))
    measured, raw = {}, {}
    for name, images in banks.items():
        values = embeddings(model, images)
        if step == 132:
            reference[name] = values.copy()
        measured[name] = {
            **geometry(values),
            "change": None if name not in reference else changes(reference[name], values),
        }
        if losses:
            measured[name]["losses"] = bank_losses(model, images).tolist()
        if step in (0, 132, 4096):
            raw[name] = values
    after = fingerprint(state(model, optimizer, memory))
    assert before == after, "Observer changed complete learner or sampler"
    return {"step": step, "metrics": measured, "state_fingerprint": before}, raw


def initial_state(lock, seed):
    start = torch.load(lock["initial_state"], weights_only=True, map_location="cpu")
    original = fingerprint({k: v for k, v in start.items() if k not in ("cpu_rng", "cuda_rng")})
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start["cpu_rng"], start["cuda_rng"] = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    assert original == fingerprint({k: v for k, v in start.items() if k not in ("cpu_rng", "cuda_rng")})
    return start


def train_arm(study, block, seed, arm, replay=False):
    out = ROOT / study.study_root
    lock = read(out / "input_lock.json")
    directory = out / f"block{block}-seed{seed}-{arm}"
    directory.mkdir(exist_ok=True)
    completed = directory / ("training-verification.json" if replay else "record.json")
    if completed.exists():
        return
    consumed = sum(read(p)["ssl_started"] for p in out.glob("block*-seed*-*/*-work.json"))
    own_journal = directory / ("reproduction-work.json" if replay else "collection-work.json")
    already = read(own_journal)["ssl_completed"] if own_journal.exists() else 0
    if consumed + 4096 - already > study.max_total_ssl:
        raise RuntimeError("The sealed physical SSL budget cannot complete another arm")
    reserve(study)
    method = OmegaConf.create(lock["model_config"])
    torch.manual_seed(11)
    template = _method(method).cuda()
    start = initial_state(lock, seed)
    model, optimizer = restore(template, method.optim, start)
    memory = memory_for(study, arm)
    stream = OrderedFrames(read(ROOT / block_config(study, block).frames / "intake.json")["clips"])
    banks = {"monitor": reference_images(lock["monitor_reference"]), "audit": reference_images(lock["audit_reference"])}
    target = read(directory / "record.json") if replay else None
    references, traces, readings, raw_banks, checkpoints = {}, [], {}, {}, {}
    journal = directory / ("reproduction-work.json" if replay else "collection-work.json")
    resume_path = directory / ("reproduction-resume.pt" if replay else "collection-resume.pt")
    counts = {
        "ssl_started": 0,
        "ssl_completed": 0,
        "complete": False,
        "block": block,
        "seed": seed,
        "arm": arm,
        "replay": replay,
        "online_seconds": 0.0,
        "observer_seconds": 0.0,
    }
    first = 0
    if resume_path.exists():
        saved = torch.load(resume_path, weights_only=True, map_location="cpu")
        model, optimizer = restore(template, method.optim, saved["state"]["learner"])
        if memory is not None:
            memory.load_state_dict(saved["state"]["memory"])
        first = saved["step"]
        traces, readings, checkpoints = saved["traces"], saved["readings"], saved["checkpoints"]
        references = {k: v.numpy() for k, v in saved["references"].items()}
        raw_banks = {k: v.numpy() for k, v in saved["raw_banks"].items()}
        counts = read(journal)
        assert counts["ssl_started"] == counts["ssl_completed"] == first, "Uncommitted work requires explicit recovery"
    elif journal.exists():
        previous = read(journal)
        assert previous["ssl_started"] == 0, "Preserve attempted learning without a resumable commit"
    write(journal, counts)
    begin = time.perf_counter()

    def record_observation(step):
        observed_at = time.perf_counter()
        both = step in lock["points"] or step == 132
        item, raw = observe(
            model, optimizer, memory, banks if both else {"monitor": banks["monitor"]}, references, step, both
        )
        readings[str(step)] = item
        raw_banks.update({f"{step}_{name}": values for name, values in raw.items()})
        if replay:
            assert item == target["readings"][str(step)], (block, seed, arm, step, "observer replay")
        if step in lock["points"]:
            encoder_hash = fingerprint(model.encoder.state_dict())
            path = out / "encoders" / ("warm.pt" if step == 0 else f"b{block}-s{seed}-{arm}-{step}.pt")
            row = {"encoder_fingerprint": encoder_hash, "path": str(path), "step": step}
            if replay:
                assert row == target["checkpoints"][str(step)]
                assert fingerprint(torch.load(path, weights_only=True, map_location="cpu")) == encoder_hash
            elif not path.exists():
                save_tensor(path, model.encoder.state_dict())
            else:
                assert fingerprint(torch.load(path, weights_only=True, map_location="cpu")) == encoder_hash
            checkpoints[str(step)] = row
        counts["observer_seconds"] += time.perf_counter() - observed_at

    if first == 0:
        record_observation(0)
    for step in range(first + 1, study.max_batches + 1):
        if step % 128 == 1:
            reserve(study)
        online_at = time.perf_counter()
        pixels = stream.batch(step - 1)
        mixed = (
            torch.from_numpy(pixels)
            if memory is None
            else memory.observe(torch.from_numpy(pixels), FilterContext(model, step - 1))
        )
        diet = (
            {"current_ids": list(range((step - 1) * 32, step * 32)), "replay_ids": []}
            if memory is None
            else copy.deepcopy(memory.last)
        )
        row = {
            "step": step,
            "input_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "train_sha256": hashlib.sha256(mixed.numpy().tobytes()).hexdigest(),
            "diet": diet,
        }
        counts["ssl_started"] += 1
        write(journal, counts)
        row["response"] = list(optimize_current(model, optimizer, mixed.permute(0, 3, 1, 2).float().cuda() / 255, True))
        counts["ssl_completed"] += 1
        counts["online_seconds"] += time.perf_counter() - online_at
        write(journal, counts)
        traces.append(row)
        if replay:
            assert row == target["trace"][step - 1], (block, seed, arm, step, "update replay")
        if step in lock["points"] or step >= 132 and (step - 132) % 8 == 0:
            record_observation(step)
        if step % 128 == 0:
            save_tensor(
                resume_path,
                {
                    "step": step,
                    "state": state(model, optimizer, memory),
                    "traces": traces,
                    "readings": readings,
                    "checkpoints": checkpoints,
                    "references": {k: torch.from_numpy(v) for k, v in references.items()},
                    "raw_banks": {k: torch.from_numpy(v) for k, v in raw_banks.items()},
                },
            )
            status = {
                "stage": "reproduce" if replay else "train",
                "pid": os.getpid(),
                "block": block,
                "seed": seed,
                "arm": arm,
                "arrival": step,
                "horizon": 4096,
                "elapsed_seconds": time.perf_counter() - begin,
            }
            write(out / "status.json", status)
            print(status, flush=True)
    final = state(model, optimizer, memory)
    result = {
        "block": block,
        "seed": seed,
        "arm": arm,
        "trace": traces,
        "readings": readings,
        "checkpoints": checkpoints,
        "final_fingerprint": fingerprint(final),
    }
    if replay:
        assert result == target
        assert fingerprint(
            torch.load(directory / "final_state.pt", weights_only=True, map_location="cpu")
        ) == fingerprint(final)
        with np.load(directory / "bank-features.npz") as old:
            assert set(old.files) == set(raw_banks)
            for key, values in raw_banks.items():
                np.testing.assert_array_equal(old[key], values)
        write(
            completed, {"passed": True, "exact_SSL_updates": 4096, "record_sha256": digest(directory / "record.json")}
        )
    else:
        save_tensor(directory / "final_state.pt", final)
        np.savez_compressed(directory / "bank-features.npz", **raw_banks)
        write(completed, result)
        lossless_compress(completed)
    counts["complete"] = True
    write(journal, counts)
    # The committed final state replaces this study's redundant transient resume.
    assert resume_path.resolve().is_relative_to(out.resolve())
    if resume_path.exists():
        resume_path.unlink()


def prepare_jobs(study):
    out = ROOT / study.study_root
    lock = read(out / "input_lock.json")
    seeds = read(ROOT / study.evaluation / "data-role-manifest.json")["support_seeds"]
    jobs = [{"id": "warm", "path": str(out / "encoders/warm.pt"), "draws": seeds}]
    for block in study.blocks:
        for seed in study.ssl_seeds:
            for arm in study.arms:
                directory = out / f"block{block}-seed{seed}-{arm}"
                assert read(directory / "training-verification.json")["passed"]
                record = read(directory / "record.json")
                for step in [4096, *lock["points"][1:-1]]:
                    jobs.append(
                        {
                            "id": f"b{block}-s{seed}-{arm}-{step}",
                            "path": record["checkpoints"][str(step)]["path"],
                            "draws": seeds if step == 4096 else list(study.curve_draws),
                        }
                    )
    assert sum(len(j["draws"]) for j in jobs) == 9679
    for job in jobs:
        job["encoder_sha256"] = digest(job["path"])
    write(out / "reader-jobs.json", jobs)
    return jobs


def reader_worker(study, worker):
    out = ROOT / study.study_root
    lock = read(out / "input_lock.json")
    reader_inputs = {
        str(ROOT / study.evaluation / name) for name in ("images.pt", "data.npz", "data-role-manifest.json")
    }
    check({p: h for p, h in lock["hashes"].items() if Path(p).suffix in (".py", ".yaml") or p in reader_inputs})
    jobs = reader_assignments(read(out / "reader-jobs.json"), study.reader_workers)[worker]
    images = torch.load(ROOT / study.evaluation / "images.pt", weights_only=True)
    with np.load(ROOT / study.evaluation / "data.npz") as stored:
        arrays = {k: stored[k] for k in stored.files}
    encoder = instantiate(OmegaConf.create(lock["model_config"]).encoder).cuda()
    counts = {f"{kind}_{stage}": 0 for kind in ("knn", "linear", "finetune") for stage in ("started", "completed")}
    counts["pid"] = os.getpid()
    journal = out / f"reader-worker-{worker}.json"
    if journal.exists():
        counts = read(journal)
        cached = sum(
            1 for job in jobs for draw in job["draws"] if (out / "readouts" / job["id"] / f"panel-{draw}.json").exists()
        )
        assert all(
            counts[f"{kind}_started"] == counts[f"{kind}_completed"] == cached for kind in ("knn", "linear", "finetune")
        ), "Uncommitted reader work requires explicit recovery"
    write(journal, counts)
    for job in jobs:
        directory = out / "readouts" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            check(read(directory / "complete.json")["hashes"])
            continue
        reserve(study)
        assert digest(job["path"]) == job["encoder_sha256"]
        encoder.load_state_dict(torch.load(job["path"], weights_only=True, map_location="cpu"))
        # Wrapper is only the interface expected by the immutable embedding helper.
        wrapper = torch.nn.Module()
        wrapper.add_module("encoder", encoder)
        features = embeddings(wrapper, images)
        feature_hash = hashlib.sha256(features.tobytes()).hexdigest()
        scores = {k: [] for k in ("knn", "linear", "finetune")}
        hashes = {}
        for draw in job["draws"]:
            path = directory / f"panel-{draw}.npz"
            audit_path = directory / f"panel-{draw}.json"
            if path.exists() and audit_path.exists():
                assert digest(path) == read(audit_path)["sha256"]
            else:
                predicted = frozen_predictions(features, arrays, [draw], counts, journal)
                counts["finetune_started"] += 1
                write(journal, counts)
                with torch.random.fork_rng(devices=[0]):
                    probabilities, audit = fit_and_score(encoder, images, arrays, draw)
                counts["finetune_completed"] += 1
                write(journal, counts)
                assert probabilities.shape == (4063, 2)
                np.savez_compressed(path, probabilities=probabilities, **predicted)
                write(
                    audit_path, {**audit, "seed": draw, "sha256": digest(path), "encoder_sha256": job["encoder_sha256"]}
                )
            with np.load(path) as saved:
                for kind in ("knn", "linear"):
                    scores[kind].append(accuracy_panel(saved[kind], arrays)[0])
                scores["finetune"].append(accuracy_panel(saved["probabilities"][:3663].argmax(-1), arrays))
            hashes[str(path)], hashes[str(audit_path)] = digest(path), digest(audit_path)
        write(
            directory / "complete.json",
            {
                "job": job,
                "scores": scores,
                "hashes": hashes,
                "features_sha256": feature_hash,
                "features_shape": list(features.shape),
                "features_reconstruction": "exact saved encoder plus pinned evaluation images",
            },
        )
        print({"reader_worker": worker, "completed": job["id"], "panels": len(job["draws"])}, flush=True)


def readers(study):
    prepare_jobs(study)
    out = ROOT / study.study_root
    processes, logs = [], []
    try:
        for worker in range(study.reader_workers):
            handle = (out / f"reader-worker-{worker}.log").open("ab")
            logs.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_long_memory_replication",
                        "--stage",
                        "worker",
                        "--worker",
                        str(worker),
                    ],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            )
        while any(p.poll() is None for p in processes):
            for p in processes:
                if p.poll() not in (None, 0):
                    raise RuntimeError("Reader worker failed; preserve logs and completed panels")
            progress = [
                read(out / f"reader-worker-{i}.json") if (out / f"reader-worker-{i}.json").exists() else {}
                for i in range(study.reader_workers)
            ]
            write(out / "status.json", {"stage": "readers", "pid": os.getpid(), "workers": progress})
            time.sleep(10)
        assert all(p.returncode == 0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for handle in logs:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "train", "readers", "worker", "all"), default="all")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    setup()
    study = OmegaConf.load(CONFIG)
    if args.stage in ("freeze", "all"):
        freeze(study)
    if args.stage in ("train", "all"):
        check(read(ROOT / study.study_root / "input_lock.json")["hashes"])
        for block in study.blocks:
            for seed in study.ssl_seeds:
                for arm in study.arms:
                    train_arm(study, int(block), int(seed), arm)
                    train_arm(study, int(block), int(seed), arm, replay=True)
    if args.stage in ("readers", "all"):
        readers(study)
    if args.stage == "worker":
        reader_worker(study, args.worker)
    if args.stage == "all":
        for module in ("scripts.verify_long_memory_replication", "scripts.report_long_memory_replication"):
            subprocess.run([sys.executable, "-u", "-m", module], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
