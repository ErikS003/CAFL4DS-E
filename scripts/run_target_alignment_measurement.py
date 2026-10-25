"""Reconstruct fixed SSL states and measure paired few-label learner sensitivity."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.functional_retention import plain
from cafl4ds.short_buffer import capture
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_collapse_risk_regression import digest, read
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import fine_tune_gpu
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_short_buffer_experiment import encode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_alignment_measurement.yaml"


def write(path, value):
    atomic_json(path, plain(value))


def fit_probabilities(encoder, images, labels, support, query, seed):
    """Invoke the pinned canonical learner; retain probabilities, not only argmax."""
    tuned, head, audit = fine_tune_gpu(encoder, images[support], labels[support], seed + 25)
    features = encode(tuned, images[query])
    with torch.no_grad():
        probabilities = head(features.cuda()).softmax(-1).cpu().numpy()
    return probabilities, audit


def read_state(model, optimizer, images, arrays, config, directory, progress):
    directory.mkdir(parents=True, exist_ok=True)
    marker = fingerprint(capture(model, optimizer))
    encoder_state = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
    encoder_marker = fingerprint(encoder_state)
    if not (directory / "encoder.pt").exists():
        torch.save(encoder_state, directory / "encoder.pt")
    else:
        assert fingerprint(torch.load(directory / "encoder.pt", weights_only=True)) == encoder_marker
    query = np.concatenate([arrays["query"], arrays["legacy_query"]])
    predictions, audits = [], []
    started = time.perf_counter()
    labels = torch.from_numpy(arrays["labels"])
    with torch.random.fork_rng(devices=[0]):
        for seed in config.support_seeds:
            path = directory / f"panel_{seed}.npz"
            audit_path = directory / f"panel_{seed}.json"
            if path.exists():
                audit = read(audit_path)
                assert audit["sha256"] == digest(path) and audit["state_fingerprint"] == marker
                with np.load(path) as saved:
                    probability = saved["probabilities"]
            else:
                probability, fitted = fit_probabilities(
                    model.encoder, images, labels, arrays[f"support_{seed}"], query, seed
                )
                np.savez_compressed(path, probabilities=probability)
                audit = {**fitted, "sha256": digest(path), "state_fingerprint": marker, "draw": seed}
                write(audit_path, audit)
            predictions.append(probability)
            audits.append(audit)
            progress(stage="downstream_fit", checkpoint=str(directory.relative_to(ROOT)), draw=seed)
    assert fingerprint(capture(model, optimizer)) == marker, "Downstream evaluation changed SSL state"
    np.savez_compressed(directory / "predictions.npz", probabilities=np.array(predictions))
    result = {
        "fingerprint": marker,
        "encoder_fingerprint": encoder_marker,
        "fits": audits,
        "seconds": time.perf_counter() - started,
        "new_query_count": len(arrays["query"]),
        "legacy_query_count": len(arrays["legacy_query"]),
    }
    write(directory / "binding.json", result)
    return result


def main():
    config = OmegaConf.load(CONFIG)
    out, source = ROOT / config.output, ROOT / config.source
    assert (out / "data-role-manifest.json").exists(), "Prepare data first"
    assert not (out / "collection.json").exists(), "Completed science is immutable"
    assert shutil.disk_usage(out).free > 2 * 2**30
    started = time.perf_counter()

    def progress(**fields):
        write(out / "status.json", {**fields, "pid": os.getpid(), "elapsed_seconds": time.perf_counter() - started})
        print(plain(fields), flush=True)

    setup()
    prior = read(source / "input_lock.json")
    assert read(source / "verification.json")["passed"] and read(source / "independent-verification.json")["passed"]
    manifest = read(out / "data-role-manifest.json")
    scientific = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/prepare_target_alignment_measurement.py",
        ROOT / "scripts/verify_target_alignment_measurement.py",
        ROOT / "cafl4ds/target_alignment_measurement.py",
        ROOT / "tests/unit/test_target_alignment_measurement.py",
        ROOT / "docs/project-plan/target-alignment-measurement-frozen.md",
    ]
    paths = scientific + [
        source / n for n in ("input_lock.json", "verification.json", "independent-verification.json", "collection.json")
    ]
    paths += [out / "data-role-manifest.json", ROOT / config.frames / "intake.json"]
    for i in range(6):
        paths.append(source / "runs" / f"fresh_{i:02d}" / f"seed_{config.ssl_seed}" / "parent-path.json")
    hashes = {
        **prior["hashes"],
        **manifest["hashes"],
        **manifest["prior_metadata_hashes"],
        **{str(p): digest(p) for p in paths},
    }
    progress(stage="verifying_input_hashes", files=len(hashes))
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    lock = {
        "hashes": hashes,
        "config": OmegaConf.to_container(config, resolve=True),
        "model_config": prior["model_config"],
        "role": "conditional_measurement_on_reused_histories_new_documented_eval_ids",
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name()},
    }
    if (out / "input_lock.json").exists():
        assert read(out / "input_lock.json") == plain(lock)
    else:
        write(out / "input_lock.json", lock)
        for path in scientific:
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    images = torch.load(out / "images.pt", weights_only=True)
    with np.load(out / "data.npz") as data:
        arrays = {k: data[k] for k in data.files}
    with np.load(out / "resamples.npz") as data:
        resamples = {k: data[k] for k in data.files}
    measure = instantiate(config.measurement)
    model_config = OmegaConf.create(prior["model_config"])
    torch.manual_seed(11)
    initial = _method(model_config).cuda()
    intake = read(ROOT / config.frames / "intake.json")
    clips = {c["clip_id"]: c for c in intake["clips"]}
    points = sorted({int(age) + k for age in config.parents.values() for k in (0, config.horizon)})
    units = []
    for stream in intake["streams"]:
        directory = out / "runs" / stream["name"]
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            unit = read(directory / "complete.json")
            assert all(digest(p) == h for p, h in unit["hashes"].items())
            units.append(unit)
            continue
        old = read(source / "runs" / stream["name"] / f"seed_{config.ssl_seed}" / "parent-path.json")
        arrivals = gather(stream, clips, list(range(296 * 32)))
        torch.manual_seed(config.ssl_seed)
        model = copy.deepcopy(initial)
        optimizer = instantiate(model_config.optim, params=model.parameters())
        bindings, trace = {}, []
        for step in range(297):
            if str(step) in old["checkpoints"]:
                marker = fingerprint(capture(model, optimizer))
                assert marker == old["checkpoints"][str(step)]["fingerprint"], (stream["name"], step)
            if step in points:
                bindings[str(step)] = read_state(
                    model, optimizer, images, arrays, config, directory / f"at_{step}", progress
                )
            if step < 296:
                result = list(optimize_current(model, optimizer, arrivals[step * 32 : (step + 1) * 32].cuda(), True))
                assert result == old["trace"][step], (stream["name"], step, "trace")
                trace.append(result)
        files = [p for p in directory.rglob("*") if p.is_file()]
        unit = {
            "source": stream["name"],
            "seed": config.ssl_seed,
            "reconstruction_ssl_updates": 296,
            "classifier_fits": len(points) * len(config.support_seeds),
            "bindings": bindings,
            "exact_prior_trace": True,
            "hashes": {str(p): digest(p) for p in files},
        }
        write(directory / "complete.json", unit)
        units.append(unit)
        progress(stage="history_complete", source=stream["name"], completed=len(units))
    records = []
    for unit in units:
        directory = out / "runs" / unit["source"]
        for context, age in config.parents.items():
            pair = []
            for step in (age, age + config.horizon):
                with np.load(directory / f"at_{step}/predictions.npz") as p:
                    pair.append(p["probabilities"])
            fresh_n = len(arrays["query"])
            result = {}
            for role, ix, part, prefix in (
                ("fresh", arrays["query"], slice(0, fresh_n), ""),
                ("legacy", arrays["legacy_query"], slice(fresh_n, None), "legacy_"),
            ):
                result[role] = measure.compare(
                    pair[0][:, part],
                    pair[1][:, part],
                    arrays["labels"][ix],
                    arrays["domains"][ix],
                    resamples[prefix + "panel"],
                    resamples[prefix + "query"],
                )
            records.append(
                {"source": unit["source"], "context": context, "age": age, "horizon": config.horizon, **result}
            )
    summary = measure.summarize(records)
    assert (
        sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "source_snapshot" not in p.parts) < 512 * 2**20
    )
    write(out / "collection.json", {"complete": True, "records": records, "units": units})
    write(
        out / "summary.json",
        {
            **summary,
            "elapsed_seconds": time.perf_counter() - started,
            "reconstruction_ssl_updates": 1776,
            "downstream_fits": 432,
            "supervised_steps": 43200,
            "health_fits": 0,
            "downloads": 0,
        },
    )
    progress(stage="measurement_complete", decision=summary["decision"], gates=summary["gates"])


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
