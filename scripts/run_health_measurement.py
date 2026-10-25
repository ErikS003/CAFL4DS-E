"""Fixed-reference MAE health calibration with beneficial-motion controls."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import spearmanr, t

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import FIELDS, changes, embeddings, forecasts, geometry
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.state_horizon import STATES
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import gather, setup
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "outputs/budget-reservation/20261013-gpu"
ANCHOR = ROOT / "outputs/retention-anchor/20261013-gpu"
CONTROL = ROOT / "outputs/state-horizon/20261013-gpu"
ORIGINAL = ROOT / "outputs/regime-windows/20261013-gpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def learner(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("learner", state)


def reference_images(manifest):
    images = [
        np.asarray(Image.open(row["path"]).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)).copy()
        for row in manifest["images"]
    ]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255


def cases():
    rows = read(BUDGET / "collection.json")["records"]
    result = []
    for r in rows:
        if r["arm"] != "spend":
            continue
        hold = next(
            h
            for h in rows
            if h["stream"] == r["stream"]
            and h["checkpoint"] == r["checkpoint"]
            and h["seed"] == r["seed"]
            and h["arm"] == "hold"
        )
        result.append(
            {
                "role": "six_stream",
                "stream": r["stream"],
                "checkpoint": r["checkpoint"],
                "seed": r["seed"],
                "point": r["point"],
                "parent": str(BUDGET / "parents" / r["stream"] / r["checkpoint"] / f"seed_{r['seed']}/state.pt"),
                "actual": str(BUDGET / r["immediate"] / "state.pt"),
                "hold": str(BUDGET / hold["immediate"] / "state.pt"),
                "training": str(BUDGET / r["directory"] / "training.json"),
            }
        )
    for state, point in STATES.items():
        for seed in (5003, 5009, 5011):
            base = CONTROL / "runs" / state
            result.append(
                {
                    "role": "control",
                    "stream": "original",
                    "checkpoint": state,
                    "seed": seed,
                    "point": point,
                    "parent": str(ORIGINAL / "runs/w32/full" / f"seed_{seed}/at_{point}/state.pt"),
                    "actual": str(base / "full" / f"seed_{seed}/at_1024/state.pt"),
                    "hold": str(base / "hold" / f"seed_{seed}/at_1024/state.pt"),
                    "training": str(base / "full" / f"seed_{seed}/training.json"),
                }
            )
    return result


def incoming(case, intake, original):
    if case["role"] == "six_stream":
        stream = next(s for s in intake["streams"] if s["name"] == case["stream"])
        return gather(
            stream, {c["clip_id"]: c for c in intake["clips"]}, list(range(case["point"], case["point"] + 32))
        )
    clips = original["clips"]
    arrays = {
        i // 528: np.load(clips[i // 528]["frames_path"], mmap_mode="r")
        for i in range(case["point"], case["point"] + 32)
    }
    images = np.stack([arrays[i // 528][i % 528] for i in range(case["point"], case["point"] + 32)])
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255


def collect(case, initial, config, bank, warm, data):
    model, optimizer = restore(initial, config.optim, learner(Path(case["parent"])))
    before_marker = fingerprint(capture(model, optimizer))
    device = next(model.parameters()).device
    synchronize(device)
    tick = time.perf_counter()
    before = embeddings(model, bank)
    synchronize(device)
    before_seconds = time.perf_counter() - tick
    if fingerprint(capture(model, optimizer)) != before_marker:
        raise ValueError("Reference reading changed parent")
    tick = time.perf_counter()
    update = optimize_current(model, optimizer, data.cuda(), True)
    synchronize(device)
    update_seconds = time.perf_counter() - tick
    virtual_marker = fingerprint(capture(model, optimizer))
    tick = time.perf_counter()
    virtual = embeddings(model, bank)
    synchronize(device)
    virtual_seconds = time.perf_counter() - tick
    if fingerprint(capture(model, optimizer)) != virtual_marker:
        raise ValueError("Reference reading changed virtual state")
    trace = read(Path(case["training"]))
    expected = (
        trace["records"][0]["loss_gradient"]
        if case["role"] == "six_stream"
        else trace["blocks"][0]["loss_gradients"][0]
    )
    if list(update) != expected:
        raise ValueError("Virtual first update disagrees with audited actual branch")
    z = {"before": before, "virtual": virtual}
    for name in ("actual", "hold"):
        model, optimizer = restore(initial, config.optim, learner(Path(case[name])))
        marker = fingerprint(capture(model, optimizer))
        z[name] = embeddings(model, bank)
        if fingerprint(capture(model, optimizer)) != marker:
            raise ValueError("Actual/hold observer changed complete state")
    if not np.array_equal(z["before"], z["hold"]):
        raise ValueError("Hold representation changed")
    tick = time.perf_counter()
    metrics = {name: changes(before, z[name]) for name in ("virtual", "actual", "hold")}
    snapshot = changes(warm, before)
    geometry_seconds = time.perf_counter() - tick
    return {
        **case,
        "parent_fingerprint": before_marker,
        "virtual_fingerprint": virtual_marker,
        "loss_gradient": list(update),
        "snapshot": snapshot,
        "geometry_before": geometry(before),
        "changes": metrics,
        "before_embedding_seconds": before_seconds,
        "virtual_embedding_seconds": virtual_seconds,
        "preview_update_seconds": update_seconds,
        "geometry_seconds_offline": geometry_seconds,
    }, z


def summarize(records):
    points = json.loads(json.dumps(read(ANCHOR / "summary.json")["points"]))
    for point in points:
        matches = [
            r
            for r in records
            if r["role"] == "six_stream" and r["stream"] == point["stream"] and r["checkpoint"] == point["checkpoint"]
        ]
        for field in FIELDS:
            point["v_" + field] = statistics.fmean(r["changes"]["virtual"][field] for r in matches)
            point["a_" + field] = statistics.fmean(r["changes"]["actual"][field] for r in matches)
            point["s_" + field] = statistics.fmean(r["snapshot"][field] for r in matches)
    models = {
        "mean": (),
        "age_loss": ("updates", "incoming_loss"),
        "virtual_cka": ("updates", "incoming_loss", "v_cka_drift"),
        "snapshot_cka": ("updates", "incoming_loss", "s_cka_drift"),
        "virtual_bundle": ("updates", "incoming_loss") + tuple("v_" + f for f in FIELDS),
    }
    predicted = {name: forecasts(points, columns) for name, columns in models.items()}
    contrasts = {}
    for baseline in ("mean", "age_loss"):
        delta = np.array(
            [
                predicted[baseline]["source_mse"][g] - predicted["virtual_cka"]["source_mse"][g]
                for g in sorted(predicted[baseline]["source_mse"])
            ]
        )
        width = float(t.ppf(0.9875, 5) * delta.std(ddof=1) / np.sqrt(6))
        contrasts[baseline] = {
            "mean_mse_improvement": float(delta.mean()),
            "adjusted_ci95": [float(delta.mean() - width), float(delta.mean() + width)],
            "reduction_fraction": 1 - predicted["virtual_cka"]["mse"] / predicted[baseline]["mse"],
        }
    control = {
        state: {
            field: [
                next(r for r in records if r["role"] == "control" and r["checkpoint"] == state and r["seed"] == seed)[
                    "changes"
                ]["actual"][field]
                for seed in (5003, 5009, 5011)
            ]
            for field in FIELDS
        }
        for state in STATES
    }
    separation = all(
        control["day_end"]["cka_drift"][i] < control[bad]["cka_drift"][i]
        for bad in ("night_end", "return_early")
        for i in range(3)
    )
    gates = {
        "incremental_prediction": all(
            v["reduction_fraction"] >= 0.1 and v["adjusted_ci95"][0] > 0 for v in contrasts.values()
        ),
        "positive_virtual_cka_folds": sum(
            v["v_cka_drift"] > 0 for v in predicted["virtual_cka"]["coefficients"].values()
        )
        >= 5,
        "beneficial_motion_quiet_control": separation,
    }
    correlations = {
        kind: {
            field: float(
                spearmanr([p[kind + "_" + field] for p in points], [p["terminal_damage"] for p in points]).statistic
            )
            for field in FIELDS
        }
        for kind in ("v", "a", "s")
    }
    return {
        "points": points,
        "models": predicted,
        "primary_contrasts": contrasts,
        "controls": control,
        "control_primary_capability_effects": read(CONTROL / "summary.json")["cells"],
        "gates": gates,
        "health_candidate_qualified": all(gates.values()),
        "rank_correlations_descriptive": correlations,
        "control_population_independent": False,
        "controller_trained": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable scientific run")
    if shutil.disk_usage(out).free < 512 * 1024**2:
        raise ValueError("Insufficient launch storage reserve")
    start = time.perf_counter()

    def progress(**fields):
        status = {**fields, "elapsed_seconds": time.perf_counter() - start}
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    progress(stage="locking")
    setup()
    old = read(ANCHOR / "input_lock.json")
    if (
        not read(ANCHOR / "independent-verification.json")["passed"]
        or not read(CONTROL / "independent-verification.json")["passed"]
    ):
        raise ValueError("Prior evidence not audited")
    reference = read(out / "reference.json")
    case_list = cases()
    paths = [
        Path(__file__),
        ROOT / "scripts/verify_health_measurement.py",
        ROOT / "scripts/prepare_health_reference.py",
        ROOT / "cafl4ds/health_calibration.py",
        ROOT / "docs/project-plan/slow-edge-health-measurement-frozen.md",
        out / "reference.json",
        ANCHOR / "input_lock.json",
        ANCHOR / "summary.json",
        ANCHOR / "independent-verification.json",
        CONTROL / "summary.json",
        CONTROL / "independent-verification.json",
    ]
    paths += [Path(row[k]) for row in case_list for k in ("parent", "actual", "hold", "training")]
    paths += [Path(row[k]) for row in reference["images"] for k in ("path", "label_path")]
    hashes = {**old["hashes"], **{str(p): digest(p) for p in paths}}
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Inherited/reference inputs changed")
    lock = {"hashes": hashes, "config": old["config"], "cases": case_list, "reference": reference}
    atomic_json(out / "input_lock.json", lock)
    for path in paths:
        if path.suffix in (".py", ".md"):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    config = OmegaConf.create(old["config"])
    bank = reference_images(reference)
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    original = read(ORIGINAL / "data-role-manifest.json")
    torch.manual_seed(11)
    initial = _method(config).cuda()
    warm = embeddings(initial, bank)
    np.save(out / "warm_embeddings.npy", warm)
    records = []
    for case in case_list:
        if shutil.disk_usage(out).free < 256 * 1024**2:
            raise ValueError("Storage reserve threatened")
        data = incoming(case, intake, original)
        if not records:
            pre = [collect(case, initial, config, bank, warm, data) for _ in range(2)]
            if pre[0][0]["virtual_fingerprint"] != pre[1][0]["virtual_fingerprint"]:
                raise ValueError("Preflight differs")
            atomic_json(out / "preflight.json", {"passed": True, "discarded_virtual_updates": 2})
        record, z = collect(case, initial, config, bank, warm, data)
        repeated, zz = collect(case, initial, config, bank, warm, data)
        if any(not np.array_equal(z[k], zz[k]) for k in z) or any(
            record[k] != repeated[k]
            for k in ("changes", "snapshot", "loss_gradient", "parent_fingerprint", "virtual_fingerprint")
        ):
            raise ValueError("Readings/virtual state replay differs")
        directory = out / "embeddings" / case["role"] / case["stream"] / case["checkpoint"] / f"seed_{case['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "features.npz", **z)
        record["features"] = str((directory / "features.npz").relative_to(out))
        records.append(record)
        atomic_json(out / "observations.json", {"complete": len(records) == 63, "records": records})
        progress(stage="collection_and_replay", completed=len(records), total=63)
    summary = summarize(records)
    summary["costs"] = {
        key: statistics.fmean(r[key] for r in records if r["role"] == "six_stream")
        for key in (
            "before_embedding_seconds",
            "virtual_embedding_seconds",
            "preview_update_seconds",
            "geometry_seconds_offline",
        )
    }
    atomic_json(out / "summary.json", summary)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Final input lock differs")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 64 * 1024**2:
        raise ValueError("Output cap exceeded")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "virtual_updates": 63,
            "replay_updates": 63,
            "actual_first_steps_checked": 63,
            "hold_bank_equalities": 63,
            "observer_invariance": True,
            "output_mib": size / 1024**2,
            "locked_inputs": len(hashes),
        },
    )
    progress(stage="complete", health_candidate_qualified=summary["health_candidate_qualified"])


if __name__ == "__main__":
    main()
