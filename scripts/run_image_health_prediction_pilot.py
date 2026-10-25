"""Collect separate marginal-image and single-image health prediction benchmarks."""

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
import torch.nn.functional as F
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import embeddings
from cafl4ds.image_health_prediction import image_step, variance_gradient
from cafl4ds.prediction_benchmark import (
    correlation,
    health_effect,
    health_values,
    held_group_ridge,
    improvement,
    scores,
)
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import cases, incoming, learner, reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
HEALTH = ROOT / "outputs/health-calibration/20261013-gpu"
BUDGET = ROOT / "outputs/budget-reservation/20261013-gpu"
SLOTS = (0, 8, 16, 24)
DRAWS = (71011, 71023)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_difference(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a if a[k].is_floating_point())


def direction_forecast(g, names, full, omitted, variance):
    return (
        sum(
            float(
                (
                    gradient.double()
                    * (
                        full["encoder." + name].to(gradient.device).double()
                        - omitted["encoder." + name].to(gradient.device).double()
                    )
                ).sum()
            )
            for name, gradient in zip(names, g, strict=True)
        )
        / variance
    )


def branch(initial, config, parent, images, bank, omit=None, reconstruct=False, ordinary=False):
    """One paid update and invariant observer; caller explicitly repeats every branch."""
    synchronize(torch.device("cuda"))
    tick = time.perf_counter()
    model, optimizer = restore(initial, config.optim, parent)
    synchronize(torch.device("cuda"))
    clone_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    if ordinary:
        loss, norm = optimize_current(model, optimizer, images.cuda(), True)
        record = {"loss": loss, "gradient_norm": norm}
    else:
        record = image_step(model, optimizer, images.cuda(), omit, reconstruct)
    synchronize(torch.device("cuda"))
    update_seconds = time.perf_counter() - tick
    end = capture(model, optimizer)
    marker = fingerprint(end)
    tick = time.perf_counter()
    z = embeddings(model, bank)
    synchronize(torch.device("cuda"))
    observer_seconds = time.perf_counter() - tick
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Endpoint observer changed complete learner")
    return (
        record,
        end,
        z,
        {
            "clone_seconds": clone_seconds,
            "update_seconds": update_seconds,
            "endpoint_observation_seconds": observer_seconds,
        },
    )


def paired_branch(*args, **kwargs):
    first = branch(*args, **kwargs)
    second = branch(*args, **kwargs)
    if (
        first[0] != second[0]
        or fingerprint(first[1]) != fingerprint(second[1])
        or not np.array_equal(first[2], second[2])
    ):
        raise ValueError("Exact scientific branch replay failed")
    return first


def candidate_losses(initial, config, parent, data):
    """Pay for isolated pre-update loss forwards without consulting trained branches."""
    model, optimizer = restore(initial, config.optim, parent)
    marker = fingerprint(capture(model, optimizer))
    tick = time.perf_counter()
    values = []
    for batch in [data] + [data[s : s + 1] for s in SLOTS]:
        with torch.random.fork_rng(devices=[0]):
            torch.set_rng_state(parent["cpu_rng"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in parent["cuda_rng"]])
            model.train()
            values.append(model.per_sample_loss(batch.cuda()).cpu().tolist())
    synchronize(torch.device("cuda"))
    elapsed = time.perf_counter() - tick
    # train() is part of the private observation; restore original module flags.
    for name, module in model.named_modules():
        module.training = parent["modes"][name]
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Candidate-loss observer changed learner")
    return values[0], {s: v[0] for s, v in zip(SLOTS, values[1:], strict=True)}, elapsed


def analyze(records):
    result = {}
    state_fields = ("updates", "parent_rank", "parent_variance", "parent_covariance", "parent_cka")
    fields = state_fields + ("rgb_mean", "rgb_std", "edge", "reconstruction_loss", "centroid_distance", "loss_variance")
    for intervention in ("marginal", "single"):
        rows = [r for r in records if r["intervention"] == intervention]
        groups = np.array([r["stream"] for r in rows])
        metrics = {}
        for metric in ("variance", "covariance", "rank", "cka", "cosine"):
            y = np.array([r["effect"][metric] for r in rows])
            predictions, folds = {"zero": np.zeros(len(y))}, {}
            for name, features in (("mean", ()), ("state", state_fields), ("image_state", fields)):
                x = np.array([[r["features"][f] for f in features] for r in rows]).reshape(len(rows), len(features))
                predictions[name], folds[name] = held_group_ridge(x, y, groups)
            if metric == "variance":
                predictions["adamw_derivative"] = np.array([r["variance_forecast"] for r in rows])
            scored = scores(y, predictions, groups)
            contrasts = {}
            if metric == "variance":
                for candidate in ("image_state", "adamw_derivative"):
                    contrasts[candidate] = {b: improvement(scored[candidate], scored[b], 4) for b in ("zero", "state")}
            ranking = {}
            for name, values in predictions.items():
                correlations = []
                for stream in sorted(set(groups)):
                    for checkpoint in ("before_shift", "stable_second", "before_return"):
                        for draw in DRAWS:
                            select = np.array(
                                [
                                    r["stream"] == stream and r["checkpoint"] == checkpoint and r["draw"] == draw
                                    for r in rows
                                ]
                            )
                            c = correlation(y[select], np.asarray(values)[select], rank=True)
                            if c is not None:
                                correlations.append(c)
                ranking[name] = {
                    "mean_within_parent_draw_spearman": float(np.mean(correlations)) if correlations else None,
                    "defined_panels": len(correlations),
                }
            draw_values = [[r["effect"][metric] for r in rows if r["draw"] == draw] for draw in DRAWS]
            metrics[metric] = {
                "targets": y.tolist(),
                "models": scored,
                "folds": folds,
                "primary_contrasts": contrasts,
                "rankings": ranking,
                "draw_repeat_correlation": correlation(*draw_values),
            }
        primary = metrics["variance"]
        result[intervention] = {
            "rows": len(rows),
            "groups": groups.tolist(),
            "metrics": metrics,
            "candidate_screen_passed": {
                c: all(t["passes"] for t in tests.values()) for c, tests in primary["primary_contrasts"].items()
            },
        }
    return {
        "role": "reused_history_new_image_interventions_development",
        "primary": "marginal_variance",
        "interventions": result,
        "controller_trained": False,
        "independent_source_confirmation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable scientific run")
    if shutil.disk_usage(out).free < 512 * 1024**2:
        raise ValueError("Launch storage reserve")
    setup()
    start = time.perf_counter()
    old = read(HEALTH / "input_lock.json")
    if not read(HEALTH / "independent-verification.json")["passed"]:
        raise ValueError("Prior health evidence not audited")
    selected = [r for r in cases() if r["role"] == "six_stream" and r["seed"] == 5003]
    if len(selected) != 18:
        raise ValueError("Expected 18 parents")
    intake_path = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"
    intake = read(intake_path)
    streams = intake["streams"]
    clip_sets = [set(c for period in s["periods"] for c in period) for s in streams]
    if any(a & b for i, a in enumerate(clip_sets) for b in clip_sets[i + 1 :]):
        raise ValueError("Cross-history clip reuse")
    paths = [
        Path(__file__),
        ROOT / "cafl4ds/image_health_prediction.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "docs/project-plan/prediction-image-health-pilot-frozen.md",
        HEALTH / "reference.json",
        intake_path,
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "cafl4ds/health_calibration.py",
        ROOT / "cafl4ds/ssl/mae.py",
        ROOT / "scripts/run_health_measurement.py",
        ROOT / "scripts/run_retention_anchor_signal.py",
        ROOT / "uv.lock",
    ]
    paths += [Path(c["parent"]) for c in selected]
    paths += [Path(c["frames_path"]) for c in intake["clips"]]
    hashes = {**old["hashes"], **{str(p): digest(p) for p in paths}}
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Inherited inputs changed")
    atomic_json(
        out / "input_lock.json",
        {
            "hashes": hashes,
            "cases": selected,
            "config": old["config"],
            "slots": SLOTS,
            "draws": DRAWS,
            "clip_disjoint": True,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "role": "reused_development",
        },
    )
    for path in paths:
        if path.suffix in (".py", ".md"):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    config = OmegaConf.create(old["config"])
    bank = reference_images(read(HEALTH / "reference.json"))
    torch.manual_seed(11)
    initial = _method(config).cuda()
    warm = embeddings(initial, bank)
    np.save(out / "warm.npy", warm)
    records, checks, costs = [], [], []
    scientific_updates = 0
    for case_number, case in enumerate(selected):
        if shutil.disk_usage(out).free < 256 * 1024**2:
            raise ValueError("Write storage reserve")
        data = incoming(case, intake, {})
        parent = learner(Path(case["parent"]))
        model, optimizer = restore(initial, config.optim, parent)
        marker = fingerprint(capture(model, optimizer))
        tick = time.perf_counter()
        before = embeddings(model, bank)
        candidate_z = embeddings(model, data)
        synchronize(torch.device("cuda"))
        observation_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        g, variance = variance_gradient(model, bank)
        synchronize(torch.device("cuda"))
        derivative_seconds = time.perf_counter() - tick
        names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
        if fingerprint(capture(model, optimizer)) != marker:
            raise ValueError("Pre-decision observer changed parent")
        baseline = health_values(before, warm)
        if abs(variance - baseline["variance"]) > 1e-12:
            raise ValueError("Derivative/NumPy metric mismatch")
        q = F.normalize(torch.from_numpy(candidate_z).double(), dim=1)
        center = F.normalize(torch.from_numpy(before).double(), dim=1).mean(0)
        np.savez_compressed(out / f"parent_{case_number:02d}.npz", before=before, candidate=candidate_z)
        for draw in DRAWS:
            private = copy.deepcopy(parent)
            seed = draw + case["point"]
            private["cpu_rng"] = torch.Generator().manual_seed(seed).get_state()
            private["cuda_rng"] = [torch.Generator(device="cuda").manual_seed(seed).get_state()]
            loss_observations, single_losses, loss_seconds = candidate_losses(initial, config, private, data)
            full = paired_branch(initial, config, private, data, bank)
            if full[0]["per_image_losses"] != loss_observations:
                raise ValueError("Pre-update candidate losses differ from actual training views")
            scientific_updates += 1
            ordinary = branch(initial, config, private, data, bank, ordinary=True)
            ordinary_error = state_difference(full[1]["model"], ordinary[1]["model"])
            if abs(full[0]["loss"] - ordinary[0]["loss"]) > 1e-7 or ordinary_error > 2e-7:
                raise ValueError("Weighted full differs from ordinary batch32")
            full_values = health_values(full[2], warm)
            np.savez_compressed(out / f"full_{case_number:02d}_{draw}.npz", after=full[2])
            costs.append(
                {
                    "kind": "full",
                    **full[3],
                    "observer_seconds": observation_seconds,
                    "derivative_seconds": derivative_seconds,
                    "candidate_loss_forward_seconds": loss_seconds,
                }
            )
            for slot in SLOTS:
                omitted = paired_branch(initial, config, private, data, bank, omit=slot)
                single = paired_branch(initial, config, private, data[slot : slot + 1], bank)
                if single[0]["per_image_losses"][0] != single_losses[slot]:
                    raise ValueError("Single-image pre-update loss differs from training")
                scientific_updates += 2
                if omitted[0]["per_image_losses"] != full[0]["per_image_losses"]:
                    raise ValueError("Omission altered another image's training view/loss")
                subtraction_error = None
                if slot == 0:
                    subtracted = branch(initial, config, private, data, bank, omit=slot, reconstruct=True)
                    subtraction_error = state_difference(omitted[1]["model"], subtracted[1]["model"])
                    if subtraction_error > 2e-7:
                        raise ValueError("Gradient subtraction control failed")
                image = data[slot]
                loss = loss_observations[slot]
                features = {
                    "updates": case["point"] // 64,
                    **{"parent_" + k: baseline[k] for k in ("rank", "variance", "covariance", "cka")},
                    "rgb_mean": float(image.mean()),
                    "rgb_std": float(image.std()),
                    "edge": float(
                        (image[:, 1:] - image[:, :-1]).abs().mean() + (image[:, :, 1:] - image[:, :, :-1]).abs().mean()
                    ),
                    "reconstruction_loss": loss,
                    "centroid_distance": float((q[slot] - center).norm()),
                    "loss_variance": loss * baseline["variance"],
                }
                # Both forecasts are complete before endpoint values enter regression data.
                marginal_forecast = direction_forecast(g, names, full[1]["model"], omitted[1]["model"], variance)
                single_forecast = direction_forecast(g, names, single[1]["model"], private["model"], variance)
                for intervention, effect, forecast, loss_feature in (
                    (
                        "marginal",
                        health_effect(baseline, full_values, health_values(omitted[2], warm)),
                        marginal_forecast,
                        loss,
                    ),
                    (
                        "single",
                        health_effect(baseline, health_values(single[2], warm), baseline),
                        single_forecast,
                        single_losses[slot],
                    ),
                ):
                    row_features = {
                        **features,
                        "reconstruction_loss": loss_feature,
                        "loss_variance": loss_feature * baseline["variance"],
                    }
                    records.append(
                        {
                            "stream": case["stream"],
                            "checkpoint": case["checkpoint"],
                            "seed": case["seed"],
                            "point": case["point"],
                            "draw": draw,
                            "slot": slot,
                            "intervention": intervention,
                            "features": row_features,
                            "effect": effect,
                            "variance_forecast": forecast,
                            "parent_fingerprint": marker,
                            "branch_fingerprint": fingerprint(omitted[1] if intervention == "marginal" else single[1]),
                        }
                    )
                np.savez_compressed(
                    out / f"image_{case_number:02d}_{draw}_{slot}.npz", omitted=omitted[2], single=single[2]
                )
                costs.extend([{"kind": "omitted", **omitted[3]}, {"kind": "single", **single[3]}])
                checks.append(
                    {
                        "parent": case_number,
                        "draw": draw,
                        "slot": slot,
                        "ordinary_parameter_error": ordinary_error,
                        "subtraction_parameter_error": subtraction_error,
                        "exact_replay": True,
                        "matched_background_losses": True,
                    }
                )
            atomic_json(out / "observations.json", {"complete": False, "records": records})
        print(
            json.dumps({"parents": case_number + 1, "total": 18, "elapsed_seconds": time.perf_counter() - start}),
            flush=True,
        )
    atomic_json(out / "observations.json", {"complete": True, "records": records})
    atomic_json(
        out / "costs.json",
        {
            "records": costs,
            "scientific_updates": scientific_updates,
            "replay_updates": scientific_updates,
            "control_updates": 72,
            "total_updates": scientific_updates * 2 + 72,
            "total_seconds": time.perf_counter() - start,
            "decoding_downloads_and_downstream_readouts_included": False,
        },
    )
    atomic_json(out / "summary.json", analyze(records))
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Locked inputs changed")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if scientific_updates != 324 or size > 96 * 1024**2:
        raise ValueError("Collection/work/output cap mismatch")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "checks": checks,
            "scientific_updates": scientific_updates,
            "total_updates": scientific_updates * 2 + 72,
            "observer_invariance": True,
            "clip_disjoint": True,
            "locked_inputs": len(hashes),
            "output_mib": size / 1024**2,
        },
    )
    print(json.dumps({"complete": True, "scientific_updates": scientific_updates}), flush=True)


if __name__ == "__main__":
    main()
