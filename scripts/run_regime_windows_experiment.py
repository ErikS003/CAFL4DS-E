"""Sustained day/night/day GPU pilot with causal variable admission windows."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import t as student_t

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation, prediction_scores
from cafl4ds.regime_windows import CHECKPOINTS, POLICIES, WIDTHS, live_scores, run_stream
from cafl4ds.short_buffer import capture, restore
from scripts.evaluate_checkpoint_robustness import probe_predictions
from scripts.run_adaptation_experiment import _method
from scripts.run_live_loss_experiment import load_inputs as original_inputs
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest, encode

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B", "micro")


def fine_tune_gpu(encoder, images, labels, seed):
    model = copy.deepcopy(encoder).requires_grad_(True).train()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    head = torch.nn.Linear(model.embed_dim, 2).to(device)
    generator = torch.Generator().manual_seed(seed)
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.05
    )
    losses = []
    for _ in range(100):
        indices = torch.randperm(len(labels), generator=generator)[:32].to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(model.embed(images[indices])), labels[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    delta = float(
        sum(
            (p.detach() - q.detach()).square().sum()
            for p, q in zip(model.parameters(), encoder.parameters(), strict=True)
        ).sqrt()
    )
    if delta <= 0 or not np.isfinite(losses).all():
        raise ValueError("Invalid fine-tuning fit")
    return model, head, {"steps": 100, "backbone_delta_l2": delta, "initial_loss": losses[0], "final_loss": losses[-1]}


def evaluate_gpu(model, optimizer, data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    marker = fingerprint(capture(model, optimizer))
    modes = {name: module.training for name, module in model.named_modules()}
    predictions, panels, audits = {}, {}, {}
    tick = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[0]):
            encoded = encode(model.encoder, data.images)
            for draw, support in data.supports.items():
                scores = {}
                for reader in ("linear", "knn"):
                    predicted = probe_predictions(reader, encoded[support], data.labels[support], encoded[data.query])
                    scores[reader] = prediction_scores(predicted, data.labels[data.query], data.domains[data.query])
                    predictions[f"{draw}_{reader}"] = predicted.numpy()
                tuned, head, audit = fine_tune_gpu(model.encoder, data.images[support], data.labels[support], draw + 25)
                z = encode(tuned, data.images[data.query])
                with torch.no_grad():
                    predicted = head(z.cuda()).argmax(1).cpu()
                scores["finetune"] = prediction_scores(predicted, data.labels[data.query], data.domains[data.query])
                predictions[f"{draw}_finetune"] = predicted.numpy()
                panels[str(draw)], audits[str(draw)] = scores, audit
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Readout mutated complete online state")
    np.savez_compressed(directory / "predictions.npz", **predictions)
    result = {
        "panels": panels,
        "fine_tuning_audits": audits,
        "readout_seconds": time.perf_counter() - tick,
        "ssl_state_unchanged": True,
    }
    atomic_json(directory / "readout.json", result)
    return result


def load_inputs(out, frames, progress):
    (out / "original-validation-verification").mkdir(parents=True, exist_ok=True)
    config, _, old_data, _, old_lock = original_inputs(out / "original-validation-verification", progress)
    intake_path = frames / "intake.json"
    intake = json.loads(intake_path.read_text())
    if not intake["complete"] or len(intake["clips"]) != 24 or intake["frames_per_clip"] != 528:
        raise ValueError("Incomplete long-clip allocation")
    paths = [
        intake_path,
        frames / "allocation.json",
        ROOT / "docs/project-plan/slow-edge-regime-windows-frozen.md",
        Path(__file__),
        ROOT / "scripts/prepare_regime_windows.py",
        ROOT / "scripts/decode_regime_video.py",
        ROOT / "scripts/run_live_loss_experiment.py",
        Path(intake["ffmpeg_path"]),
    ]
    arrays = []
    delays = []
    for clip in intake["clips"]:
        cache = Path(clip["frames_path"])
        video = Path(clip["video_path"])
        if digest(cache) != clip["frames_sha256"] or digest(video) != clip["video_sha256"]:
            raise ValueError("Long-frame/source hash differs")
        array = np.load(cache, allow_pickle=False)
        if array.shape != (528, 64, 64, 3) or array.dtype != np.uint8:
            raise ValueError("Long-frame shape differs")
        previous = -1
        for k, (row, image) in enumerate(zip(clip["frames"], array, strict=True)):
            if (
                row["pts_seconds"] <= previous
                or row["pts_seconds"] < k / 16
                or hashlib.sha256(image.tobytes()).hexdigest() != row["rgb64_sha256"]
            ):
                raise ValueError("Frame identity/timestamp differs")
            previous = row["pts_seconds"]
            delays.append(previous - k / 16)
        arrays.append(array)
        paths.extend((cache, video, cache.parent / "manifest.json", cache.parent / "decoder.log"))
    arrivals = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
    query = old_data.query[old_data.labels[old_data.query] < 2]
    supports = {draw: ids[old_data.labels[ids] < 2] for draw, ids in old_data.supports.items()}
    if len(query) != 400 or any(len(ids) != 50 for ids in supports.values()):
        raise ValueError("Shared-task reader allocation differs")
    for domain in (0, 1):
        for label in (0, 1):
            if int(((old_data.domains[query] == domain) & (old_data.labels[query] == label)).sum()) != 100:
                raise ValueError("Unbalanced shared-task query")
    data = Evaluation(old_data.images, old_data.labels, old_data.domains, query, supports, old_data.manifest)
    np.savez_compressed(
        out / "evaluation_arrays.npz",
        labels=data.labels.numpy(),
        domains=data.domains.numpy(),
        query=query.numpy(),
        **{f"support_{d}": v.numpy() for d, v in supports.items()},
    )
    atomic_json(
        out / "data-role-manifest.json",
        {
            "clips": intake["clips"],
            "query_positions": query.tolist(),
            "supports": {str(d): v.tolist() for d, v in supports.items()},
            "classes": ["city street", "highway"],
            "development_only": True,
            "sealed_queries": False,
            "capture_sessions_known": False,
            "timestamp_slot_delay_max_seconds": max(delays),
            "timestamp_slot_delay_p99_seconds": float(np.quantile(delays, 0.99)),
        },
    )
    paths.extend((ROOT / "cafl4ds").rglob("*.py"))
    lock = {
        "hashes": {**old_lock["hashes"], **{str(p): digest(p) for p in paths}},
        "config": OmegaConf.to_container(config, resolve=True),
        "seeds": SEEDS,
        "widths": WIDTHS,
        "policies": POLICIES,
        "checkpoints": CHECKPOINTS,
        "draws": DRAWS,
        "arrivals": 12672,
        "readout_classes": 2,
        "scientific_updates": 8910,
        "primary_fits": 543,
    }
    lock = json.loads(json.dumps(lock))
    if (out / "input_lock.json").exists() and json.loads((out / "input_lock.json").read_text()) != lock:
        raise ValueError("Resume inputs changed")
    atomic_json(out / "input_lock.json", lock)
    for path in paths:
        if path.suffix in (".py", ".md") and path.is_relative_to(ROOT):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return config, arrivals, data, lock


def preflight(initial, config, arrivals, data, out):
    if (out / "preflight.json").exists():
        return json.loads((out / "preflight.json").read_text())
    torch.manual_seed(SEEDS[0])
    optimizer = instantiate(config.optim, params=list(initial.parameters()))
    parent = capture(initial, optimizer)
    marker = fingerprint(parent)
    live_scores(initial, arrivals[:128], 72003)
    if fingerprint(capture(initial, optimizer)) != marker:
        raise ValueError("Live scoring state impurity")
    work = 0
    for width in WIDTHS:
        for policy in POLICIES:
            model, optim = restore(initial, config.optim, parent)
            training = run_stream(model, optim, arrivals[:512], policy, width, SEEDS[0], (512,))
            work += training["updates"]
    nonempty = capture(model, optim)
    signatures = []
    for _ in range(2):
        model2, optim2 = restore(initial, config.optim, nonempty)
        training = run_stream(model2, optim2, arrivals[512:1024], "live_loss_half", 128, SEEDS[0], (512,))
        work += training["updates"]
        signatures.append(fingerprint(capture(model2, optim2)))
    if signatures[0] != signatures[1] or fingerprint(parent) != marker:
        raise ValueError("Exact continuation failure")
    evaluate_gpu(model, optim, data, out / "preflight")
    result = {
        "passed": True,
        "discarded_ssl_updates": work,
        "discarded_primary_fits": 3,
        "scoring_purity": True,
        "exact_cuda_continuation": True,
        "readout_isolation": True,
    }
    atomic_json(out / "preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "total_trajectories": 36, "completed_trajectories": 0}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        progress(stage="verify_inputs")
        config, arrivals, data, lock = load_inputs(out, args.frames.resolve(), progress)
        torch.manual_seed(11)
        initial = _method(config).cuda()
        progress(stage="discarded_preflight")
        preflight(initial, config, arrivals, data, out)
        if not (out / "frozen/readout.json").exists():
            torch.manual_seed(SEEDS[0])
            optim = instantiate(config.optim, params=list(initial.parameters()))
            evaluate_gpu(initial, optim, data, out / "frozen")
        records = []
        arms = [(width, policy) for width in WIDTHS for policy in POLICIES]
        for position, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            optim = instantiate(config.optim, params=list(initial.parameters()))
            parent = capture(initial, optim)
            for width, policy in arms[position:] + arms[:position]:
                directory = out / "runs" / f"w{width}" / policy / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "result.json").exists():
                    records.append(json.loads((directory / "result.json").read_text()))
                    continue
                model, optimizer = restore(initial, config.optim, parent)
                progress(stage="trajectory", width=width, policy=policy, seed=seed)

                def observe(clock, state):
                    checkpoint = directory / f"at_{clock}"
                    checkpoint.mkdir(exist_ok=True)
                    signature = fingerprint(state)
                    if (checkpoint / "state.pt").exists():
                        if (
                            fingerprint(torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True))
                            != signature
                        ):
                            raise ValueError("Partial collection checkpoint changed")
                    else:
                        torch.save(state, checkpoint / "state.pt")
                    if not (checkpoint / "readout.json").exists():
                        progress(stage="checkpoint_readout", width=width, policy=policy, seed=seed, clock=clock)
                        evaluate_gpu(model, optimizer, data, checkpoint)
                    return {
                        "state_fingerprint": signature,
                        "updates": state["updates"],
                        "pending_frames": len(state["pending_indices"]),
                        "queued_frames": len(state["queued_indices"]),
                    }

                training = run_stream(model, optimizer, arrivals, policy, width, seed, observe=observe)
                atomic_json(directory / "training.json", training)
                row = {
                    "width": width,
                    "policy": policy,
                    "seed": seed,
                    "directory": str(directory.relative_to(out)),
                    "initial_fingerprint": fingerprint(parent),
                    "updates": training["updates"],
                    "accepted_frames": training["accepted_frames"],
                    "online_seconds": training["online_seconds"],
                }
                atomic_json(directory / "result.json", row)
                records.append(row)
                progress(completed_trajectories=len(records))
        if any(digest(Path(p)) != value for p, value in lock["hashes"].items()):
            raise ValueError("Inputs changed during collection")
        atomic_json(
            out / "collection.json",
            {"records": records, "scientific_ssl_updates": sum(r["updates"] for r in records), "primary_fits": 543},
        )
        audit(initial, config, arrivals, records, out, progress)
        result = report(data, records, out)
        progress(
            stage="complete",
            completed_trajectories=36,
            scientific_ssl_updates=8910,
            primary_fits=543,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


def audit(initial, config, arrivals, records, out, progress):
    checked = 0
    for row in records:
        progress(
            stage="training_replay",
            audited_trajectories=checked,
            width=row["width"],
            policy=row["policy"],
            seed=row["seed"],
        )
        torch.manual_seed(row["seed"])
        optimizer = instantiate(config.optim, params=list(initial.parameters()))
        parent = capture(initial, optimizer)
        if fingerprint(parent) != row["initial_fingerprint"]:
            raise ValueError("Initial replay differs")
        model, optim = restore(initial, config.optim, parent)
        directory = out / row["directory"]

        def observe(clock, state):
            stored = torch.load(directory / f"at_{clock}/state.pt", map_location="cpu", weights_only=True)
            if fingerprint(stored) != fingerprint(state):
                raise ValueError("Complete checkpoint including transient buffers differs")
            return {
                "state_fingerprint": fingerprint(state),
                "updates": state["updates"],
                "pending_frames": len(state["pending_indices"]),
                "queued_frames": len(state["queued_indices"]),
            }

        repeated = run_stream(model, optim, arrivals, row["policy"], row["width"], row["seed"], observe=observe)
        stored = json.loads((directory / "training.json").read_text())
        if stored["observations"] != {str(k): v for k, v in repeated["observations"].items()}:
            raise ValueError("Observation traces differ")
        for first, second in zip(stored["blocks"], repeated["blocks"], strict=True):
            for key in (
                "start",
                "end",
                "selected_indices",
                "exposure_indices",
                "queued_indices",
                "scores",
                "scoring_seed",
                "updates",
                "loss_gradients",
            ):
                if json.dumps(first[key]) != json.dumps(second[key]):
                    raise ValueError(f"Training replay differs: {key}")
        checked += 1
    atomic_json(
        out / "training-replay-audit.json",
        {
            "passed": True,
            "trajectories": checked,
            "checkpoints": checked * 5,
            "audit_ssl_updates": sum(r["updates"] for r in records),
        },
    )


def interval(values):
    # Every contrast lies on the 1/12pp integer-count lattice (400 queries,
    # three panels; domain contrasts have twice that spacing). Preserve exact
    # zero ties when subtracting differences represented in binary floats.
    if any(abs(v - round(v * 12) / 12) > 1e-10 for v in values):
        raise ValueError("Contrast does not match exact prediction-count lattice")
    values = [round(v * 12) / 12 for v in values]
    mean = statistics.fmean(values)
    half = float(student_t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / (len(values) ** 0.5)
    return {
        "mean_pp": mean,
        "ci95_pp": [mean - half, mean + half],
        "seed_values_pp": values,
        "wins": sum(v > 0 for v in values),
        "ties": sum(v == 0 for v in values),
    }


def report(data, records, out):
    from matplotlib.figure import Figure

    counts = {}
    directories = [("frozen", out / "frozen")] + [
        (row["directory"] + f"/at_{clock}", out / row["directory"] / f"at_{clock}")
        for row in records
        for clock in CHECKPOINTS
    ]
    for name, directory in directories:
        saved = json.loads((directory / "readout.json").read_text())
        with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
            for draw in DRAWS:
                for reader in READERS:
                    pred = predictions[f"{draw}_{reader}"]
                    correct = pred == data.labels[data.query].numpy()
                    domains = data.domains[data.query].numpy()
                    measured = prediction_scores(
                        torch.from_numpy(pred.copy()), data.labels[data.query], data.domains[data.query]
                    )
                    if any(abs(measured[m] - saved["panels"][str(draw)][reader][m]) > 1e-12 for m in METRICS):
                        raise ValueError("Prediction reconstruction differs")
                    counts[name, draw, reader] = {
                        "balanced": int(correct.sum()),
                        "micro": int(correct.sum()),
                        "A": int(correct[domains == 0].sum()),
                        "B": int(correct[domains == 1].sum()),
                    }
    by = {(r["width"], r["policy"], r["seed"]): r for r in records}

    def score(width, policy, seed, clock, metric, reader="finetune"):
        name = "frozen" if policy == "frozen" else by[width, policy, seed]["directory"] + f"/at_{clock}"
        denominator = 200 if metric in ("A", "B") else 400
        return 100 * sum(counts[name, draw, reader][metric] for draw in DRAWS) / (denominator * 3)

    frozen = {m: score(32, "frozen", SEEDS[0], 0, m) for m in METRICS}
    results = {}
    for width in WIDTHS:
        for policy in POLICIES:
            trajectory = {
                str(clock): {m: statistics.fmean(score(width, policy, s, clock, m) for s in SEEDS) for m in METRICS}
                for clock in CHECKPOINTS
            }
            contrasts = {}
            pairs = {
                "initial_day_learning": (4224, 0, "A"),
                "day_retention_change": (8448, 4224, "A"),
                "night_adaptation": (8448, 4224, "B"),
                "early_night_response": (4480, 4224, "B"),
                "day_recovery": (12672, 8448, "A"),
                "early_day_recovery": (8704, 8448, "A"),
                "final_day_vs_first_day": (12672, 4224, "A"),
                "night_retention_on_return": (12672, 8448, "B"),
            }
            for name, (first, second, metric) in pairs.items():
                contrasts[name] = interval(
                    [
                        score(width, policy, s, first, metric)
                        - (frozen[metric] if second == 0 else score(width, policy, s, second, metric))
                        for s in SEEDS
                    ]
                )
            contrasts["final_balanced_vs_frozen"] = interval(
                [score(width, policy, s, 12672, "balanced") - frozen["balanced"] for s in SEEDS]
            )
            full = {
                str(clock): {
                    m: interval([score(width, policy, s, clock, m) - score(width, "full", s, clock, m) for s in SEEDS])
                    for m in METRICS
                }
                for clock in CHECKPOINTS
            }
            retention = interval(
                [
                    (score(width, policy, s, 8448, "A") - score(width, policy, s, 4224, "A"))
                    - (score(width, "full", s, 8448, "A") - score(width, "full", s, 4224, "A"))
                    for s in SEEDS
                ]
            )
            savings = [
                1 - by[width, policy, s]["online_seconds"] / by[width, "full", s]["online_seconds"] for s in SEEDS
            ]
            useful = contrasts["final_balanced_vs_frozen"]["ci95_pp"][0] > 0
            guard = (
                all(full["12672"][m]["ci95_pp"][0] > -1 for m in ("balanced", "A", "B"))
                and retention["ci95_pp"][0] > -1
            )
            results[f"w{width}/{policy}"] = {
                "trajectory_accuracy_percent": trajectory,
                "contrasts": contrasts,
                "matched_full": full,
                "retention_change_minus_full": retention,
                "mean_processing_saving_fraction": statistics.fmean(savings),
                "paired_processing_savings": savings,
                "useful_final_ssl": useful,
                "absence_decline_provisionally_supported": contrasts["initial_day_learning"]["ci95_pp"][0] > 0
                and contrasts["day_retention_change"]["ci95_pp"][1] < 0,
                "efficiency_pilot_lead": policy != "full" and useful and guard and statistics.fmean(savings) >= 0.25,
                "content_vs_uniform_random": {
                    other: {
                        str(clock): interval(
                            [
                                score(width, policy, s, clock, "balanced") - score(width, other, s, clock, "balanced")
                                for s in SEEDS
                            ]
                        )
                        for clock in CHECKPOINTS
                    }
                    for other in ("uniform_half", "random_half")
                },
            }
    decision = (
        "provisional_efficiency_lead_requires_confirmation"
        if any(r["efficiency_pilot_lead"] for r in results.values())
        else "no_efficiency_pilot_lead_qualified"
    )
    payload = {
        "decision": decision,
        "frozen_accuracy_percent": frozen,
        "arms": results,
        "scientific_ssl_updates": 8910,
        "primary_fits": 543,
        "readout_prediction_sets": 1629,
        "readout_audit_passed": True,
        "development_only": True,
        "uncertainty": "paired t df2 across three optimization seeds on reused constructed clips; no source independence/multiplicity correction",
        "window_delay_arrivals": {str(w): w - 1 for w in WIDTHS},
        "width32_half_training_queue_note": "16 admitted frames carry to the next window; earliest full update after64 arrivals",
    }
    atomic_json(out / "summary.json", payload)
    lines = [
        "# Sustained day/night/day window/filter pilot",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Window | Policy | Day after day1 % | Day after night % | Day final % | Night after day1 % | Night after night % | Night final % | Final balanced % | Processing saving % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        v = row["trajectory_accuracy_percent"]
        lines.append(
            f"| {name.split('/')[0]} | {name.split('/')[1]} | {v['4224']['A']:.3f} | {v['8448']['A']:.3f} | {v['12672']['A']:.3f} | {v['4224']['B']:.3f} | {v['8448']['B']:.3f} | {v['12672']['B']:.3f} | {v['12672']['balanced']:.3f} | {100 * row['mean_processing_saving_fraction']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Frozen day/night/balanced: {frozen}.",
            "",
            "Three-seed conditional development pilot, two shared scene classes. No forgetting assumed; sampled endpoint changes do not isolate absence from continued training without a no-training continuation. Buffers cross regimes without flushing; checkpoints retain pending and accepted-but-untrained frames. No replay or health steering.",
            "",
            "All36 trajectories/8910 scientific updates and543 primary fits completed. Exact training/transient-buffer replay and1629 raw prediction reconstructions passed. Independent verification is recorded separately. Processing charges every live-loss score; decoding/readouts/checkpoint IO are excluded.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    fig = Figure(figsize=(13, 8), layout="constrained")
    axes = fig.subplots(3, 2)
    for i, width in enumerate(WIDTHS):
        for j, metric in enumerate(("A", "B")):
            axis = axes[i, j]
            for policy in POLICIES:
                row = results[f"w{width}/{policy}"]["trajectory_accuracy_percent"]
                axis.plot(
                    [0, *CHECKPOINTS],
                    [frozen[metric], *[row[str(clock)][metric] for clock in CHECKPOINTS]],
                    marker="o",
                    label=policy,
                )
            for boundary in (4224, 8448):
                axis.axvline(boundary, color="gray", linestyle="--")
            axis.axhline(frozen[metric], color="black", linewidth=0.7, linestyle=":")
            axis.set(
                title=f"Window {width}: {'day' if metric == 'A' else 'night'}",
                xlabel="Arrivals (day -> night -> day)",
                ylabel="Mean fine-tuning accuracy (%)",
            )
            if i == 0 and j == 0:
                axis.legend(fontsize=8)
    fig.savefig(out / "trajectories.png", dpi=140)
    return payload


if __name__ == "__main__":
    main()
