"""Matched exposure gradual/abrupt historical-diet continuation from identical night-trained states."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.diet_smoothing import ACTIONS, CHECKPOINTS, run_branch, set_history
from cafl4ds.short_buffer import capture, restore
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import evaluate_gpu
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/regime-windows/20261013-gpu"
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
CONTEXTS = ("day2", "night2")
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B")


def load_inputs(out, night_frames):
    verified = json.loads((PRIOR / "independent-verification.json").read_text())
    old_lock = json.loads((PRIOR / "input_lock.json").read_text())
    if not verified["passed"] or any(digest(Path(p)) != v for p, v in old_lock["hashes"].items()):
        raise ValueError("Prior verified scientific inputs changed")
    role = json.loads((PRIOR / "data-role-manifest.json").read_text())
    next_night = json.loads((night_frames / "intake.json").read_text())
    if not next_night["complete"] or len(next_night["clips"]) != 8:
        raise ValueError("Continued-night intake incomplete")
    contexts = {"day2": [c for c in role["clips"] if c["visit"] == "day2"], "night2": next_night["clips"]}
    prior_ids = {c["clip_id"] for c in role["clips"]}
    if prior_ids & {c["clip_id"] for c in contexts["night2"]}:
        raise ValueError("Continued-night source reuse in parent history")
    paths = [
        PRIOR / "independent-verification.json",
        PRIOR / "input_lock.json",
        PRIOR / "data-role-manifest.json",
        PRIOR / "evaluation_arrays.npz",
        PRIOR / "frozen/predictions.npz",
        night_frames / "intake.json",
        night_frames / "allocation.json",
        ROOT / "docs/project-plan/slow-edge-diet-smoothing-frozen.md",
        Path(__file__),
        ROOT / "scripts/prepare_return_action_control.py",
        ROOT / "scripts/verify_diet_smoothing_experiment.py",
        ROOT / "scripts/decode_regime_video.py",
        ROOT / "scripts/run_regime_windows_experiment.py",
        ROOT / "scripts/run_adaptation_experiment.py",
        ROOT / "scripts/run_paired_update_response.py",
        ROOT / "scripts/run_short_buffer_experiment.py",
    ]
    arrays = {}
    for context, clips in contexts.items():
        if len(clips) != 8 or sum(c["keyframe_attributes"]["scene"] == "city street" for c in clips) != 4:
            raise ValueError("Continuation scene counts differ")
        values = []
        for clip in clips:
            cache = Path(clip["frames_path"])
            video = Path(clip["video_path"])
            if digest(cache) != clip["frames_sha256"] or digest(video) != clip["video_sha256"]:
                raise ValueError("Continuation source/cache hash differs")
            frames = np.load(cache, allow_pickle=False)
            if frames.shape != (528, 64, 64, 3) or frames.dtype != np.uint8:
                raise ValueError("Bad continuation frames")
            previous = -1
            for k, (metadata, image) in enumerate(zip(clip["frames"], frames, strict=True)):
                if metadata["pts_seconds"] <= previous or metadata["pts_seconds"] < k / 16:
                    raise ValueError("Invalid source timestamp")
                import hashlib

                if hashlib.sha256(image.tobytes()).hexdigest() != metadata["rgb64_sha256"]:
                    raise ValueError("Frame hash differs")
                previous = metadata["pts_seconds"]
            values.append(frames)
            paths.extend((cache, video, cache.parent / "manifest.json", cache.parent / "decoder.log"))
        arrays[context] = torch.from_numpy(np.concatenate(values)).permute(0, 3, 1, 2).float() / 255
    history_clips = [c for c in role["clips"] if c["visit"] == "night"]
    if len(history_clips) != 8:
        raise ValueError("Expected eight historical night clips")
    memory_tick = time.perf_counter()
    memory = np.concatenate([np.load(c["frames_path"], allow_pickle=False)[-64:] for c in history_clips])
    set_history(torch.from_numpy(memory).permute(0, 3, 1, 2).float() / 255)
    atomic_json(
        out / "memory-construction.json",
        {
            "seconds": time.perf_counter() - memory_tick,
            "storage_bytes": 512 * 3 * 64 * 64 * 4,
            "clip_ids": [c["clip_id"] for c in history_clips],
            "note": "offline retrospective construction; excludes original online memory maintenance",
        },
    )
    paths.extend(Path(c["frames_path"]) for c in history_clips)
    parents = {}
    for seed in SEEDS:
        directory = PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448"
        checkpoint = directory / "state.pt"
        key = next(
            k for k in verified["artifact_hashes"] if k.replace("\\", "/") == f"runs/w32/full/seed_{seed}/at_8448"
        )
        if digest(checkpoint) != verified["artifact_hashes"][key]["state"]:
            raise ValueError("Parent checkpoint differs from audited state")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["updates"] != 264 or state["pending_indices"] or state["queued_indices"]:
            raise ValueError("Unexpected parent clocks/buffers")
        parents[seed] = state["learner"]
        paths.extend((checkpoint, directory / "predictions.npz", directory / "readout.json"))
    image_path = ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt"
    images = torch.load(image_path, map_location="cpu", weights_only=True)
    paths.append(image_path)
    with np.load(PRIOR / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            images,
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in DRAWS},
            role,
        )
    if len(data.query) != 400 or any(len(v) != 50 for v in data.supports.values()):
        raise ValueError("Shared readout allocation differs")
    config = OmegaConf.create(old_lock["config"])
    paths.extend((ROOT / "cafl4ds").rglob("*.py"))
    hashes = {**old_lock["hashes"], **{str(p): digest(p) for p in paths}}
    lock = {
        "hashes": hashes,
        "config": old_lock["config"],
        "seeds": SEEDS,
        "contexts": CONTEXTS,
        "actions": ACTIONS,
        "checkpoints": CHECKPOINTS,
        "parent_fingerprints": {str(s): fingerprint(p) for s, p in parents.items()},
        "scientific_ssl_updates": 2376,
        "primary_fits": 117,
    }
    lock = json.loads(json.dumps(lock))
    if (out / "input_lock.json").exists() and json.loads((out / "input_lock.json").read_text()) != lock:
        raise ValueError("Resume inputs changed")
    atomic_json(out / "input_lock.json", lock)
    atomic_json(
        out / "data-role-manifest.json",
        {
            "contexts": contexts,
            "parent_source": str(PRIOR),
            "parent_arrivals": 8448,
            "parent_updates": 264,
            "query_positions": data.query.tolist(),
            "supports": {str(k): v.tolist() for k, v in data.supports.items()},
            "development_only": True,
            "fresh_source_confirmation": False,
        },
    )
    for path in paths:
        if path.suffix in (".py", ".md") and path.is_relative_to(ROOT):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return config, arrays, parents, data, lock


def preflight(initial, config, arrays, parent, data, out):
    if (out / "preflight.json").exists():
        return json.loads((out / "preflight.json").read_text())
    work = 0
    before = fingerprint(parent)
    for context in CONTEXTS:
        for action in ACTIONS:
            model, optim = restore(initial, config.optim, parent)
            training = run_branch(model, optim, arrays[context][:128], action, SEEDS[0], (128,))
            work += training["updates"]
            if action == "hold" and fingerprint(capture(model, optim)) != before:
                raise ValueError("Hold mutated parent")
    signatures = []
    for _ in range(2):
        model2, optim2 = restore(initial, config.optim, parent)
        training = run_branch(model2, optim2, arrays["day2"][:128], "smooth", SEEDS[0], (128,))
        work += training["updates"]
        signatures.append(fingerprint(capture(model2, optim2)))
    if signatures[0] != signatures[1] or fingerprint(parent) != before:
        raise ValueError("Parent restoration/alias check failed")
    evaluate_gpu(model, optim, data, out / "preflight")
    result = {
        "passed": True,
        "discarded_ssl_updates": work,
        "discarded_primary_fits": 3,
        "hold_invariant": True,
        "exact_nonempty_adam_continuation": True,
        "readout_isolation": True,
    }
    atomic_json(out / "preflight.json", result)
    return result


def stamp(state):
    state.update(parent_arrivals=8448, parent_updates=264, total_updates=264 + state["updates"])
    return state


def audit(initial, config, arrays, parents, rows, out, progress):
    for number, row in enumerate(rows):
        progress(
            stage="replay_audit",
            audited_branches=number,
            context=row["context"],
            action=row["action"],
            seed=row["seed"],
        )
        model, optim = restore(initial, config.optim, parents[row["seed"]])
        directory = out / row["directory"]

        def observe(clock, state):
            state = stamp(state)
            stored = torch.load(directory / f"at_{clock}/state.pt", map_location="cpu", weights_only=True)
            if fingerprint(state) != fingerprint(stored):
                raise ValueError("Complete branch checkpoint replay differs")
            if row["action"] == "hold" and fingerprint(state["learner"]) != fingerprint(parents[row["seed"]]):
                raise ValueError("Hold state not invariant")
            if row["context"] == "day2" and row["action"] == "full":
                old = torch.load(
                    PRIOR / "runs/w32/full" / f"seed_{row['seed']}" / f"at_{8448 + clock}/state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if fingerprint(old["learner"]) != fingerprint(state["learner"]):
                    raise ValueError("Prior full-day continuation not reproduced")
            return {
                "state_fingerprint": fingerprint(state),
                "updates": state["updates"],
                "total_updates": state["total_updates"],
            }

        repeated = run_branch(model, optim, arrays[row["context"]], row["action"], row["seed"], observe=observe)
        stored = json.loads((directory / "training.json").read_text())
        if stored["observations"] != {str(k): v for k, v in repeated["observations"].items()}:
            raise ValueError("Observation trace differs")
        for a, b in zip(stored["blocks"], repeated["blocks"], strict=True):
            for key in (
                "start",
                "end",
                "selected_indices",
                "exposure_indices",
                "queued_indices",
                "updates",
                "loss_gradients",
            ):
                if json.dumps(a[key]) != json.dumps(b[key]):
                    raise ValueError(f"Branch trace differs: {key}")
    atomic_json(
        out / "training-replay-audit.json",
        {
            "passed": True,
            "branches": len(rows),
            "checkpoints": len(rows) * 2,
            "audit_ssl_updates": sum(r["updates"] for r in rows),
            "all_hold_states_invariant": True,
            "full_day_continuations_exact": True,
        },
    )


def interval_counts(values, denominator):
    pp = [100 * v / denominator for v in values]
    mean = statistics.fmean(pp)
    half = float(t.ppf(0.975, 2)) * statistics.stdev(pp) / 3**0.5
    return {
        "mean_pp": mean,
        "ci95_pp": [mean - half, mean + half],
        "seed_values_pp": pp,
        "wins": sum(v > 0 for v in values),
        "ties": sum(v == 0 for v in values),
    }


def report(data, rows, out):
    counts = {}
    directories = (
        [(f"parents/seed_{s}", out / "parents" / f"seed_{s}") for s in SEEDS]
        + [
            (r["directory"] + f"/at_{c}", out / r["directory"] / f"at_{c}")
            for r in rows
            if r["action"] != "hold"
            for c in CHECKPOINTS
        ]
        + [("warm_frozen", PRIOR / "frozen")]
    )
    for name, directory in directories:
        with np.load(directory / "predictions.npz", allow_pickle=False) as pred:
            for draw in DRAWS:
                for reader in READERS:
                    correct = pred[f"{draw}_{reader}"] == data.labels[data.query].numpy()
                    domain = data.domains[data.query].numpy()
                    counts[name, draw, reader] = {
                        "balanced": int(correct.sum()),
                        "A": int(correct[domain == 0].sum()),
                        "B": int(correct[domain == 1].sum()),
                    }
    by = {(r["context"], r["action"], r["seed"]): r for r in rows}

    def total(context, action, seed, clock, metric, reader="finetune"):
        name = (
            "warm_frozen"
            if action == "warm_frozen"
            else f"parents/seed_{seed}"
            if action == "hold"
            else by[context, action, seed]["directory"] + f"/at_{clock}"
        )
        return sum(counts[name, d, reader][metric] for d in DRAWS)

    contexts = {}
    for context in CONTEXTS:
        means = {
            a: {
                str(c): {
                    m: 100 * sum(total(context, a, s, c, m) for s in SEEDS) / ((600 if m != "balanced" else 1200) * 3)
                    for m in METRICS
                }
                for c in CHECKPOINTS
            }
            for a in ACTIONS
        }
        contrasts = {}
        for first, second in ((a, b) for a in ACTIONS for b in ACTIONS if a != b):
            contrasts[first + "_minus_" + second] = {
                str(c): {
                    m: interval_counts(
                        [total(context, first, s, c, m) - total(context, second, s, c, m) for s in SEEDS],
                        600 if m != "balanced" else 1200,
                    )
                    for m in METRICS
                }
                for c in CHECKPOINTS
            }
        preferred = {}
        for action in ACTIONS:
            others = [a for a in ACTIONS if a != action]
            preferred[action] = all(
                contrasts[action + "_minus_" + other]["4224"]["balanced"]["ci95_pp"][0] > 0
                and contrasts[action + "_minus_" + other]["4224"]["balanced"]["mean_pp"] >= 0.1
                and all(contrasts[action + "_minus_" + other]["4224"][m]["ci95_pp"][0] > -1 for m in ("A", "B"))
                for other in others
            )
        smooth_useful = interval_counts(
            [
                total(context, "smooth", s, 4224, "balanced") - total(context, "warm_frozen", s, 4224, "balanced")
                for s in SEEDS
            ],
            1200,
        )
        paired_savings = [
            1 - by[context, "smooth", s]["online_seconds"] / by[context, "full", s]["online_seconds"] for s in SEEDS
        ]
        resource = False  # Full matched work and additional memory; no efficiency promotion.
        contexts[context] = {
            "means_accuracy_percent": means,
            "contrasts": contrasts,
            "quality_preferred": preferred,
            "smooth_useful_vs_warm_frozen": smooth_useful,
            "smooth_resource_lead_descriptive_only": resource,
            "mean_paired_time_saving": statistics.fmean(paired_savings),
            "paired_time_savings": paired_savings,
            "secondary_accuracy_percent": {
                reader: {
                    a: {
                        str(c): {
                            m: 100
                            * sum(total(context, a, s, c, m, reader) for s in SEEDS)
                            / ((600 if m != "balanced" else 1200) * 3)
                            for m in METRICS
                        }
                        for c in CHECKPOINTS
                    }
                    for a in ACTIONS
                }
                for reader in ("linear", "knn")
            },
        }
    interaction = {
        action: {
            str(c): {
                m: interval_counts(
                    [
                        (total("night2", action, s, c, m) - total("night2", "hold", s, c, m))
                        - (total("day2", action, s, c, m) - total("day2", "hold", s, c, m))
                        for s in SEEDS
                    ],
                    600 if m != "balanced" else 1200,
                )
                for m in METRICS
            }
            for c in CHECKPOINTS
        }
        for action in ("full", "smooth", "abrupt")
    }
    winners = {c: [a for a, v in contexts[c]["quality_preferred"].items() if v] for c in CONTEXTS}
    decision = (
        "different_context_quality_preferences"
        if winners["day2"] and winners["night2"] and winners["day2"] != winners["night2"]
        else "shared_quality_preference"
        if winners["day2"] and winners["day2"] == winners["night2"]
        else "local_action_preference_only"
        if any(winners.values())
        else "no_quality_action_preference_qualified"
    )
    result = {
        "decision": decision,
        "contexts": contexts,
        "context_action_interaction_night_minus_day": interaction,
        "scientific_ssl_updates": 2376,
        "primary_fits": 117,
        "new_prediction_sets": 351,
        "shared_parent_hold_readouts": 3,
        "memory_storage_bytes": 512 * 3 * 64 * 64 * 4,
        "development_only": True,
        "previous_day2_full_outcome_known": True,
        "historic_exposures_per_mix_branch": 272,
        "new_exposures_per_mix_branch": 3952,
        "memory_images": 512,
        "efficiency_promotion": False,
        "uncertainty": "paired t df2 over three reused-history optimization seeds; no independent-source or multiplicity confirmation",
    }
    atomic_json(out / "summary.json", result)
    lines = [
        "# Matched-budget training-diet smoothing",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Context | Action | Early day % | Early night % | Final day % | Final night % | Final balanced % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for context, row in contexts.items():
        for action in ACTIONS:
            early = row["means_accuracy_percent"][action]["256"]
            final = row["means_accuracy_percent"][action]["4224"]
            lines.append(
                f"| {context} | {action} | {early['A']:.3f} | {early['B']:.3f} | {final['A']:.3f} | {final['B']:.3f} | {final['balanced']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"{context} quality preferences: {row['quality_preferred']}; efficiency promotion: {row['smooth_resource_lead_descriptive_only']}; measured smooth/full time difference: {100 * row['mean_paired_time_saving']:.1f}%.",
            ]
        )
        for name in ("smooth_minus_hold", "smooth_minus_full", "smooth_minus_abrupt", "abrupt_minus_hold"):
            contrast = row["contrasts"][name]["4224"]["balanced"]
            lines.append(
                f"{name}: {contrast['mean_pp']:+.3f}pp [{contrast['ci95_pp'][0]:+.3f}, {contrast['ci95_pp'][1]:+.3f}]."
            )
    lines.extend(
        [
            "",
            "Three-seed conditional development diagnostic from the same audited full-SSL night-end parents. Returning-day/full outcomes were previously observed; continued-night clips add a control but do not isolate illumination from clip identity. No labels choose actions. No health model or deployed policy is trained.",
            "",
            "All2376 scientific updates/117 primary fits and exact branch replay completed. Hold shares one verified parent readout per seed. Independent readout/source checking is recorded separately.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--night-frames", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "completed_branches": 0, "total_branches": 24}

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
        progress(stage="locking_inputs")
        config, arrays, parents, data, lock = load_inputs(out, args.night_frames.resolve())
        torch.manual_seed(11)
        initial = _method(config).cuda()
        progress(stage="discarded_preflight")
        preflight(initial, config, arrays, parents[SEEDS[0]], data, out)
        for seed in SEEDS:
            directory = out / "parents" / f"seed_{seed}"
            if not (directory / "readout.json").exists():
                model, optim = restore(initial, config.optim, parents[seed])
                evaluate_gpu(model, optim, data, directory)
            with (
                np.load(directory / "predictions.npz") as a,
                np.load(PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448/predictions.npz") as b,
            ):
                if a.files != b.files or not all(np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError("Parent readout not reproduced")
        rows = []
        arms = [(c, a) for c in CONTEXTS for a in ACTIONS]
        for position, seed in enumerate(SEEDS):
            for context, action in arms[position:] + arms[:position]:
                directory = out / "runs" / context / action / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                if (directory / "result.json").exists():
                    rows.append(json.loads((directory / "result.json").read_text()))
                    continue
                model, optim = restore(initial, config.optim, parents[seed])
                progress(stage="branch", context=context, action=action, seed=seed)

                def observe(clock, state):
                    state = stamp(state)
                    checkpoint = directory / f"at_{clock}"
                    checkpoint.mkdir(exist_ok=True)
                    signature = fingerprint(state)
                    if (checkpoint / "state.pt").exists():
                        if (
                            fingerprint(torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True))
                            != signature
                        ):
                            raise ValueError("Partial branch state changed")
                    else:
                        torch.save(state, checkpoint / "state.pt")
                    if action != "hold" and not (checkpoint / "readout.json").exists():
                        evaluate_gpu(model, optim, data, checkpoint)
                    if action == "hold":
                        atomic_json(
                            checkpoint / "readout-alias.json",
                            {
                                "directory": f"parents/seed_{seed}",
                                "reason": "same unchanged learner; shared evaluation, no duplicate fits",
                            },
                        )
                    return {
                        "state_fingerprint": signature,
                        "updates": state["updates"],
                        "total_updates": state["total_updates"],
                    }

                training = run_branch(model, optim, arrays[context], action, seed, observe=observe)
                expected = {"hold": 0, "smooth": 132, "abrupt": 132, "full": 132}[action]
                if (
                    training["updates"] != expected
                    or fingerprint(parents[seed]) != lock["parent_fingerprints"][str(seed)]
                ):
                    raise ValueError("Branch budget or parent mutated")
                atomic_json(directory / "training.json", training)
                row = {
                    "context": context,
                    "action": action,
                    "seed": seed,
                    "directory": str(directory.relative_to(out)),
                    "updates": expected,
                    "accepted_frames": training["accepted_frames"],
                    "online_seconds": training["online_seconds"],
                }
                atomic_json(directory / "result.json", row)
                rows.append(row)
                progress(completed_branches=len(rows))
        if any(digest(Path(p)) != v for p, v in lock["hashes"].items()):
            raise ValueError("Locked inputs changed")
        atomic_json(
            out / "collection.json",
            {"records": rows, "scientific_ssl_updates": sum(r["updates"] for r in rows), "primary_fits": 117},
        )
        audit(initial, config, arrays, parents, rows, out, progress)
        result = report(data, rows, out)
        if len(rows) != 24 or sum(r["updates"] for r in rows) != 2376:
            raise ValueError("Final budget mismatch")
        progress(
            stage="complete",
            completed_branches=24,
            audited_branches=24,
            scientific_ssl_updates=2376,
            primary_fits=117,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
