"""Provisional local-annotation temporal state experiment with paired causal branches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.temporal_action_response import restore, run_action, snapshot
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_recurrence_retention import evaluate
from scripts.temporal_state_features import causal_features

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
OFFSETS = (26, 74)
HORIZON = 2
SCENES = ("city street", "highway", "residential")


def sha(path: Path) -> str:
    """Hash file contents before use."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_state(path: Path, value: dict) -> None:
    """Atomically replace the resumable checkpoint."""
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def order_cohort(clips: list[dict], cohort: int) -> list[dict]:
    """Rotate balanced cells; both decision horizons stay inside clips 7 and 19."""
    cells = [(day, scene) for day in ("daytime", "night") for scene in SCENES]
    pools = {}
    for day, scene in cells:
        selected = [
            c
            for c in clips
            if c["keyframe_attributes"]["timeofday"] == day and c["keyframe_attributes"]["scene"] == scene
        ]
        if len(selected) != 4:
            raise ValueError("Cohort must contain exactly four clips in every cell")
        pools[(day, scene)] = sorted(
            selected, key=lambda c: hashlib.sha256(f"order:{c['clip_id']}".encode()).hexdigest()
        )
    ordered = []
    for repeat in range(4):
        shift = (cohort + (3 if repeat == 3 else 0)) % 6
        for cell in cells[shift:] + cells[:shift]:
            ordered.append(pools[cell][repeat])
    return ordered


def make_evaluation(rows: list[dict], out: Path) -> Evaluation:
    """Use only fixed support roles and development query; final queries remain unscored."""
    selected, supports = [], {}
    for panel, draw in enumerate(DRAWS):
        positions = []
        for label, scene in enumerate(SCENES):
            daytime_count = 12 + (panel + label) % 2
            for day, count in (("daytime", daytime_count), ("night", 25 - daytime_count)):
                candidates = [
                    r
                    for r in rows
                    if r["role"] == f"support_panel_{panel}"
                    and r["keyframe_attributes"]["scene"] == scene
                    and r["keyframe_attributes"]["timeofday"] == day
                ]
                candidates.sort(key=lambda r: hashlib.sha256(f"support:{draw}:{r['clip_id']}".encode()).hexdigest())
                if len(candidates) < count:
                    raise ValueError("Insufficient fixed support cell")
                positions.extend(range(len(selected), len(selected) + count))
                selected.extend(candidates[:count])
        supports[draw] = torch.tensor(positions)
    query_rows = sorted((r for r in rows if r["role"] == "development_query"), key=lambda r: r["clip_id"])
    query = torch.arange(len(selected), len(selected) + len(query_rows))
    selected.extend(query_rows)
    if len(query) != 768 or len({r["clip_id"] for r in selected}) != 993:
        raise ValueError("Support/query allocation differs from protocol")
    images, labels, domains = [], [], []
    for row in selected:
        image, label_path = Path(row["image_path"]), Path(row["label_path"])
        if sha(image) != row["image_sha256"] or sha(label_path) != row["label_sha256"]:
            raise ValueError("Reader image or label changed")
        attrs = json.loads(label_path.read_text())["attributes"]
        if attrs != row["keyframe_attributes"] or attrs["weather"] != "clear":
            raise ValueError("Reader metadata differs")
        with Image.open(image) as opened:
            images.append(np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        labels.append(SCENES.index(attrs["scene"]))
        domains.append(0 if attrs["timeofday"] == "daytime" else 1)
    data = Evaluation(
        torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255,
        torch.tensor(labels),
        torch.tensor(domains),
        query,
        supports,
        {"selected": selected, "label_provenance_verified": False, "query_role": "development_query"},
    )
    if any(
        int(((data.labels[query] == c) & (data.domains[query] == d)).sum()) != 128 for c in range(3) for d in range(2)
    ):
        raise ValueError("Queries are not six-cell balanced")
    atomic_json(out / "reader-allocation.json", data.manifest)
    torch.save(data.images, out / "evaluation_images.pt")
    np.savez_compressed(
        out / "evaluation_arrays.npz",
        labels=data.labels.numpy(),
        domains=data.domains.numpy(),
        query=query.numpy(),
        **{f"support_{d}": p.numpy() for d, p in supports.items()},
    )
    return data


def report(records: list[dict], out: Path) -> None:
    """Aggregate seeds inside cohort states, with no population qualification claims."""
    groups = {}
    for row in records:
        groups.setdefault((row["cohort"], row["history"], row["offset"]), []).append(row)
    points = []
    for (cohort, history, offset), values in sorted(groups.items()):
        point = {"cohort": cohort, "history": history, "offset": offset, "features": values[0]["features"]}
        for reader in ("finetune", "linear", "knn"):
            point[reader] = {
                metric: 100
                * statistics.fmean(
                    r["full"]["scores"][reader][metric] - r["half"]["scores"][reader][metric] for r in values
                )
                for metric in ("balanced", "A", "B")
            }
            for action in ("full", "half"):
                point[f"{action}_{reader}_gain_over_hold_pp"] = {
                    metric: 100
                    * statistics.fmean(
                        r[action]["scores"][reader][metric] - r["hold"]["scores"][reader][metric] for r in values
                    )
                    for metric in ("balanced", "A", "B")
                }
        point["processing_saving_fraction"] = statistics.fmean(r["processing_saving_fraction"] for r in values)
        points.append(point)
    x = np.array(
        [
            [
                r["features"]["mean_adjacent_rgb_change"],
                r["features"]["within_window_rgb_spread"],
                int(r["history"] == "full"),
                int(r["offset"] == 74),
            ]
            for r in points
        ]
    )
    y = np.array([r["finetune"]["balanced"] for r in points])
    cv = []
    for cohort in range(6):
        train = np.array([r["cohort"] != cohort for r in points])
        test = ~train
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        std[std == 0] = 1
        design = np.column_stack((np.ones(train.sum()), (x[train] - mean) / std))
        weights = np.linalg.lstsq(design, y[train], rcond=None)[0]
        prediction = np.column_stack((np.ones(test.sum()), (x[test] - mean) / std)) @ weights
        cv.append(
            {
                "held_out_cohort": cohort,
                "observed_pp": y[test].tolist(),
                "predicted_pp": prediction.tolist(),
                "model_mse_pp2": float(((prediction - y[test]) ** 2).mean()),
                "intercept_mse_pp2": float(((y[train].mean() - y[test]) ** 2).mean()),
            }
        )
    atomic_json(
        out / "summary.json",
        {
            "mode": "provisional_local_annotation_development",
            "label_provenance_verified": False,
            "source_session_independence_established": False,
            "sealed_confirmation": False,
            "paired_parents": len(records),
            "trained_branches": 2 * len(records),
            "cohort_state_means": points,
            "leave_one_cohort_out": cv,
            "qualified_half_preferred_states": None,
            "caveat": (
                "Six designed clip cohorts with shared queries; descriptive discovery, "
                "not independent population confirmation"
            ),
        },
    )
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, key, title in zip(
        axes,
        ("mean_adjacent_rgb_change", "within_window_rgb_spread"),
        ("Past temporal change", "Past visual spread"),
        strict=True,
    ):
        for cohort in range(6):
            subset = [r for r in points if r["cohort"] == cohort]
            axis.scatter(
                [r["features"][key] for r in subset],
                [r["finetune"]["balanced"] for r in subset],
                label=f"Cohort {cohort + 1}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        axis.set(xlabel=title, ylabel="Full minus half fine-tuning accuracy (pp)")
    axes[1].legend(fontsize=7)
    figure.suptitle("Provisional state discovery; seeds averaged within cohort states")
    figure.savefig(out / "state-opportunity.png", dpi=180)


def main() -> None:  # noqa: C901 - explicit resumable histories, isolated branches and readout guards
    """Run the fixed development design, recording uncertainty rather than certifying labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "pid": os.getpid(),
        "started_unix": started,
        "mode": "provisional_local_annotation_development",
        "total_paired_parents": 72,
        "total_trained_branches": 144,
        "completed_paired_parents": 0,
    }

    def progress(**fields) -> None:
        status.update(fields, updated_unix=time.time())
        atomic_json(out / "status.json", status)

    try:
        progress(stage="locking_inputs")
        prepared = json.loads(args.prepared.read_text())
        if prepared["mode"] != status["mode"] or prepared["sealed_final_predictions_allowed"]:
            raise ValueError("Incorrect development scope")
        preflight_path = args.prepared.resolve().parent / "readout-preflight/preflight.json"
        preflight = json.loads(preflight_path.read_text())
        if not preflight["passed"] or not preflight["ssl_state_unchanged"]:
            raise ValueError("Reader compatibility preflight has not passed")
        sources = [
            args.prepared.resolve(),
            args.protocol.resolve(),
            Path(__file__),
            ROOT / "scripts/temporal_state_features.py",
            ROOT / "cafl4ds/temporal_action_response.py",
            ROOT / "cafl4ds/temporal_anchored_filter.py",
            ROOT / "cafl4ds/update_filtering.py",
            ROOT / "scripts/run_recurrence_retention.py",
            ROOT / "scripts/evaluate_checkpoint_robustness.py",
            ROOT / "scripts/run_adaptation_experiment.py",
            ROOT / "scripts/run_paired_update_response.py",
            ROOT / "cafl4ds/recurrence.py",
            ROOT / "scripts/prepare_temporal_state_development.py",
            preflight_path,
        ]
        old_path = ROOT / "outputs/paired-update-response/20261010-states/input_lock.json"
        old = json.loads(old_path.read_text())
        config = OmegaConf.create(old["config"])
        warm = Path(config.well).resolve()
        if sha(warm) != old["hashes"][str(warm)]:
            raise ValueError("Warm checkpoint changed")
        sources.extend((warm, old_path))
        for row in prepared["allocation"]:
            if (
                sha(Path(row["image_path"])) != row["image_sha256"]
                or sha(Path(row["label_path"])) != row["label_sha256"]
            ):
                raise ValueError("Allocation changed before collection")
            sources.extend((Path(row["image_path"]), Path(row["label_path"])))
        for clip in prepared["clips"]:
            cache = Path(clip["frames_path"])
            if sha(cache) != clip["frames_sha256"]:
                raise ValueError("Prepared cache changed")
            sources.append(cache)
        hashes = {str(p): sha(p) for p in sources}
        lock_path = out / "input_lock.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text())["hashes"] != hashes:
                raise ValueError("Resume sources differ; original run preserved")
        else:
            atomic_json(
                lock_path,
                {
                    "hashes": hashes,
                    "config": OmegaConf.to_container(config, resolve=True),
                    "seeds": SEEDS,
                    "offsets": OFFSETS,
                    "horizon": HORIZON,
                    "label_provenance_verified": False,
                },
            )
            for path in sources:
                if path.suffix in {".py", ".md"} and path.is_relative_to(ROOT):
                    target = out / "source_snapshot" / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        data = make_evaluation(prepared["allocation"], out)
        torch.manual_seed(11)
        initial = _method(config)

        def readout(
            model: SSLMethod,
            optimizer: torch.optim.Optimizer,
            state: dict,
            directory: Path,
        ) -> dict:
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.json"
            if result_path.exists():
                return json.loads(result_path.read_text())
            marker = fingerprint(state)
            tick = time.perf_counter()
            outcome, _ = evaluate(model, data, None, 0, True, 100, directory)
            if fingerprint(snapshot(model, optimizer, state["optimizer_updates"], state["arrivals"])) != marker:
                raise ValueError("Readout changed the SSL state")
            outcome["readout_seconds"] = time.perf_counter() - tick
            atomic_json(result_path, outcome)
            return outcome

        progress(stage="frozen_readout")
        frozen_optimizer = instantiate(config.optim, params=[p for p in initial.parameters() if p.requires_grad])
        frozen_state = snapshot(initial, frozen_optimizer, 0, 0)
        readout(initial, frozen_optimizer, frozen_state, out / "qualification/frozen")
        cohorts = []
        for cohort in range(6):
            members = [r for r in prepared["clips"] if r["role"] == f"stream_cohort_{cohort}"]
            cohorts.append(order_cohort(members, cohort))
        atomic_json(
            out / "stream-order.json",
            {
                "cohorts": [[c["clip_id"] for c in group] for group in cohorts],
                "artificial_clip_concatenation": True,
                "natural_recurrence_claimed": False,
            },
        )
        # Two one-per-cell bundles, selected by the fixed first round in cohorts 0 and 1.
        for cohort in (0, 1):
            frames = np.concatenate([np.load(c["frames_path"], allow_pickle=False) for c in cohorts[cohort][:6]])
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
            for action in ("full", "half"):
                directory = out / "qualification" / f"bundle_{cohort}" / action
                if (directory / "result.json").exists():
                    continue
                progress(stage="small_qualification", cohort=cohort, action=action)
                torch.manual_seed(5003)
                model = copy.deepcopy(initial)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                windows = run_action(model, optimizer, images, 0, action, horizon=24)
                state = snapshot(model, optimizer, sum(w["optimizer_updates"] for w in windows), 1536)
                directory.mkdir(parents=True, exist_ok=True)
                save_state(directory / "terminal.pt", state)
                atomic_json(directory / "training.json", {"windows": windows})
                readout(model, optimizer, state, directory)
        records = []
        discarded_controls = 0
        for cohort, ordered in enumerate(cohorts):
            frames = np.concatenate([np.load(c["frames_path"], allow_pickle=False) for c in ordered])
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
            for seed in SEEDS:
                for history in ("full", "half"):
                    directory = out / "histories" / f"cohort_{cohort}" / f"seed_{seed}" / history
                    directory.mkdir(parents=True, exist_ok=True)
                    latest = directory / "latest.pt"
                    if latest.exists():
                        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
                        state, windows = checkpoint["state"], checkpoint["windows"]
                        model, optimizer = restore(initial, config, state)
                        start = state["arrivals"] // 64
                        updates = state["optimizer_updates"]
                    else:
                        torch.manual_seed(seed)
                        model = copy.deepcopy(initial)
                        optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                        start, updates, windows = 0, 0, []
                    for window in range(start, max(OFFSETS)):
                        progress(stage="parent_history", cohort=cohort, seed=seed, history=history, window=window)
                        row = run_action(model, optimizer, images, window, history, horizon=1)[0]
                        windows.append(row)
                        updates += row["optimizer_updates"]
                        state = snapshot(model, optimizer, updates, (window + 1) * 64)
                        if window + 1 in OFFSETS:
                            save_state(directory / f"parent_{window + 1}.pt", state)
                            atomic_json(directory / f"features_{window + 1}.json", causal_features(frames, window + 1))
                        save_state(latest, {"state": state, "windows": windows})
                    for offset_index, offset in enumerate(OFFSETS):
                        branch_root = directory / f"decision_{offset}"
                        paired_path = branch_root / "paired.json"
                        if paired_path.exists():
                            records.append(json.loads(paired_path.read_text()))
                            progress(completed_paired_parents=len(records))
                            continue
                        parent = torch.load(directory / f"parent_{offset}.pt", map_location="cpu", weights_only=True)
                        marker = fingerprint(parent)
                        parent_updates = parent["optimizer_updates"]
                        features = json.loads((directory / f"features_{offset}.json").read_text())
                        current, current_optimizer = restore(initial, config, parent)
                        progress(stage="parent_readout", cohort=cohort, seed=seed, history=history, offset=offset)
                        parent_outcome = readout(current, current_optimizer, parent, branch_root / "hold")
                        branches = {}
                        actions = (
                            ("full", "half")
                            if (cohort + SEEDS.index(seed) + offset_index + int(history == "half")) % 2 == 0
                            else ("half", "full")
                        )
                        for action in actions:
                            progress(
                                stage="paired_branch",
                                action=action,
                                cohort=cohort,
                                seed=seed,
                                history=history,
                                offset=offset,
                            )
                            model, optimizer = restore(initial, config, parent)
                            if fingerprint(snapshot(model, optimizer, parent_updates, offset * 64)) != marker:
                                raise ValueError("Restored branch differs from parent")
                            rows = run_action(model, optimizer, images, offset, action, HORIZON)
                            update_count = sum(w["optimizer_updates"] for w in rows)
                            terminal = snapshot(
                                model, optimizer, parent_updates + update_count, (offset + HORIZON) * 64
                            )
                            terminal_marker = fingerprint(terminal)
                            branch_dir = branch_root / action
                            branch_dir.mkdir(parents=True, exist_ok=True)
                            save_state(branch_dir / "terminal.pt", terminal)
                            if action == "full":
                                repeated, repeated_optimizer = restore(initial, config, parent)
                                repeat_rows = run_action(repeated, repeated_optimizer, images, offset, action, HORIZON)
                                repeated_terminal = snapshot(
                                    repeated, repeated_optimizer, parent_updates + update_count, (offset + HORIZON) * 64
                                )
                                if fingerprint(repeated_terminal) != terminal_marker or any(
                                    a["loss_gradients"] != b["loss_gradients"]
                                    for a, b in zip(rows, repeat_rows, strict=True)
                                ):
                                    raise ValueError("Independent full continuation differs")
                                discarded_controls += update_count
                                # Restore branch RNG after the independent control before isolated readout.
                                model, optimizer = restore(initial, config, terminal)
                            outcome = readout(model, optimizer, terminal, branch_dir)
                            branches[action] = {
                                **outcome,
                                "windows": rows,
                                "terminal_fingerprint": terminal_marker,
                                "online_seconds": sum(w["online_seconds"] for w in rows),
                                "optimizer_updates": update_count,
                            }
                            if fingerprint(parent) != marker:
                                raise ValueError("Parent state was mutated")
                        record = {
                            "cohort": cohort,
                            "seed": seed,
                            "history": history,
                            "offset": offset,
                            "features": features,
                            "hold": parent_outcome,
                            **branches,
                            "source_clip_id": ordered[offset // 4]["clip_id"],
                            "processing_saving_fraction": 1
                            - branches["half"]["online_seconds"] / branches["full"]["online_seconds"],
                            "parent_fingerprint": marker,
                            "exact_continuation": True,
                        }
                        atomic_json(paired_path, record)
                        records.append(record)
                        progress(completed_paired_parents=len(records), completed_trained_branches=2 * len(records))
        if any(sha(Path(p)) != expected for p, expected in hashes.items()):
            raise ValueError("Locked input changed during collection")
        progress(stage="reporting")
        report(records, out)
        progress(
            stage="complete",
            completed_paired_parents=len(records),
            completed_trained_branches=2 * len(records),
            main_scientific_ssl_updates=4428,
            qualification_ssl_updates=144,
            discarded_restore_updates=discarded_controls,
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
