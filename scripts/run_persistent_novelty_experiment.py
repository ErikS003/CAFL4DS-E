"""Five-arm fixed-budget persistent-novelty SSL study on the positive temporal setting."""

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
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import _driving_attributes, _image_filename, _iter_bdd_records
from cafl4ds.persistent_novelty import ANCHORS, POLICIES, persistent_indices, run_stream
from cafl4ds.recurrence import Evaluation, prediction_scores
from cafl4ds.short_buffer import capture, restore
from scripts.confirm_replay_advantage import paired_interval
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest, evaluate_gpu, remap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/temporal-bdd/20261011-feasibility-v2"
CACHE_PRIOR = ROOT / "outputs/content-filter-bdd/20261011-coverage-half"
INTAKE = ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-persistent-novelty-frozen.md"
SEEDS = (5003, 5009, 5011, 5021, 5023)
DRAWS = (51011, 51017, 51023)
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B", "micro")
DENOMINATORS = {"balanced": 600, "A": 300, "B": 300, "micro": 600}


def exact_correct_counts(predictions, labels, domains):
    """Keep integer totals so equal accuracies cannot become floating-point wins."""
    for domain in (0, 1):
        for label in range(3):
            if int(((domains == domain) & (labels == label)).sum()) != 100:
                raise ValueError("Exact statistics require the frozen balanced query cells")
    correct = predictions == labels
    return {
        "balanced": int(correct.sum()),
        "micro": int(correct.sum()),
        "A": int(correct[domains == 0].sum()),
        "B": int(correct[domains == 1].sum()),
    }


def exact_contrast(by, counts, first, second, reader, metric, draws):
    values = []
    for seed in SEEDS:
        first_path = by[first, seed]["directory"]
        second_path = "frozen" if second == "frozen" else by[second, seed]["directory"]
        difference = sum(counts[first_path, d, reader][metric] - counts[second_path, d, reader][metric] for d in draws)
        values.append(difference / (DENOMINATORS[metric] * len(draws)))
    result = paired_interval(values)
    result["ties"] = sum(v == 0 for v in values)
    return result


def normalized(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def independent_selection(features, previous):
    """Independent NumPy persistence calculation from the frozen warm features."""
    z = normalized(features.astype(np.float64))
    reference = z[0] if previous is None else normalized(previous.mean(0))
    residual = z - reference
    magnitude = np.linalg.norm(residual, axis=1)
    direction = normalized(residual)
    scores = np.zeros(64)
    for index in range(62):
        group = direction[index : index + 3]
        coherence = max(0.0, min(float(group[0] @ group[1]), float(group[0] @ group[2]), float(group[1] @ group[2])))
        scores[index] = min(magnitude[index : index + 3]) * coherence
    anchors = list(range(3, 64, 4))
    candidates = [i for i in range(62) if i not in anchors]
    content = sorted(candidates, key=lambda i: (-scores[i], i))[:16]
    selected = np.array(sorted(anchors + content))
    return selected, scores, reference, z[selected][-16:]


def load_inputs(out, progress):
    audit_path = PRIOR / "independent_verification.json"
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["arms_checked"] != 20:
        raise ValueError("Earlier positive temporal collection is not audited")
    paths = [audit_path, PRIOR / "report.md", PRIOR / "report.json", PROTOCOL, INTAKE]
    for name, value in audit["artifact_hashes"].items():
        path = PRIOR / name
        if digest(path) != value:
            raise ValueError(f"Earlier audited artifact differs: {name}")
        paths.append(path)
    old_path = PRIOR / "input_lock.json"
    old = json.loads(old_path.read_text())
    config = OmegaConf.create(old["config"])
    original_warm = str(config.well)
    warm = remap(original_warm)
    if digest(warm) != old["hashes"][original_warm]:
        raise ValueError("Warm checkpoint differs")
    config.well, config.device = str(warm), "cuda"
    paths += [old_path, warm]
    selection_path = ROOT / "outputs/temporal-bdd/20261011-intake/selection-recheck.json"
    selection = json.loads(selection_path.read_text())
    intake = json.loads(INTAKE.read_text())
    if not intake["complete"] or intake["infrastructure_smoke"] or len(intake["clips"]) != 48:
        raise ValueError("Incomplete original intake")
    if selection["warm_overlap_count"] or selection["validation_overlap_count"]:
        raise ValueError("Clip-role overlap")
    if [c["clip_id"] for c in selection["clips"]] != [c["clip_id"] for c in intake["clips"]]:
        raise ValueError("Original chronological allocation differs")
    paths.append(selection_path)
    arrays, frame_hashes = [], []
    for index, clip in enumerate(intake["clips"]):
        cache = remap(clip["frames_path"])
        if digest(cache) != clip["frames_sha256"]:
            raise ValueError("Original frame cache differs")
        frames = np.load(cache, allow_pickle=False)
        if frames.shape != (128, 64, 64, 3) or frames.dtype != np.uint8:
            raise ValueError("Invalid clip cache")
        times = [r["pts_seconds"] for r in clip["frames"]]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Clip timestamps do not advance")
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        if hashes != [r["rgb64_sha256"] for r in clip["frames"]]:
            raise ValueError("Individual frame hash differs")
        video = remap(clip["video_path"])
        if digest(video) != clip["video_sha256"]:
            raise ValueError("Original source video differs")
        clip["local_frames_path"] = str(cache)
        arrays.append(frames)
        frame_hashes.extend(hashes)
        paths.extend((cache, video))
        if index % 12 == 0:
            progress(stage="checking_original_clips", checked_clips=index + 1)
    arrivals = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
    del arrays
    prior_data_path = PRIOR / "data_lock.json"
    prior_data = json.loads(prior_data_path.read_text())
    if prior_data["frame_hashes"] != frame_hashes:
        raise ValueError("Decoded frame population/order differs from positive run")
    cached_data_path = CACHE_PRIOR / "data_lock.json"
    cached = json.loads(cached_data_path.read_text())
    if cached["evaluation"] != prior_data["evaluation"]:
        raise ValueError("Cached reader membership differs")
    manifest = prior_data["evaluation"]
    pixel_path = CACHE_PRIOR / "evaluation_images.pt"
    array_path = CACHE_PRIOR / "evaluation_arrays.npz"
    cache_lock_path = CACHE_PRIOR / "input_lock.json"
    cache_lock = json.loads(cache_lock_path.read_text())
    for path in (pixel_path, array_path):
        original = next(
            (
                v
                for p, v in cache_lock["hashes"].items()
                if p.replace("\\", "/").endswith("/" + str(path.relative_to(ROOT)).replace("\\", "/"))
            ),
            None,
        )
        if original is None or digest(path) != original:
            raise ValueError("Audited validation cache hash differs")
    images = torch.load(pixel_path, map_location="cpu", weights_only=True)
    with np.load(array_path, allow_pickle=False) as arrays:
        labels = torch.from_numpy(arrays["labels"].copy())
        domains = torch.from_numpy(arrays["domains"].copy())
        query = torch.from_numpy(arrays["query"].copy())
    if images.shape != (900, 3, 64, 64) or query.tolist() != manifest["query_positions"]:
        raise ValueError("Validation shapes/queries differ")
    supports = {d: torch.tensor(manifest["supports"][str(d)]) for d in DRAWS}
    # Reconstruct record ordering from the transferred per-image validation labels.
    labels_root = ROOT.parents[1] / "datasets/bdd100k/labels/val"
    image_root = ROOT.parents[1] / "datasets/bdd100k/images/100k/val"
    scene_names = ("city street", "highway", "residential")
    records = []
    progress(stage="reconstructing_original_validation")
    for row in _iter_bdd_records(labels_root):
        attrs = _driving_attributes(row)
        if attrs is not None and attrs[2] in scene_names and (image_root / _image_filename(row)).exists():
            records.append(row)
    if len(records) != 8753:
        raise ValueError(f"Original three-scene validation population differs: {len(records)}")
    reader_rows = []
    stream_ids = {c["clip_id"] for c in intake["clips"]}
    selected_frame_hashes = set(frame_hashes)
    for position, source_index in enumerate(manifest["original_indices"]):
        row = records[source_index]
        attrs = _driving_attributes(row)
        path = image_root / _image_filename(row)
        label_path = labels_root / (path.stem + ".json")
        if path.stem in stream_ids or attrs[1] != "clear":
            raise ValueError("Validation role/attribute mismatch")
        if int(labels[position]) != scene_names.index(attrs[2]) or int(domains[position]) != (attrs[0] == "night"):
            raise ValueError("Original validation labels differ")
        with Image.open(path) as opened:
            pixels = np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
        expected = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        if (
            not torch.equal(expected, images[position])
            or hashlib.sha256(pixels.tobytes()).hexdigest() in selected_frame_hashes
        ):
            raise ValueError("Validation pixels differ or overlap stream")
        reader_rows.append(
            {
                "clip_id": path.stem,
                "image_path": str(path),
                "label_path": str(label_path),
                "attributes": row["attributes"],
                "source_index": source_index,
            }
        )
        paths.extend((path, label_path))
    for domain in (0, 1):
        for label in range(3):
            if int(((domains[query] == domain) & (labels[query] == label)).sum()) != 100:
                raise ValueError("Queries do not match six balanced100-image cells")
    paths += [prior_data_path, cached_data_path, pixel_path, array_path, cache_lock_path, Path(__file__)]
    paths += list((ROOT / "cafl4ds").rglob("*.py"))
    paths += [
        ROOT / f"scripts/{name}.py"
        for name in (
            "run_short_buffer_experiment",
            "run_adaptation_experiment",
            "run_paired_update_response",
            "evaluate_checkpoint_robustness",
            "confirm_replay_advantage",
            "confirm_last_block",
        )
    ]
    lock = {
        "hashes": {str(p): digest(p) for p in paths},
        "config": OmegaConf.to_container(config, resolve=True),
        "policies": POLICIES,
        "seeds": SEEDS,
        "draws": DRAWS,
        "arrivals": 6144,
        "clips": [c["clip_id"] for c in intake["clips"]],
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "device": torch.cuda.get_device_name(),
        "precision": "float32; AMP/TF32 off; deterministic algorithms",
        "query_role": "reused_validation_development",
        "scientific_source_independence": False,
        "sealed_queries_scored": False,
        "validation_labels": "original local BDD validation records",
        "old_cpu_runtime_reproduced": False,
        "selector": {"window": 64, "history": 16, "confirmation": 3, "exploration_slots": 16},
        "timing_scope": "host-cache gather, transfers, all reference features, scoring/history/assembly, synchronized SSL; no acquisition/decoding/readouts",
    }
    lock = json.loads(json.dumps(lock))
    if (out / "input_lock.json").exists():
        if json.loads((out / "input_lock.json").read_text()) != lock:
            raise ValueError("Resume sources/data/runtime differ")
    else:
        atomic_json(out / "input_lock.json", lock)
        atomic_json(
            out / "data-role-manifest.json",
            {
                "clips": intake["clips"],
                "evaluation": manifest,
                "reader_images": reader_rows,
                "warm_overlap_count": 0,
                "validation_clip_overlap_count": 0,
                "validation_pixel_overlap": 0,
                "development_only": True,
                "source_group_level": "clip only; ride/session independence unknown",
                "attributes_scope": "keyframe attributes used only for diagnostic strata; not frame labels",
            },
        )
        for path in paths:
            if path.suffix in (".py", ".md") and path.is_relative_to(ROOT):
                target = out / "source_snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    return config, arrivals, Evaluation(images, labels, domains, query, supports, manifest), intake["clips"], lock


def selection_diagnostic(initial, arrivals, clips, out):
    """Inspect fixed candidate diets without training or querying downstream outcomes."""
    if (out / "selection-diagnostic.json").exists():
        payload = json.loads((out / "selection-diagnostic.json").read_text())
        features = torch.load(out / "reference_features.pt", map_location="cpu", weights_only=True)
        return payload, features
    reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
    with torch.no_grad():
        features = torch.cat([reference.embed(batch.cuda()).cpu() for batch in arrivals.split(32)])
    torch.save(features, out / "reference_features.pt")
    history = None
    numpy_history = None
    rows = []
    max_error = 0.0
    for start in range(0, len(arrivals), 64):
        if start % 128 == 0:
            history, numpy_history = None, None
        current = features[start : start + 64]
        indices, detail = persistent_indices(current, history)
        expected, scores, _, numpy_history = independent_selection(current.numpy(), numpy_history)
        error = float(np.max(np.abs(scores - np.array(detail["scores"]))))
        max_error = max(max_error, error)
        if not np.array_equal(expected, indices.numpy()) or error > 1e-6:
            raise ValueError("Independent NumPy selection calculation differs")
        history = torch.nn.functional.normalize(current[indices][-16:], dim=1)
        uniform = set(range(1, 64, 2))
        overlap = len(uniform.intersection(indices.tolist()))
        clip = clips[start // 128]
        attrs = clip["keyframe_attributes"]
        row = {
            "start": start,
            "selected_positions": indices.tolist(),
            "detail": detail,
            "replacement_fraction_vs_uniform": 1 - overlap / 32,
            "clip_id": clip["clip_id"],
            "keyframe_stratum": f"{attrs['timeofday']}/{attrs['scene']}",
            "arriving_mean_intensity": float(arrivals[start : start + 64].mean()),
            "selected_mean_intensity": float(arrivals[start : start + 64][indices].mean()),
        }
        rows.append(row)
    replacement = statistics.fmean(r["replacement_fraction_vs_uniform"] for r in rows)
    strata = {}
    for row in rows:
        group = strata.setdefault(row["keyframe_stratum"], {"windows": 0, "arrivals": 0, "accepted": 0})
        group["windows"] += 1
        group["arrivals"] += 64
        group["accepted"] += 32
    if replacement < 0.1 or any(v["accepted"] * 2 != v["arrivals"] for v in strata.values()):
        raise ValueError("Selector has no distinct diet or violates stratum quota")
    result = {
        "passed": True,
        "replacement_fraction_vs_uniform": replacement,
        "window_count": 96,
        "numpy_score_max_error": max_error,
        "all_independent_indices_match": True,
        "strata": strata,
        "rows": rows,
    }
    atomic_json(out / "selection-diagnostic.json", result)
    for day in ("daytime", "night"):
        eligible = [(i, c) for i, c in enumerate(clips) if c["keyframe_attributes"]["timeofday"] == day]
        index, clip = min(
            eligible, key=lambda pair: hashlib.sha256(f"inspection:20261013:{pair[1]['clip_id']}".encode()).hexdigest()
        )
        start = index * 128
        row = rows[start // 64]
        chosen = set(row["selected_positions"])
        fig = Figure(figsize=(10, 10), layout="constrained")
        axes = fig.subplots(8, 8)
        for position, axis in enumerate(axes.flat):
            axis.imshow(arrivals[start + position].permute(1, 2, 0).numpy())
            axis.set_xticks([])
            axis.set_yticks([])
            color = "#2374ab" if position in ANCHORS.tolist() else "#248232" if position in chosen else "#bbbbbb"
            for spine in axis.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            axis.set_title(str(position), fontsize=8, color=color)
        fig.suptitle(f"{day}: {clip['clip_id']} | blue=anchor, green=content, gray=rejected", fontsize=11)
        fig.savefig(out / f"selection-{day}.png", dpi=120)
    return result, features


def preflight(initial, config, arrivals, data, out):
    if (out / "preflight.json").exists():
        return json.loads((out / "preflight.json").read_text())
    torch.manual_seed(SEEDS[0])
    optimizer = instantiate(config.optim, params=list(initial.parameters()))
    parent = capture(initial, optimizer)
    marker = fingerprint(parent)
    reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
    work = 0
    for policy in POLICIES:
        model, optim = restore(initial, config.optim, parent)
        training = run_stream(model, optim, arrivals[:128], policy, reference, SEEDS[0])
        work += training["updates"]
    nonempty = capture(model, optim)
    terminals = []
    for _ in range(2):
        repeated, repeated_optim = restore(initial, config.optim, nonempty)
        training = run_stream(repeated, repeated_optim, arrivals[128:256], "persistent_half", reference, SEEDS[0])
        work += training["updates"]
        terminals.append(fingerprint(capture(repeated, repeated_optim)))
    if terminals[0] != terminals[1] or fingerprint(parent) != marker:
        raise ValueError("Complete CUDA continuation/restoration failed")
    outcome = evaluate_gpu(model, optim, data, out / "preflight")
    result = {
        "passed": True,
        "discarded_ssl_updates": work,
        "discarded_primary_fits": 3,
        "cuda_exact_continuation": True,
        "readout_state_unchanged": outcome["ssl_state_unchanged"],
        "representative_readout_seconds": outcome["readout_seconds"],
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(out / "preflight.json", result)
    return result


def save_terminal(directory, model, optimizer, training):
    state = {
        "learner": capture(model, optimizer),
        "selector": training.pop("selector_state"),
        "arrivals": 6144,
        "updates": training["updates"],
        "accepted_frames": training["accepted_frames"],
    }
    partial = directory / "terminal.partial"
    torch.save(state, partial)
    partial.replace(directory / "terminal.pt")
    atomic_json(directory / "training.json", training)
    return state


def audit_trajectories(initial, config, arrivals, features, records, out, progress):
    reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
    checked = []
    max_score_error = 0.0
    for number, row in enumerate(records):
        progress(stage="replay_audit", audited_arms=number, policy=row["policy"], seed=row["seed"])
        torch.manual_seed(row["seed"])
        optimizer = instantiate(config.optim, params=list(initial.parameters()))
        parent = capture(initial, optimizer)
        if fingerprint(parent) != row["initial_fingerprint"]:
            raise ValueError("Initial replay state differs")
        model, optim = restore(initial, config.optim, parent)
        training = run_stream(model, optim, arrivals, row["policy"], reference, row["seed"])
        state = {
            "learner": capture(model, optim),
            "selector": training.pop("selector_state"),
            "arrivals": 6144,
            "updates": training["updates"],
            "accepted_frames": training["accepted_frames"],
        }
        directory = out / row["directory"]
        stored = torch.load(directory / "terminal.pt", map_location="cuda", weights_only=True)
        if fingerprint(state) != row["terminal_fingerprint"] or fingerprint(stored) != row["terminal_fingerprint"]:
            raise ValueError("Complete replayed/stored terminal differs")
        trace = json.loads((directory / "training.json").read_text())
        if len(trace["blocks"]) != 96:
            raise ValueError("Wrong saved arrival horizon")
        numpy_history = None
        for original, repeated in zip(trace["blocks"], training["blocks"], strict=True):
            for key in (
                "start",
                "end",
                "clip_index",
                "clip_reference_reset",
                "selected_indices",
                "reference_source_indices",
                "feature_frames",
                "accepted_frames",
                "updates",
                "loss_gradients",
                "detail",
            ):
                if json.dumps(original[key]) != json.dumps(repeated[key]):
                    raise ValueError(f"Replay trace differs: {key}")
            if row["policy"] == "persistent_half":
                start = original["start"]
                if start % 128 == 0:
                    numpy_history = None
                indices, scores, ref, numpy_history = independent_selection(
                    features[start : start + 64].numpy(), numpy_history
                )
                error = float(np.max(np.abs(scores - np.array(original["detail"]["scores"]))))
                max_score_error = max(max_score_error, error)
                if error > 1e-6 or (indices + start).tolist() != original["selected_indices"]:
                    raise ValueError("Independent candidate scores/selection differ")
        checked.append(
            {"policy": row["policy"], "seed": row["seed"], "updates": training["updates"], "exact_complete_state": True}
        )
    result = {
        "passed": True,
        "arms_checked": len(checked),
        "audit_ssl_updates": sum(r["updates"] for r in checked),
        "numpy_score_max_error": max_score_error,
        "records": checked,
        "complete_model_optimizer_cpu_cuda_rng_selector_reproduced": True,
    }
    atomic_json(out / "training-replay-audit.json", result)


def summarize(data, frozen, records, out):
    all_rows = [("frozen", frozen), *[(r["directory"], r) for r in records]]
    error = 0.0
    counts = {}
    for directory, row in all_rows:
        with np.load(out / directory / "predictions.npz", allow_pickle=False) as predictions:
            for draw in DRAWS:
                for reader in READERS:
                    counts[directory, draw, reader] = exact_correct_counts(
                        predictions[f"{draw}_{reader}"],
                        data.labels[data.query].numpy(),
                        data.domains[data.query].numpy(),
                    )
                    measured = prediction_scores(
                        torch.from_numpy(predictions[f"{draw}_{reader}"].copy()),
                        data.labels[data.query],
                        data.domains[data.query],
                    )
                    for metric in METRICS:
                        error = max(error, abs(measured[metric] - row["panels"][str(draw)][reader][metric]))
        for reader in READERS:
            for metric in METRICS:
                expected = statistics.fmean(row["panels"][str(d)][reader][metric] for d in DRAWS)
                error = max(error, abs(expected - row["scores"][reader][metric]))
    if error > 1e-12:
        raise ValueError("Prediction scoring or panel averaging differs")
    by = {(r["policy"], r["seed"]): r for r in records}
    means = {
        "frozen": {
            reader: {
                metric: 100 * sum(counts["frozen", d, reader][metric] for d in DRAWS) / (DENOMINATORS[metric] * 3)
                for metric in METRICS
            }
            for reader in READERS
        }
    }
    means.update(
        {
            p: {
                reader: {
                    metric: 100
                    * sum(counts[by[p, s]["directory"], d, reader][metric] for s in SEEDS for d in DRAWS)
                    / (DENOMINATORS[metric] * 15)
                    for metric in METRICS
                }
                for reader in READERS
            }
            for p in POLICIES
        }
    )
    contrasts = {}
    comparisons = [(p + "_minus_frozen", p, "frozen") for p in POLICIES]
    comparisons += [(p + "_minus_full", p, "full") for p in POLICIES[1:]]
    comparisons += [
        ("persistent_minus_uniform", "persistent_half", "uniform_half"),
        ("persistent_minus_random", "persistent_half", "random_half"),
    ]
    for name, first, second in comparisons:
        contrast = {"readers": {}, "panels": {}}
        for reader in READERS:
            contrast["readers"][reader] = {
                metric: exact_contrast(by, counts, first, second, reader, metric, DRAWS) for metric in METRICS
            }
        for draw in DRAWS:
            contrast["panels"][str(draw)] = {
                metric: exact_contrast(by, counts, first, second, "finetune", metric, (draw,)) for metric in METRICS
            }
        contrasts[name] = contrast
    timing = {}
    for policy in POLICIES:
        pairs = [1 - by[policy, s]["online_seconds"] / by["full", s]["online_seconds"] for s in SEEDS]
        timing[policy] = {
            "mean_paired_saving_fraction": statistics.fmean(pairs),
            "paired_savings": pairs,
            "online_total_seconds": sum(by[policy, s]["online_seconds"] for s in SEEDS),
        }
    gates = {}
    for policy in POLICIES:
        useful = contrasts[policy + "_minus_frozen"]["readers"]["finetune"]["balanced"]["ci95_pp"][0] > 0
        guard = policy == "full" or all(
            contrasts[policy + "_minus_full"]["readers"]["finetune"][m]["ci95_pp"][0] > -1
            for m in ("balanced", "A", "B")
        )
        saves = policy != "full" and timing[policy]["mean_paired_saving_fraction"] >= 0.25
        gates[policy] = {
            "useful_ssl": useful,
            "quality_guardrail": guard,
            "cost_screen": saves,
            "efficiency_lead": useful and guard and saves,
        }
    content_gates = {
        name: contrasts[name]["readers"]["finetune"]["balanced"]["ci95_pp"][0] > 0
        and contrasts[name]["readers"]["finetune"]["balanced"]["mean_pp"] >= 0.1
        and contrasts[name]["readers"]["finetune"]["balanced"]["wins"] >= 4
        for name in ("persistent_minus_uniform", "persistent_minus_random")
    }
    content = all(content_gates.values())
    quality = content and gates["persistent_half"]["useful_ssl"] and gates["persistent_half"]["quality_guardrail"]
    decision = (
        "persistent_filter_quality_and_efficiency_lead"
        if quality and gates["persistent_half"]["cost_screen"]
        else "persistent_filter_quality_lead_cost_unqualified"
        if quality
        else "static_efficiency_lead_without_content_value"
        if any(g["efficiency_lead"] for g in gates.values())
        else "no_filter_lead_qualified"
    )
    result = {
        "development_only": True,
        "independent_source_confirmation": False,
        "means_accuracy_percent": means,
        "contrasts": contrasts,
        "timing": timing,
        "gates": gates,
        "content_gates": content_gates,
        "content_value_qualified": content,
        "decision": decision,
        "scientific_ssl_updates": sum(r["updates"] for r in records),
        "trained_trajectories": len(records),
        "primary_fine_tuning_fits": 63,
        "readout_audit": {
            "passed": True,
            "prediction_sets": 189,
            "max_error": error,
            "statistics_from_exact_integer_counts": True,
        },
        "uncertainty": "paired t across5 training seeds conditional on one reused constructed stream and averaged support panels; no drive/session interval",
        "timing_scope": "resident-host-cache online work including all selection/reference features; excludes acquisition/decoding/readouts",
    }
    atomic_json(out / "summary.json", result)
    lines = [
        "# Persistent-novelty admission: GPU development results",
        "",
        f"Decision: **{decision}**.",
        "",
        "| Arm | Fine-tune balanced % | Day % | Night % | Gain over frozen pp [conditional95%] | Mean processing saving vs full % |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for policy in ("frozen", *POLICIES):
        m = means[policy]["finetune"]
        if policy == "frozen":
            gain = "—"
            saving = "—"
        else:
            c = contrasts[policy + "_minus_frozen"]["readers"]["finetune"]["balanced"]
            gain = f"{c['mean_pp']:+.3f} [{c['ci95_pp'][0]:+.3f}, {c['ci95_pp'][1]:+.3f}]"
            saving = f"{100 * timing[policy]['mean_paired_saving_fraction']:.1f}"
        lines.append(f"| {policy} | {m['balanced']:.3f} | {m['A']:.3f} | {m['B']:.3f} | {gain} | {saving} |")
    for name in ("persistent_minus_uniform", "persistent_minus_random"):
        c = contrasts[name]["readers"]["finetune"]["balanced"]
        lines += [
            "",
            f"{name}: {c['mean_pp']:+.3f} pp [{c['ci95_pp'][0]:+.3f}, {c['ci95_pp'][1]:+.3f}], {c['wins']}/5 positive seeds, {c['ties']} exact ties.",
        ]
    lines += [
        "",
        f"Content-value screen: **{content}**. Candidate useful-SSL, quality and cost gates: `{gates['persistent_half']}`.",
        "",
        "All20 trained trajectories and2400 scientific SSL updates completed; one shared frozen endpoint plus20 terminals produced63 primary fine-tuning fits. All189 prediction sets were independently rescored; complete training/RNG/selector replay and separate NumPy selection audits passed.",
        "",
        "This uses the earlier positive temporal allocation and original validation labels, not the provisional test-label population of the short-buffer screen. Clip ordering is constructed; ride/session independence and unseen-source transfer remain unestablished. All outcomes are development evidence; sealed final queries were untouched.",
        "",
        "GPU float32 synchronized timing includes host gathering/transfers, all reference feature extraction, persistence/history selection and SSL. Acquisition, decoding, initial resize, checkpoint IO and downstream fitting are excluded. GPU and historical CPU timing/results are not pooled.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = Figure(figsize=(11, 4), layout="constrained")
    axes = fig.subplots(1, 2)
    values = [contrasts[p + "_minus_frozen"]["readers"]["finetune"]["balanced"] for p in POLICIES]
    x = np.arange(len(POLICIES))
    means_pp = np.array([v["mean_pp"] for v in values])
    bounds = np.array([v["ci95_pp"] for v in values])
    axes[0].errorbar(means_pp, x, xerr=np.stack((means_pp - bounds[:, 0], bounds[:, 1] - means_pp)), fmt="o", capsize=4)
    axes[0].set_yticks(x, POLICIES)
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set(xlabel="Fine-tuning gain over frozen (pp)", title="Conditional five-seed ranges; reused stream")
    axes[1].barh(POLICIES, [100 * timing[p]["mean_paired_saving_fraction"] for p in POLICIES])
    axes[1].set(xlabel="Mean paired processing saving versus full (%)", title="RTX2080; all selection costs charged")
    fig.savefig(out / "results.png", dpi=150)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "completed_trained_arms": 0, "total_trained_arms": 20, "scientific_updates_cap": 2400}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        progress(stage="locking_original_environment")
        config, arrivals, data, clips, lock = load_inputs(out, progress)
        torch.manual_seed(11)
        initial = _method(config).cuda()
        progress(stage="inspect_filter_diet_before_training")
        diagnostic, features = selection_diagnostic(initial, arrivals, clips, out)
        if not diagnostic["passed"]:
            raise ValueError("Selection diagnostic failed")
        if args.prepare_only:
            progress(
                stage="prepared_no_scientific_training",
                replacement_fraction=diagnostic["replacement_fraction_vs_uniform"],
            )
            return
        progress(stage="discarded_gpu_preflight")
        if not preflight(initial, config, arrivals, data, out)["passed"]:
            raise ValueError("Preflight failed")
        frozen_dir = out / "frozen"
        frozen_dir.mkdir(exist_ok=True)
        if (frozen_dir / "result.json").exists():
            frozen = json.loads((frozen_dir / "result.json").read_text())
        else:
            progress(stage="shared_frozen_readout")
            torch.manual_seed(SEEDS[0])
            optim = instantiate(config.optim, params=list(initial.parameters()))
            frozen = evaluate_gpu(initial, optim, data, frozen_dir)
            atomic_json(frozen_dir / "result.json", frozen)
        reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
        reference_marker = fingerprint(reference.state_dict())
        records = []
        for position, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            optim = instantiate(config.optim, params=list(initial.parameters()))
            parent = capture(initial, optim)
            parent_marker = fingerprint(parent)
            shift = position % 4
            for policy in POLICIES[shift:] + POLICIES[:shift]:
                directory = out / "runs" / policy / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                result_path = directory / "result.json"
                if result_path.exists():
                    records.append(json.loads(result_path.read_text()))
                    progress(completed_trained_arms=len(records))
                    continue
                progress(stage="training", policy=policy, seed=seed)
                model, optimizer = restore(initial, config.optim, parent)
                if (directory / "terminal.pt").exists() and (directory / "training.json").exists():
                    state = torch.load(directory / "terminal.pt", map_location="cuda", weights_only=True)
                    model, optimizer = restore(initial, config.optim, state["learner"])
                    training = json.loads((directory / "training.json").read_text())
                else:
                    training = run_stream(model, optimizer, arrivals, policy, reference, seed)
                    state = save_terminal(directory, model, optimizer, training)
                if training["updates"] != (192 if policy == "full" else 96):
                    raise ValueError("Scientific work mismatch")
                progress(stage="terminal_readout", policy=policy, seed=seed)
                outcome = evaluate_gpu(model, optimizer, data, directory)
                if fingerprint(parent) != parent_marker or fingerprint(reference.state_dict()) != reference_marker:
                    raise ValueError("Parent/reference mutated")
                row = {
                    "policy": policy,
                    "seed": seed,
                    **outcome,
                    "updates": training["updates"],
                    "accepted_frames": training["accepted_frames"],
                    "online_seconds": training["online_seconds"],
                    "initial_fingerprint": parent_marker,
                    "terminal_fingerprint": fingerprint(state),
                    "directory": str(directory.relative_to(out)),
                }
                atomic_json(result_path, row)
                records.append(row)
                progress(completed_trained_arms=len(records))
        if any(digest(Path(p)) != v for p, v in lock["hashes"].items()):
            raise ValueError("Locked inputs changed")
        audit_trajectories(initial, config, arrivals, features, records, out, progress)
        progress(stage="independent_readout_reconstruction")
        result = summarize(data, frozen, records, out)
        if result["scientific_ssl_updates"] != 2400 or len(records) != 20:
            raise ValueError("Final cap mismatch")
        progress(
            stage="complete",
            completed_trained_arms=20,
            audited_arms=20,
            scientific_ssl_updates=2400,
            primary_fine_tuning_fits=63,
            decision=result["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
