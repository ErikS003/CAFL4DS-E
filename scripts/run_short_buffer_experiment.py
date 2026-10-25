"""Frozen five-arm GPU development screen, with isolated downstream evaluation."""

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
from PIL import Image
from matplotlib.figure import Figure
from sklearn import __version__ as sklearn_version

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation, prediction_scores
from cafl4ds.short_buffer import ARMS, capture, restore, run_episode
from scripts.evaluate_checkpoint_robustness import probe_predictions
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_temporal_state_development import order_cohort

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/temporal-state-opportunity/20261011-development-v1"
SEEDS = (5003, 5009)
DRAWS = (51011, 51017, 51023)
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B", "micro")
PROTOCOL = ROOT / "docs/project-plan/slow-edge-short-buffer-frozen.md"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256")
    return value.hexdigest()


def remap(path: str) -> Path:
    """Relocate prior-machine paths without modifying historical artifacts."""
    text = path.replace("\\", "/")
    if "/cafl4ds/outputs/" in text:
        return ROOT / "outputs" / text.split("/cafl4ds/outputs/", 1)[1]
    if "/datasets/bdd100k-full/" in text:
        return ROOT.parents[1] / "datasets/bdd100k-full" / text.split("/datasets/bdd100k-full/", 1)[1]
    raise ValueError(f"Unknown historical path: {path}")


def load_inputs(out: Path):
    prepared_path = PRIOR / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    if prepared["sealed_final_predictions_allowed"] or prepared["scientific_independence_established"]:
        raise ValueError("Expected provisional development allocation")
    old_path = ROOT / "outputs/paired-update-response/20261010-states/input_lock.json"
    old = json.loads(old_path.read_text())
    config = OmegaConf.create(old["config"])
    warm_original = str(config.well)
    warm = remap(warm_original)
    if digest(warm) != old["hashes"][warm_original]:
        raise ValueError("Warm checkpoint hash differs")
    config.well = str(warm)
    config.device = "cuda"
    cohorts = []
    source_paths = [prepared_path, old_path, warm, PROTOCOL]
    for cohort in (0, 1):
        members = [c for c in prepared["clips"] if c["role"] == f"stream_cohort_{cohort}"]
        ordered = order_cohort(members, cohort)[:6]
        if len(ordered) != 6:
            raise ValueError("Expected six selected clips")
        for clip in ordered:
            frames = remap(clip["frames_path"])
            if digest(frames) != clip["frames_sha256"]:
                raise ValueError("Frame cache differs")
            clip["local_frames_path"] = str(frames)
            source_paths.append(frames)
        cohorts.append(ordered)
    reader_root = PRIOR / "experiment"
    arrays_path = reader_root / "evaluation_arrays.npz"
    images_path = reader_root / "evaluation_images.pt"
    manifest_path = reader_root / "reader-allocation.json"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        labels = torch.from_numpy(arrays["labels"].copy())
        domains = torch.from_numpy(arrays["domains"].copy())
        query = torch.from_numpy(arrays["query"].copy())
        supports = {d: torch.from_numpy(arrays[f"support_{d}"].copy()) for d in DRAWS}
    images = torch.load(images_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["selected"]
    scenes = ("city street", "highway", "residential")
    stream_ids = {c["clip_id"] for group in cohorts for c in group}
    allocation = {r["clip_id"]: r for r in prepared["allocation"]}
    if images.shape != (993, 3, 64, 64) or len(query) != 768 or len(selected) != 993:
        raise ValueError("Reader dimensions differ")
    original_label_files = 0
    for index, row in enumerate(selected):
        original = allocation[row["clip_id"]]
        if row["role"] != original["role"] or row["keyframe_attributes"] != original["keyframe_attributes"]:
            raise ValueError("Archived reader allocation differs from prepared input")
        if row["role"] == "sealed_final_query" or row["clip_id"] in stream_ids:
            raise ValueError("Data-role leakage")
        attrs = row["keyframe_attributes"]
        if int(labels[index]) != scenes.index(attrs["scene"]) or int(domains[index]) != (attrs["timeofday"] == "night"):
            raise ValueError("Reader labels differ from archived attributes")
        image_path = remap(row["image_path"])
        if digest(image_path) != row["image_sha256"]:
            raise ValueError("Reader source image differs")
        with Image.open(image_path) as opened:
            pixels = np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
        expected = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        if not torch.equal(expected, images[index]):
            raise ValueError("Archived reader pixels differ from source")
        source_paths.append(image_path)
        label_path = remap(row["label_path"])
        if label_path.exists():
            if digest(label_path) != row["label_sha256"]:
                raise ValueError("Original label hash differs")
            original_label_files += 1
            source_paths.append(label_path)
    if not torch.all((query >= 225) & (query < 993)):
        raise ValueError("Wrong query positions")
    for panel, draw in enumerate(DRAWS):
        if supports[draw].tolist() != list(range(panel * 75, (panel + 1) * 75)):
            raise ValueError("Fixed support membership differs")
        if any(selected[i]["role"] != f"support_panel_{panel}" for i in supports[draw].tolist()):
            raise ValueError("Support role differs")
    if any(selected[i]["role"] != "development_query" for i in query.tolist()):
        raise ValueError("Query role differs")
    for domain in (0, 1):
        for label in range(3):
            if int(((labels[query] == label) & (domains[query] == domain)).sum()) != 128:
                raise ValueError("Queries are not six-cell balanced")
    source_paths += [
        arrays_path,
        images_path,
        manifest_path,
        Path(__file__),
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "cafl4ds/update_filtering.py",
        ROOT / "scripts/evaluate_checkpoint_robustness.py",
    ]
    source_paths += list((ROOT / "cafl4ds").rglob("*.py"))
    source_paths += [
        ROOT / f"scripts/{name}.py"
        for name in (
            "run_adaptation_experiment",
            "run_temporal_state_development",
            "run_paired_update_response",
            "run_recurrence_retention",
            "confirm_last_block",
            "confirm_replay_advantage",
        )
    ]
    lock = {
        "hashes": {str(p): digest(p) for p in source_paths},
        "config": OmegaConf.to_container(config, resolve=True),
        "seeds": SEEDS,
        "draws": DRAWS,
        "bundles": [[c["clip_id"] for c in group] for group in cohorts],
        "arms": ARMS,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "sklearn": sklearn_version,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "precision": "float32; AMP and TF32 disabled",
        "deterministic_algorithms": True,
        "historical_cpu_runtime": "2.13.0+cpu; not pooled with this GPU run",
        "query_role": "reused_development_query",
        "sealed_final_scored": False,
        "annotation_provenance_verified": False,
        "capture_sessions_known": False,
        "original_reader_label_files_present": original_label_files,
        "label_source": "archived selected attributes, independently matched to saved arrays; no new annotation",
        "timing_scope": "host-resident RGB cache: gather, host-device transfer, selection, all reference features, SSL",
        "excluded_costs": ["acquisition", "video decoding", "image resize", "checkpoint IO", "downstream readouts"],
    }
    lock = json.loads(json.dumps(lock))
    lock_path = out / "input_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Frozen inputs/runtime changed; preserve the prior output")
    else:
        atomic_json(lock_path, lock)
        atomic_json(
            out / "data-role-manifest.json",
            {
                "bundles": [
                    {
                        "bundle": b,
                        "clips": [
                            {
                                k: c[k]
                                for k in (
                                    "clip_id",
                                    "role",
                                    "keyframe_attributes",
                                    "frames_sha256",
                                    "local_frames_path",
                                )
                            }
                            for c in group
                        ],
                    }
                    for b, group in enumerate(cohorts)
                ],
                "reader": manifest,
                "sealed_final_scored": False,
                "development_only": True,
                "annotation_provenance_verified": False,
                "capture_sessions_known": False,
                "original_reader_label_files_present": original_label_files,
                "caveat": "Artificial clip concatenation and reused archived labels; no independent confirmation claim",
            },
        )
        for path in source_paths:
            if path.suffix in (".py", ".md") and path.is_relative_to(ROOT):
                target = out / "source_snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    return config, cohorts, Evaluation(images, labels, domains, query, supports, manifest), lock


def encode(encoder, images):
    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        return torch.cat([encoder.embed(batch.to(device)).cpu() for batch in images.split(64)])


def fine_tune_gpu(encoder, images, labels, seed):
    """Preserve the fixed 100-step, support-only recipe on an independent GPU copy."""
    model = copy.deepcopy(encoder).requires_grad_(True).train()
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    head = torch.nn.Linear(model.embed_dim, 3).to(device)
    generator = torch.Generator().manual_seed(seed)
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.05
    )
    losses = []
    for _ in range(100):
        indices = torch.randperm(len(labels), generator=generator)[:32].to(device)
        optimizer.zero_grad()
        loss = F_cross_entropy(head(model.embed(images[indices])), labels[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    delta = (
        sum(float((p - q).square().sum()) for p, q in zip(model.parameters(), encoder.parameters(), strict=True)) ** 0.5
    )
    if delta <= 0 or not np.isfinite(losses).all():
        raise ValueError("Invalid fine-tuning updates")
    return model, head, {"steps": 100, "backbone_delta_l2": delta, "initial_loss": losses[0], "final_loss": losses[-1]}


F_cross_entropy = torch.nn.functional.cross_entropy


def evaluate_gpu(model, optimizer, data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    marker = fingerprint(capture(model, optimizer))
    modes = {name: module.training for name, module in model.named_modules()}
    predictions, panels, audits = {}, {}, {}
    tick = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[0]):
            encoded = encode(model.encoder, data.images)
            query_labels, query_domains = data.labels[data.query], data.domains[data.query]
            for draw, support in data.supports.items():
                scores = {}
                for kind in ("linear", "knn"):
                    predicted = probe_predictions(kind, encoded[support], data.labels[support], encoded[data.query])
                    scores[kind] = prediction_scores(predicted, query_labels, query_domains)
                    predictions[f"{draw}_{kind}"] = predicted.numpy()
                tuned, head, audit = fine_tune_gpu(model.encoder, data.images[support], data.labels[support], draw + 25)
                z = encode(tuned, data.images[data.query])
                with torch.no_grad():
                    predicted = head(z.to("cuda")).argmax(1).cpu()
                scores["finetune"] = prediction_scores(predicted, query_labels, query_domains)
                predictions[f"{draw}_finetune"] = predicted.numpy()
                panels[str(draw)], audits[str(draw)] = scores, audit
                del tuned, head
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Evaluation mutated learner/optimizer/RNG state")
    result = {
        "panels": panels,
        "fine_tuning_audits": audits,
        "scores": {
            reader: {metric: statistics.fmean(p[reader][metric] for p in panels.values()) for metric in METRICS}
            for reader in READERS
        },
        "readout_seconds": time.perf_counter() - tick,
        "ssl_state_unchanged": True,
    }
    np.savez_compressed(directory / "predictions.npz", **predictions)
    return result


def historical_audit(initial, config, out, progress):
    """Audit every saved legacy tensor/fingerprint and exact round-trip restoration."""
    from cafl4ds.temporal_action_response import restore as legacy_restore, snapshot

    audit_path = out / "historical-checkpoint-audit.json"
    if audit_path.exists():
        return json.loads(audit_path.read_text())
    legacy = copy.deepcopy(initial).cpu()
    checked, artifact_hashes = 0, {}
    paired_paths = sorted((PRIOR / "experiment/histories").glob("cohort_*/seed_*/*/decision_*/paired.json"))
    if len(paired_paths) != 72:
        raise ValueError("Missing legacy paired parents")

    def audit_state(path, expected, arrivals, updates):
        nonlocal checked
        state = torch.load(path, map_location="cpu", weights_only=True)
        if fingerprint(state) != expected or state["arrivals"] != arrivals or state["optimizer_updates"] != updates:
            raise ValueError(f"Checkpoint identity/clocks differ: {path}")
        for value in state["model"].values():
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite historical parameter")
        for value in state["optimizer"]["state"].values():
            if int(value["step"]) != updates or any(
                not torch.isfinite(v).all() for v in value.values() if isinstance(v, torch.Tensor)
            ):
                raise ValueError("Historical Adam state differs from update clock")
        parent_marker = fingerprint(state)
        model, optimizer = legacy_restore(legacy, config, state)
        if (
            fingerprint(snapshot(model, optimizer, updates, arrivals)) != parent_marker
            or fingerprint(state) != parent_marker
        ):
            raise ValueError("Legacy round-trip changed state")
        # In-place writes to restored tensors must not change the saved parent.
        with torch.no_grad():
            next(model.parameters()).add_(1)
        for value in optimizer.state.values():
            if "exp_avg" in value:
                value["exp_avg"].add_(1)
                break
        if fingerprint(state) != parent_marker:
            raise ValueError("Legacy restoration aliases parent")
        checked += 1
        artifact_hashes[str(path)] = digest(path)

    for index, path in enumerate(paired_paths):
        row = json.loads(path.read_text())
        parent_dir = path.parent.parent
        parent_updates = row["offset"] * (2 if row["history"] == "full" else 1)
        audit_state(
            parent_dir / f"parent_{row['offset']}.pt", row["parent_fingerprint"], row["offset"] * 64, parent_updates
        )
        for action in ("full", "half"):
            windows = row[action]["windows"]
            expected_updates = 4 if action == "full" else 2
            if sum(w["optimizer_updates"] for w in windows) != expected_updates:
                raise ValueError("Legacy branch work mismatch")
            audit_state(
                path.parent / action / "terminal.pt",
                row[action]["terminal_fingerprint"],
                (row["offset"] + 2) * 64,
                parent_updates + expected_updates,
            )
        if index % 12 == 0:
            progress(stage="historical_tensor_audit", audited_parents=index + 1)
    for latest_path in sorted((PRIOR / "experiment/histories").glob("cohort_*/seed_*/*/latest.pt")):
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        parent = torch.load(latest_path.parent / "parent_74.pt", map_location="cpu", weights_only=True)
        if fingerprint(latest["state"]) != fingerprint(parent) or len(latest["windows"]) != 74:
            raise ValueError("Legacy resumed latest state differs from saved parent")
        artifact_hashes[str(latest_path)] = digest(latest_path)
    result = {
        "passed": True,
        "paired_parents": 72,
        "tensor_states_checked": checked,
        "resumable_histories_checked": 36,
        "exact_fingerprint_and_restoration": True,
        "saved_model_optimizer_cpu_rng_checked": True,
        "historical_training_reexecuted": False,
        "historical_cpu_runtime_reproduced": False,
        "remaining_limit": "Original CPU runtime 2.13.0 differs; audit certifies saved tensor consistency/restoration, not historical bitwise replay or annotation provenance",
        "retry_record": "Legacy temporal_checkpoint_retry.py retries only latest.partial to latest.pt sharing locks; original logs/status preserved",
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(audit_path, result)
    return result


def preflight(initial, config, first_images, data, out):
    torch.manual_seed(5003)
    optimizer = instantiate(config.optim, params=[p for p in initial.parameters() if p.requires_grad])
    parent = capture(initial, optimizer)
    reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
    records = {}
    for arm in ARMS[1:]:
        model, optim = restore(initial, config.optim, parent)
        records[arm] = run_episode(model, optim, first_images[:128], arm, reference, 5003)
    model, optim = restore(initial, config.optim, parent)
    # Warm CUDA/optimizer kernels before the measured representative job.
    run_episode(model, optim, first_images[:128], "full", reference, 5003)
    nonempty = capture(model, optim)
    endings = []
    for _ in range(2):
        repeated, repeated_optim = restore(initial, config.optim, nonempty)
        run_episode(repeated, repeated_optim, first_images[128:256], "full", reference, 5003)
        endings.append(fingerprint(capture(repeated, repeated_optim)))
    if endings[0] != endings[1] or fingerprint(parent) != fingerprint(capture(*restore(initial, config.optim, parent))):
        raise ValueError("CUDA restored continuation differs")
    tick = time.perf_counter()
    readout = evaluate_gpu(model, optim, data, out / "preflight")
    result = {
        "passed": True,
        "cuda_exact_continuation": True,
        "evaluation_ssl_state_unchanged": readout["ssl_state_unchanged"],
        "representative_readout_seconds": time.perf_counter() - tick,
        "discarded_ssl_updates": 22,
        "discarded_primary_fits": 3,
        "online_block_seconds": {a: r["online_seconds"] for a, r in records.items()},
        "estimated_main_seconds": 20 * readout["readout_seconds"]
        + 12 * sum(r["online_seconds"] for r in records.values()),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(out / "preflight.json", result)
    return result


def audit_new_trajectories(initial, config, episodes, records, out, progress):
    """Reexecute all selected training trajectories and compare complete CUDA states."""
    reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
    checked = []
    for index, row in enumerate(records):
        progress(stage="independent_training_replay", audited_arms=index, arm=row["arm"])
        torch.manual_seed(row["seed"])
        optimizer = instantiate(config.optim, params=[p for p in initial.parameters() if p.requires_grad])
        parent = capture(initial, optimizer)
        if fingerprint(parent) != row["initial_fingerprint"]:
            raise ValueError("Audit initial RNG/model differs")
        model, optimizer = restore(initial, config.optim, parent)
        training = run_episode(model, optimizer, episodes[row["bundle"]], row["arm"], reference, row["seed"])
        selector_rng = training.pop("selector_rng")
        state = {
            "learner": capture(model, optimizer),
            "selector_rng": selector_rng,
            "buffer": [],
            "arrivals": 1536,
            "updates": training["updates"],
            "accepted_frames": training["accepted_frames"],
        }
        directory = out / row["relative_directory"]
        saved = torch.load(directory / "terminal.pt", map_location="cuda", weights_only=True)
        if fingerprint(state) != row["terminal_fingerprint"] or fingerprint(saved) != row["terminal_fingerprint"]:
            raise ValueError("Independent GPU trajectory or stored terminal differs")
        trace = json.loads((directory / "training.json").read_text())
        if len(trace["blocks"]) != len(training["blocks"]):
            raise ValueError("Independent block count differs")
        for expected, actual in zip(trace["blocks"], training["blocks"], strict=True):
            for key in (
                "start",
                "end",
                "selected_indices",
                "feature_frames",
                "accepted_frames",
                "updates",
                "loss_gradients",
            ):
                if json.dumps(expected[key]) != json.dumps(actual[key]):
                    raise ValueError(f"Independent selection/training replay differs: {key}")
        checked.append(
            {
                "bundle": row["bundle"],
                "seed": row["seed"],
                "arm": row["arm"],
                "exact_terminal": True,
                "exact_selection_losses": True,
                "updates": training["updates"],
            }
        )
    result = {
        "passed": True,
        "arms_checked": len(checked),
        "audit_ssl_updates": sum(r["updates"] for r in checked),
        "complete_model_optimizer_cpu_cuda_rng_reproduced": True,
        "records": checked,
    }
    atomic_json(out / "independent-training-audit.json", result)
    return result


def report(out, records, data):
    """Reconstruct every primary/secondary score independently from saved predictions."""
    max_error = 0.0
    for row in records:
        with np.load(out / row["relative_directory"] / "predictions.npz", allow_pickle=False) as arrays:
            for draw in DRAWS:
                for reader in READERS:
                    scores = prediction_scores(
                        torch.from_numpy(arrays[f"{draw}_{reader}"].copy()),
                        data.labels[data.query],
                        data.domains[data.query],
                    )
                    for metric in METRICS:
                        max_error = max(max_error, abs(scores[metric] - row["panels"][str(draw)][reader][metric]))
    if max_error > 1e-12:
        raise ValueError("Independent readout reconstruction failed")
    by_key = {(r["bundle"], r["seed"], r["arm"]): r for r in records}
    means = {
        a: {
            reader: {
                metric: 100 * statistics.fmean(by_key[b, s, a]["scores"][reader][metric] for b in (0, 1) for s in SEEDS)
                for metric in METRICS
            }
            for reader in READERS
        }
        for a in ARMS
    }
    contrasts = {}
    comparisons = [(a + "_minus_frozen", a, "frozen") for a in ARMS[1:]]
    comparisons += [(a + "_minus_full", a, "full") for a in ARMS[2:]]
    comparisons += [
        ("buffering", "buffered_random_half", "uniform_half"),
        ("content", "buffered_diverse_half", "buffered_random_half"),
    ]
    for name, first, second in comparisons:
        differences = [
            {
                "bundle": b,
                "seed": s,
                "pp": {
                    reader: {
                        metric: 100
                        * (
                            by_key[b, s, first]["scores"][reader][metric]
                            - by_key[b, s, second]["scores"][reader][metric]
                        )
                        for metric in METRICS
                    }
                    for reader in READERS
                },
                "panels": {
                    str(d): {
                        metric: 100
                        * (
                            by_key[b, s, first]["panels"][str(d)]["finetune"][metric]
                            - by_key[b, s, second]["panels"][str(d)]["finetune"][metric]
                        )
                        for metric in METRICS
                    }
                    for d in DRAWS
                },
            }
            for b in (0, 1)
            for s in SEEDS
        ]
        contrasts[name] = {
            "pairs": differences,
            "mean_pp": {
                reader: {metric: statistics.fmean(r["pp"][reader][metric] for r in differences) for metric in METRICS}
                for reader in READERS
            },
            "bundle_mean_primary_pp": {
                str(b): statistics.fmean(r["pp"]["finetune"]["balanced"] for r in differences if r["bundle"] == b)
                for b in (0, 1)
            },
            "seed_mean_primary_pp": {
                str(s): statistics.fmean(r["pp"]["finetune"]["balanced"] for r in differences if r["seed"] == s)
                for s in SEEDS
            },
            "draw_mean_primary_pp": {
                str(d): statistics.fmean(r["panels"][str(d)]["balanced"] for r in differences) for d in DRAWS
            },
        }
    timing = {}
    for arm in ARMS[1:]:
        ratios = [
            by_key[b, s, arm]["online_seconds"] / by_key[b, s, "full"]["online_seconds"] for b in (0, 1) for s in SEEDS
        ]
        total = sum(by_key[b, s, arm]["online_seconds"] for b in (0, 1) for s in SEEDS)
        full_total = sum(by_key[b, s, "full"]["online_seconds"] for b in (0, 1) for s in SEEDS)
        timing[arm] = {
            "total_online_seconds": total,
            "summed_time_saving_fraction": 1 - total / full_total,
            "mean_paired_saving_fraction": 1 - statistics.fmean(ratios),
            "paired_saving_fractions": [1 - r for r in ratios],
        }

    def positive_screen(contrast):
        return contrast["mean_pp"]["finetune"]["balanced"] >= 0.1 and all(
            v > 0
            for key in ("bundle_mean_primary_pp", "seed_mean_primary_pp", "draw_mean_primary_pp")
            for v in contrast[key].values()
        )

    screens = {}
    for arm in ARMS[1:]:
        useful = positive_screen(contrasts[arm + "_minus_frozen"])
        noninferior = arm == "full" or all(
            r["panels"][str(d)][metric] > -1
            for r in contrasts[arm + "_minus_full"]["pairs"]
            for d in DRAWS
            for metric in ("balanced", "A", "B")
        )
        saves = arm != "full" and timing[arm]["summed_time_saving_fraction"] >= 0.25
        screens[arm] = {
            "useful_ssl_pilot_screen": useful,
            "descriptive_quality_guardrail": noninferior,
            "charged_cost_screen": saves,
            "budget_quality_lead": useful and noninferior and saves,
        }
    content = positive_screen(contrasts["content"])
    result = {
        "development_only": True,
        "independent_confirmation": False,
        "means_accuracy_percent": means,
        "contrasts": contrasts,
        "timing": timing,
        "screens": screens,
        "content_pilot_screen": content,
        "completed_arms": len(records),
        "scientific_ssl_updates": sum(r["updates"] for r in records),
        "primary_fine_tuning_fits": len(records) * 3,
        "numerical_audit": {"passed": True, "score_error": max_error, "prediction_sets": len(records) * 9},
        "decision": "budget_quality_lead_for_qualification"
        if any(v["budget_quality_lead"] for v in screens.values())
        else "no_qualified_pilot_lead",
        "limitations": [
            "two reused clip bundles; no independent-group confidence interval",
            "unknown label provider/capture sessions",
            "artificial concatenation",
            "GPU timing excludes acquisition/decoding/downstream fitting",
            "GPU runtime differs from historical CPU",
        ],
    }
    atomic_json(out / "summary.json", result)
    lines = [
        "# Five-arm short-buffer GPU development screen",
        "",
        "This is reused-data feasibility evidence, not independent confirmation.",
        "",
        "| Arm | Fine-tune balanced % | Day % | Night % | Gain over frozen pp | Charged time s | Saving vs full % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = means[arm]["finetune"]
        gain = 0 if arm == "frozen" else contrasts[arm + "_minus_frozen"]["mean_pp"]["finetune"]["balanced"]
        t = timing.get(arm, {"total_online_seconds": 0, "summed_time_saving_fraction": 1})
        lines.append(
            f"| {arm} | {m['balanced']:.3f} | {m['A']:.3f} | {m['B']:.3f} | {gain:+.3f} | {t['total_online_seconds']:.2f} | {100 * t['summed_time_saving_fraction']:.1f} |"
        )
    lines += [
        "",
        f"Buffering effect (random buffered minus uniform half): {contrasts['buffering']['mean_pp']['finetune']['balanced']:+.3f} pp.",
        f"Content effect (diverse minus random buffered): {contrasts['content']['mean_pp']['finetune']['balanced']:+.3f} pp.",
        "",
        f"Decision: **{result['decision']}**. Content pilot screen: **{content}**.",
        "",
        "All 480 scientific SSL updates and 60 primary fine-tuning fits completed. All 180 prediction sets were reconstructed independently.",
        "",
        "The descriptive pilot screens do not establish statistical noninferiority or generalization. Bundle, seed and support-draw effects are reported separately in summary.json; none is counted as an independent drive.",
        "",
        "Timing includes host-cache gathering, transfers, selection, all-arrival reference features and synchronized SSL training. It excludes acquisition, decoding, checkpoint IO and downstream fitting. CPU timing is not pooled.",
        "",
        "Archived reader attributes were matched to saved labels and source pixels. Original local test-label JSON files are absent on this machine; provider and capture-session provenance remain unverified. Sealed final queries were not scored.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = Figure(figsize=(10, 4), layout="constrained")
    axes = fig.subplots(1, 2)
    names = list(ARMS[1:])
    axes[0].barh(names, [contrasts[a + "_minus_frozen"]["mean_pp"]["finetune"]["balanced"] for a in names])
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set(xlabel="Terminal fine-tuning gain over frozen (pp)", title="Reused development bundles")
    axes[1].barh(names, [100 * timing[a]["summed_time_saving_fraction"] for a in names])
    axes[1].set(xlabel="Charged processing saving versus full (%)", title="RTX 2080; float32")
    fig.savefig(out / "results.png", dpi=150)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "completed_arms": 0, "total_arms": 20}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA required for this authorized run")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        progress(stage="locking_inputs")
        config, cohorts, data, lock = load_inputs(out)
        torch.manual_seed(11)
        initial = _method(config).to("cuda")
        historical_audit(initial, config, out, progress)
        episodes = [
            torch.from_numpy(np.concatenate([np.load(c["local_frames_path"], allow_pickle=False) for c in group]))
            .permute(0, 3, 1, 2)
            .float()
            / 255
            for group in cohorts
        ]
        progress(stage="cuda_preflight")
        if not (out / "preflight.json").exists():
            preflight(initial, config, episodes[0], data, out)
        if not json.loads((out / "preflight.json").read_text())["passed"]:
            raise ValueError("Preflight failed")
        records = []
        reference = copy.deepcopy(initial.encoder).eval().requires_grad_(False)
        reference_marker = fingerprint(reference.state_dict())
        for bundle in (0, 1):
            for seed_index, seed in enumerate(SEEDS):
                torch.manual_seed(seed)
                optimizer = instantiate(config.optim, params=[p for p in initial.parameters() if p.requires_grad])
                parent = capture(initial, optimizer)
                parent_marker = fingerprint(parent)
                # Rotate execution order independently of outcomes to expose timing drift.
                shift = (bundle * 2 + seed_index) % len(ARMS)
                order = ARMS[shift:] + ARMS[:shift]
                for arm in order:
                    directory = out / "runs" / f"bundle_{bundle}" / f"seed_{seed}" / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    result_path = directory / "result.json"
                    if result_path.exists():
                        records.append(json.loads(result_path.read_text()))
                        continue
                    progress(stage="training", bundle=bundle, seed=seed, arm=arm)
                    model, optim = restore(initial, config.optim, parent)
                    if fingerprint(capture(model, optim)) != parent_marker:
                        raise ValueError("Branch initial state differs")
                    terminal_path = directory / "terminal.pt"
                    trace_path = directory / "training.json"
                    if terminal_path.exists() and trace_path.exists():
                        state = torch.load(terminal_path, map_location="cuda", weights_only=True)
                        model, optim = restore(initial, config.optim, state["learner"])
                        training = json.loads(trace_path.read_text())
                    else:
                        training = run_episode(model, optim, episodes[bundle], arm, reference, seed)
                        selector_rng = training.pop("selector_rng")
                        state = {
                            "learner": capture(model, optim),
                            "selector_rng": selector_rng,
                            "buffer": [],
                            "arrivals": 1536,
                            "updates": training["updates"],
                            "accepted_frames": training["accepted_frames"],
                        }
                        torch.save(state, terminal_path)
                        atomic_json(trace_path, training)
                    expected = 0 if arm == "frozen" else 48 if arm == "full" else 24
                    if training["updates"] != expected or len(episodes[bundle]) != 1536:
                        raise ValueError("Frozen update budget differs")
                    marker = fingerprint(state)
                    progress(stage="readout", bundle=bundle, seed=seed, arm=arm)
                    outcome = evaluate_gpu(model, optim, data, directory)
                    if (
                        fingerprint(state) != marker
                        or fingerprint(parent) != parent_marker
                        or fingerprint(reference.state_dict()) != reference_marker
                    ):
                        raise ValueError("Branch isolation failed")
                    result = {
                        "bundle": bundle,
                        "seed": seed,
                        "arm": arm,
                        **outcome,
                        "updates": training["updates"],
                        "accepted_frames": training["accepted_frames"],
                        "online_seconds": training["online_seconds"],
                        "initial_fingerprint": parent_marker,
                        "terminal_fingerprint": marker,
                        "relative_directory": str(directory.relative_to(out)),
                    }
                    atomic_json(result_path, result)
                    records.append(result)
                    progress(completed_arms=len(records))
                    del model, optim
        progress(stage="independent_readout_audit")
        if any(digest(Path(path)) != value for path, value in lock["hashes"].items()):
            raise ValueError("Locked source/input changed during collection")
        audit_new_trajectories(initial, config, episodes, records, out, progress)
        summary = report(out, records, data)
        if summary["scientific_ssl_updates"] != 480 or summary["primary_fine_tuning_fits"] != 60:
            raise ValueError("Collection budget differs")
        progress(
            stage="complete",
            completed_arms=len(records),
            audited_arms=len(records),
            scientific_ssl_updates=480,
            primary_fine_tuning_fits=60,
            decision=summary["decision"],
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
