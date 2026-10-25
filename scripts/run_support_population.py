"""Replay a fixed development panel and cache features for disjoint support-population comparisons."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from cafl4ds.action_identification import action_at
from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import BDD100KSource, _driving_attributes, _image_filename, _iter_bdd_records
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.image_utility import embed
from cafl4ds.reader_reliability import TRAINING_REPEATS, read_cached, support_draws
from cafl4ds.response_calibration import CALIBRATION_SCHEDULES
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.support_population import disjoint_panels
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_alpha_response import memory_signature
from scripts.run_image_utility_experiment import identical, restore_bdd_inputs
from scripts.run_slow_edge_pilot import ROOT, optimizer_for, update


def isolated_embed(method: SSLMethod, images: torch.Tensor) -> torch.Tensor:
    """Verify cached observations preserve learning parameters, modes, gradients and RNG."""
    state = copy.deepcopy(method.state_dict())
    modes = [m.training for m in method.modules()]
    gradients = [None if p.grad is None else p.grad.clone() for p in method.parameters()]
    rng = torch.get_rng_state().clone()
    z = embed(method, images)
    if not identical(state, method.state_dict()) or modes != [m.training for m in method.modules()]:
        raise ValueError("Observation changed method")
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("Observation changed training RNG")
    for gradient, parameter in zip(gradients, method.parameters(), strict=True):
        if not identical(gradient, parameter.grad):
            raise ValueError("Observation changed gradients")
    return z


def main() -> None:  # noqa: C901 - explicit restored paired replay and cache verification
    """Do not read or replay former final-test outcomes; preserve every selected branch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed response-calibration directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    atomic_json(output / "progress.json", {"state": "loading"})
    previous = json.loads((source / "inputs.json").read_text())
    parent = Path(previous["source"])
    original = json.loads((parent / "inputs.json").read_text())
    warm_path = ROOT / "outputs/image-utility/20261006-bdd-alpha-response/inputs.json"
    warm_manifest = json.loads(warm_path.read_text())
    if original["warm_paths"] != warm_manifest["warm_paths"]:
        raise ValueError("Warm identity changed")
    streams = [s for s in original["streams"] if s["split"] in ("fit", "validation")]
    histories, repeats = list(original["histories"]), list(TRAINING_REPEATS)
    if args.smoke:
        streams, histories, repeats = streams[:1], histories[:1], repeats[:1]
    planned = len(streams) * len(histories) * len(repeats) * 4
    step = 8
    reference = ROOT / "outputs/image-utility/20261007-bdd-reader-reliability"
    capacity = ROOT / "outputs/image-utility/20261007-bdd-support-capacity"
    reference_manifest = json.loads((reference / "cache_manifest.json").read_text())
    reference_hashes = {str(reference / n): h for n, h in reference_manifest["files"].items()}
    for path, expected in reference_hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Reference cache changed: {path}")
    source_data = {k: v for k, v in original["config"]["data"].items() if not k.startswith("_")}
    source_data.update(split="val", partition_role=None)
    bdd = BDD100KSource(**source_data)
    images_dir, labels_path = bdd._paths()
    domain_names = original["evaluation"]["domain_names"]
    class_names = [original["evaluation"]["class_names"][str(i)] for i in range(3)]
    cells: dict[tuple[int, int], list[str]] = {(d, c): [] for d in (0, 1) for c in range(3)}
    for record in _iter_bdd_records(labels_path):
        attrs = _driving_attributes(record)
        if attrs is None:
            continue
        domain_name = f"{attrs[0]}·{attrs[1]}"
        path = str(images_dir / _image_filename(record))
        if domain_name in domain_names and attrs[2] in class_names and Path(path).is_file():
            cells[(domain_names.index(domain_name), class_names.index(attrs[2]))].append(path)
    panels = disjoint_panels(cells, set(original["validation_paths"]))
    selected_paths = panels[0] + panels[1]
    resize = transforms.Compose(
        [transforms.Resize((source_data["img_size"], source_data["img_size"])), transforms.ToTensor()]
    )
    decoded = []
    for path in selected_paths:
        with Image.open(path) as image:
            decoded.append(resize(image.convert("RGB")))
    support_images = torch.stack(decoded)
    old_capacity = json.loads((capacity / "inputs.json").read_text())
    older_overlap = {}
    historical_manifests = []
    for manifest_path in sorted((ROOT / "outputs/image-utility").glob("*/inputs.json")):
        prior = json.loads(manifest_path.read_text())
        validation_paths = prior.get("validation_paths", [])
        if validation_paths:
            historical_manifests.append(manifest_path)
            older_overlap[str(manifest_path)] = len(set(selected_paths) & set(validation_paths))
    input_paths = [
        source / "inputs.json",
        parent / "inputs.json",
        warm_path,
        reference / "cache_manifest.json",
        capacity / "inputs.json",
        capacity / "reader_scores.csv",
        *(sorted(labels_path.glob("*.json")) if labels_path.is_dir() else [labels_path]),
        *historical_manifests,
        *map(Path, selected_paths),
    ]
    for stream in streams:
        for history in histories:
            input_paths.append(parent / f"snapshot_{stream['stream']}_{history}_{step}.pt")
            input_paths.extend(
                source / f"branch_{stream['stream']}_{history}_{step}_{r}_{s}.json"
                for r in repeats
                for s in CALIBRATION_SCHEDULES
            )
    input_hashes = {str(p): digest(p) for p in input_paths}
    scientific_paths = [
        *sorted((ROOT / "cafl4ds").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
        ROOT / "docs/project-plan/slow-edge-support-population.md",
        ROOT / "notebooks/16_support_population.ipynb",
    ]
    source_hashes = {str(p): digest(p) for p in scientific_paths}
    atomic_json(
        output / "inputs.json",
        {
            "source": str(source),
            "reference_cache": str(reference),
            "capacity_reference": str(capacity),
            "reference_hashes": reference_hashes,
            "support_paths": panels,
            "support_images_sha256": tensor_hash(support_images),
            "older_validation_overlap": older_overlap,
            "support_cells_available": {
                f"{d}_{c}": len(set(paths) - set(original["validation_paths"])) for (d, c), paths in cells.items()
            },
            "parent": str(parent),
            "input_hashes": input_hashes,
            "source_hashes": source_hashes,
            "streams": streams,
            "histories": histories,
            "step": step,
            "repeats": repeats,
            "schedules": CALIBRATION_SCHEDULES,
            "planned_branches": planned,
            "smoke": args.smoke,
            "limits": "Fixed previously observed development states; no new qualification",
        },
    )
    records: list[dict[str, Any]] = []
    cache_hashes: dict[str, str] = {}
    maximum_error = 0.0
    try:
        for stream in streams:
            manifest = {
                **original,
                "train_paths": stream["paths"],
                "train_source_order": stream["source_order"],
                "arrivals_sha256": stream["images_sha256"],
                "warm_sha256": warm_manifest["warm_sha256"],
            }
            torch.manual_seed(62000)
            initial, warm, arrivals, evaluation, _ = restore_bdd_inputs(manifest)
            del warm
            roles = {name: torch.tensor(indices) for name, indices in original["roles"].items()}
            if not torch.equal(torch.cat(list(roles.values())).sort().values, evaluation.query.sort().values):
                raise ValueError("Roles must partition reserved queries")
            supports = support_draws(
                evaluation.labels, evaluation.domains, evaluation.query, next(iter(evaluation.supports.values()))
            )
            metadata: dict[str, Any] = {
                "labels": torch.cat(
                    [
                        evaluation.labels,
                        torch.tensor([c for _ in range(2) for d in (0, 1) for c in range(3) for _ in range(44)]),
                    ]
                ),
                "domains": torch.cat(
                    [
                        evaluation.domains,
                        torch.tensor([d for _ in range(2) for d in (0, 1) for c in range(3) for _ in range(44)]),
                    ]
                ),
                "query": evaluation.query,
                "roles": roles,
                "supports": {
                    "old264": torch.tensor(old_capacity["support264"]),
                    "new1": torch.arange(len(evaluation.labels), len(evaluation.labels) + 264),
                    "new2": torch.arange(len(evaluation.labels) + 264, len(evaluation.labels) + 528),
                },
            }
            if (output / "evaluation.pt").exists():
                if not identical(metadata, torch.load(output / "evaluation.pt", weights_only=False)):
                    raise ValueError("Evaluation identities changed across streams")
            else:
                torch.save(metadata, output / "evaluation.pt")
                cache_hashes["evaluation.pt"] = digest(output / "evaluation.pt")
                atomic_json(
                    output / "support_indices.json",
                    {name: indices.tolist() for name, indices in metadata["supports"].items()},
                )
            for history in histories:
                snapshot = torch.load(parent / f"snapshot_{stream['stream']}_{history}_{step}.pt", weights_only=False)
                common = copy.deepcopy(initial)
                common.load_state_dict(snapshot["method"])
                common.train()
                selector_parent = SteerableReplay(
                    initial.encoder,
                    torch.tensor(original["centroids"]),
                    torch.tensor(original["target"]),
                    seed=stream["seed"],
                )
                selector_parent.load_state_dict(snapshot["actuator"])
                pre_name = f"pre_{stream['stream']}_{history}_{step}.pt"
                old_pre = isolated_embed(common, evaluation.images)
                if not torch.equal(old_pre, torch.load(reference / pre_name, weights_only=False)):
                    raise ValueError("Reference parent features changed")
                torch.save(torch.cat([old_pre, isolated_embed(common, support_images)]), output / pre_name)
                torch.save(common.state_dict(), output / f"model_{pre_name}")
                cache_hashes[f"model_{pre_name}"] = digest(output / f"model_{pre_name}")
                cache_hashes[pre_name] = digest(output / pre_name)
                for repeat in repeats:
                    paired_memory = None
                    for schedule in CALIBRATION_SCHEDULES:
                        recorded_path = source / f"branch_{stream['stream']}_{history}_{step}_{repeat}_{schedule}.json"
                        recorded = json.loads(recorded_path.read_text())
                        if recorded["split"] == "test":
                            raise ValueError("Final-test reuse prohibited")
                        trial = copy.deepcopy(common)
                        optimizer = optimizer_for(trial)
                        optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
                        selector = copy.deepcopy(selector_parent)
                        torch.set_rng_state(snapshot["rng"])
                        if repeat:
                            torch.manual_seed(stream["seed"] * 10000 + step * 100 + repeat)
                        decisions, memory = [], []
                        for offset in range(12):
                            selector.set_action(action_at(schedule, offset))
                            audit = update(
                                trial, optimizer, arrivals[offset * 32 : (offset + 1) * 32], step + offset, selector
                            )
                            if selector.last["current_count"] != 16 or selector.last["replay_count"] != 16:
                                raise ValueError("Unequal training work")
                            decisions.append({**copy.deepcopy(selector.last), **audit})
                            memory.append(memory_signature(selector))
                        if decisions != recorded["decisions"] or memory != recorded["memory"]:
                            raise ValueError("Existing branch decisions/memory changed")
                        if paired_memory is None:
                            paired_memory = memory
                        elif memory != paired_memory:
                            raise ValueError("Unmatched paired memory/private RNG")
                        if selector.optimizer_images - selector_parent.optimizer_images != 384:
                            raise ValueError("Unequal optimizer image count")
                        z = isolated_embed(trial, evaluation.images)
                        read = read_cached(z, evaluation.labels, evaluation.domains, roles, supports["original36"])
                        for role, metrics in read.items():
                            for metric, value in metrics.items():
                                error = abs(value - recorded["outcomes"]["12"][role][metric])
                                if error > (1e-10 if metric.startswith("refit_ce") else 0):
                                    raise ValueError(f"Cached original reader changed: {role}/{metric}: {error}")
                                maximum_error = max(maximum_error, error)
                        name = f"post_{stream['stream']}_{history}_{step}_{repeat}_{schedule}.pt"
                        if not torch.equal(z, torch.load(reference / name, weights_only=False)):
                            raise ValueError("Reference post-branch features changed")
                        torch.save(torch.cat([z, isolated_embed(trial, support_images)]), output / name)
                        torch.save(trial.state_dict(), output / f"model_{name}")
                        cache_hashes[f"model_{name}"] = digest(output / f"model_{name}")
                        cache_hashes[name] = digest(output / name)
                        records.append(
                            {
                                "stream": stream["stream"],
                                "split": stream["split"],
                                "history": history,
                                "step": step,
                                "repeat": repeat,
                                "schedule": schedule,
                                "pre_cache": pre_name,
                                "post_cache": name,
                                "original_read": read,
                                "decisions": decisions,
                                "memory": memory,
                            }
                        )
                        atomic_json(
                            output / "progress.json",
                            {
                                "state": "collecting",
                                "branches": len(records),
                                "planned_branches": planned,
                                "stream": stream["stream"],
                                "elapsed_seconds": time.time() - started,
                            },
                        )
                        print(f"cached verified branches: {len(records)}/{planned}", flush=True)
                if not identical(common.state_dict(), snapshot["method"]) or not identical(
                    selector_parent.state_dict(), snapshot["actuator"]
                ):
                    raise ValueError("Saved parent changed")
        for path, expected in {**input_hashes, **source_hashes, **reference_hashes}.items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Source/input changed: {path}")
        atomic_json(output / "cache_manifest.json", {"files": cache_hashes, "rows": records})
        atomic_json(
            output / "progress.json",
            {
                "state": "completed",
                "branches": len(records),
                "executed_updates": len(records) * 12,
                "maximum_original_reader_error": maximum_error,
                "observation_isolation_verified": True,
                "matched_memory_every_update": True,
                "parent_unchanged": True,
                "elapsed_seconds": time.time() - started,
            },
        )
    except Exception as error:
        atomic_json(output / "progress.json", {"state": "failed", "branches": len(records), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
