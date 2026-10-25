"""Collect larger development readers, supervised controls and isolated alpha gradients."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import BDD100KSource, _driving_attributes
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.fresh_alpha import action_observations, allocate_roles, support_folds
from cafl4ds.image_utility import balanced_losses
from cafl4ds.measurement_controls import shuffle_within_domain, train_control
from cafl4ds.reader_calibration import C_GRID, choose_c, fit_probe
from cafl4ds.support_capacity import capacity_outcomes
from scripts.audit_fresh_alpha_data import image_paths
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_alpha_response import memory_signature
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import ROOT, optimizer_for, update
from scripts.run_support_population import isolated_embed


def prepare(audit: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:  # noqa: C901
    """Reserve unused queries before permitting declared historical stream exposure elsewhere."""
    generator = torch.Generator().manual_seed(77011)
    queries: dict[str, list[str]] = {"development": [], "holdout": []}
    tags: dict[str, tuple[int, int, str]] = {}
    for split, cap in (("val", 128), ("train", 64)):
        for key, cell in sorted(audit["pools"][split].items()):
            paths = cell["unused_paths"]
            order = torch.randperm(len(paths), generator=generator).tolist()[:cap]
            selected = [paths[i] for i in order]
            domain, label = map(int, key.split("_"))
            for i, path in enumerate(selected):
                queries["development" if i % 2 == 0 else "holdout"].append(path)
                tags[str(Path(path).resolve()).casefold()] = (domain, label, split)
    query_paths = set(tags)
    candidates = {
        str(Path(path).resolve()).casefold()
        for cell in audit["pools"]["train"].values()
        for path in cell["unused_paths"]
    }
    warm: set[str] = set()
    eligible = []
    for name, record in audit["historical_manifests"].items():
        path = Path(name)
        if digest(path) != record["sha256"]:
            raise ValueError(f"Historical allocation changed: {path}")
        if path.name != "inputs.json":
            continue
        prior = json.loads(path.read_text())
        warm.update(image_paths(prior.get("warm_paths", [])))
        config = prior.get("config", {}).get("data", {})
        current = audit["config"]["data"]
        if (
            config.get("partition_role") == "stream"
            and config.get("partition_seed") == current["partition_seed"]
            and config.get("warm_fraction") == current["warm_fraction"]
        ):
            candidates.update(image_paths(prior.get("train_paths", [])))
            candidates.update(image_paths([stream.get("paths", []) for stream in prior.get("streams", [])]))
            eligible.append(name)
    source = BDD100KSource(**{k: v for k, v in audit["config"]["data"].items() if not k.startswith("_")})
    image_dir, label_dir = source._paths()
    cells: dict[str, dict[str, Any]] = {f"{d}_{c}": {"unused_paths": []} for d in range(2) for c in range(3)}
    for name in sorted(candidates - warm - query_paths):
        path = Path(name)
        if path.parent.resolve() != image_dir.resolve():
            continue
        record = json.loads((label_dir / f"{path.stem}.json").read_text(encoding="utf-8"))
        attrs = _driving_attributes(record)
        if attrs is None:
            continue
        domain_name = "·".join(attrs[:2])
        if domain_name in audit["domain_names"] and attrs[2] in audit["class_names"]:
            d, c = audit["domain_names"].index(domain_name), audit["class_names"].index(attrs[2])
            cells[f"{d}_{c}"]["unused_paths"].append(str(path))
            tags[name] = (d, c, "train")
    counts = {
        "fitting": 96,
        "internal_validation": 32,
        "support1": 64,
        "support2": 64,
        "support3": 64,
        "monitor": 16,
        "stream1": 128,
        "stream2": 128,
        "stream3": 128,
    }
    roles = {**allocate_roles(cells, counts, 77013), **queries}
    selected = [path for paths in roles.values() for path in paths]
    if len({str(Path(path).resolve()).casefold() for path in selected}) != len(selected):
        raise ValueError("Cross-role image overlap")
    recorded = set().union(*(image_paths(json.loads(Path(name).read_text())) for name in audit["historical_manifests"]))
    overlap = {
        role: sum(str(Path(path).resolve()).casefold() in recorded for path in paths) for role, paths in roles.items()
    }
    if overlap["development"] or overlap["holdout"]:
        raise ValueError("New query images appeared in recorded inputs")
    return roles, {
        "tags": tags,
        "historical_overlap": overlap,
        "eligible_manifests": eligible,
        "allocation_counts_per_cell": counts,
        "available_nonquery_cells": {key: len(cell["unused_paths"]) for key, cell in cells.items()},
    }


def decode(
    roles: dict[str, list[str]], tags: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Decode fixed role lists; attribute labels enter readouts and controls only."""
    resize = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    images: list[torch.Tensor] = []
    labels: list[int] = []
    domains: list[int] = []
    indices: dict[str, torch.Tensor] = {}
    for role, paths in roles.items():
        indices[role] = torch.arange(len(images), len(images) + len(paths))
        for path in paths:
            with Image.open(path) as image:
                images.append(resize(image.convert("RGB")))
            domain, label, _ = tags[str(Path(path).resolve()).casefold()]
            labels.append(label)
            domains.append(domain)
    return torch.stack(images), torch.tensor(labels), torch.tensor(domains), indices


def main() -> None:  # noqa: C901 - explicit measurement and branch invariance audit
    """Complete the frozen development study without fitting a response predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(62000)
    started = time.time()
    atomic_json(output / "progress.json", {"state": "allocation"})
    audit = json.loads(args.audit.read_text())
    roles, allocation = prepare(audit)
    atomic_json(output / "roles.json", {"paths": roles, **allocation})
    scientific = [
        *ROOT.glob("cafl4ds/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        ROOT / "docs/project-plan/slow-edge-fresh-alpha.md",
        ROOT / "notebooks/20_fresh_alpha.ipynb",
    ]
    hashes = {str(path): digest(path) for path in scientific}
    image_hashes = {path: digest(Path(path)) for paths in roles.values() for path in paths}
    config = OmegaConf.create(audit["config"])
    initial = _method(config)
    inputs = {
        "audit_sha256": digest(args.audit),
        "well_sha256": digest(Path(config.well)),
        "source_hashes": hashes,
        "image_hashes": image_hashes,
        "smoke": args.smoke,
        "limits": "Development; historical SSL image reuse and scarce rare-cell fresh queries",
    }
    atomic_json(output / "inputs.json", inputs)
    images, labels, domains, indices = decode(roles, allocation["tags"])
    query_roles = {name: indices[name] for name in ("development", "holdout")}
    panels = {name: indices[name] for name in ("support1", "support2", "support3")}
    panels["pooled"] = torch.cat(list(panels.values()))
    torch.save({"labels": labels, "domains": domains, "indices": indices}, output / "evaluation.pt")
    warm_z = isolated_embed(initial, images)
    torch.save(warm_z, output / "warm_features.pt")
    cv = []
    cs = {}
    for panel, support in panels.items():
        values: dict[float, list[float]] = {c: [] for c in C_GRID}
        for fold, (fit, validation) in enumerate(support_folds(labels, domains, support, 77017)):
            for c in C_GRID:
                probe = fit_probe(warm_z, labels, fit, c)
                ce = balanced_losses(
                    probe.losses(warm_z[validation], labels[validation]), labels[validation], domains[validation]
                )["balanced"]
                values[c].append(ce)
                cv.append({"panel": panel, "fold": fold, "C": c, "ce": ce})
        cs[panel] = choose_c({c: sum(v) / len(v) for c, v in values.items()})
    pd.DataFrame(cv).to_csv(output / "support_cv.csv", index=False)
    atomic_json(output / "reader_frozen.json", {"selected_C": cs, "before_query_scoring": True})
    scores: list[dict[str, Any]] = []
    readout_indices = torch.cat([panels["pooled"], *query_roles.values()])
    lookup = torch.full((len(images),), -1, dtype=torch.long)
    lookup[readout_indices] = torch.arange(len(readout_indices))
    reader_panels = {name: lookup[indices] for name, indices in panels.items()}
    reader_roles = {name: lookup[indices] for name, indices in query_roles.items()}
    torch.save(
        {
            "indices": readout_indices,
            "labels": labels[readout_indices],
            "domains": domains[readout_indices],
            "supports": reader_panels,
            "roles": reader_roles,
        },
        output / "readout_metadata.pt",
    )

    def score(z: torch.Tensor, kind: str, state: str, **metadata: object) -> None:
        for panel, support in reader_panels.items():
            result = capacity_outcomes(
                z, labels[readout_indices], domains[readout_indices], support, reader_roles, cs[panel]
            )
            for role, metrics in result.items():
                scores.append({"kind": kind, "state": state, "panel": panel, "role": role, **metadata, **metrics})
        pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)

    score(warm_z[readout_indices], "warm", "warm")
    fitting, validation = indices["fitting"], indices["internal_validation"]
    traces: list[dict[str, Any]] = []
    seeds = [77021] if args.smoke else [77021, 77027]
    epochs = 1 if args.smoke else 20
    for seed in seeds:
        for shuffled in (False, True):
            name = f"{'shuffled' if shuffled else 'real'}_{seed}"
            atomic_json(output / "progress.json", {"state": "supervised_controls", "arm": name})
            y = shuffle_within_domain(labels[fitting], domains[fitting], 77019) if shuffled else labels[fitting]
            encoder, head, trace = train_control(initial.encoder, images[fitting], y, seed, epochs=epochs)
            trial = copy.deepcopy(initial)
            trial.encoder.load_state_dict(encoder.state_dict())
            z = isolated_embed(trial, images)
            torch.save(z, output / f"features_{name}.pt")
            torch.save({"method": trial.state_dict(), "head": head.state_dict()}, output / f"control_{name}.pt")
            with torch.no_grad():
                logits = head(z[validation])
                internal = {
                    "ce": float(torch.nn.functional.cross_entropy(logits, labels[validation])),
                    "accuracy": float((logits.argmax(1) == labels[validation]).float().mean()),
                }
            traces.extend({"arm": name, **row} for row in trace)
            atomic_json(output / f"control_{name}.json", {"internal_validation": internal, "trace": trace})
            score(z[readout_indices], "control", name, seed=seed, shuffled=shuffled)
    pd.DataFrame(traces).to_csv(output / "control_traces.csv", index=False)
    head_probe = fit_probe(warm_z, labels, fitting, 0.1)
    atomic_json(
        output / "head_only_internal.json",
        balanced_losses(
            head_probe.losses(warm_z[validation], labels[validation]), labels[validation], domains[validation]
        ),
    )
    # Fixed warm features provide label-free coordinates and health reference.
    original = json.loads((ROOT / "outputs/image-utility/20261006-bdd-alpha-response/inputs.json").read_text())
    centers, target = torch.tensor(original["centroids"]), torch.tensor(original["target"])
    monitor, warm_monitor = images[indices["monitor"]], warm_z[indices["monitor"]]
    observations: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    stream_names = ["stream1"] if args.smoke else ["stream1", "stream2", "stream3"]
    learning_seeds = [77041] if args.smoke else [77041, 77047]
    prefixes = [4] if args.smoke else [4, 12]
    repeats = [0] if args.smoke else [0, 1]
    for stream_name in stream_names:
        pool = indices[stream_name]
        generator = torch.Generator().manual_seed(77051 + int(stream_name[-1]))
        visits: list[list[torch.Tensor]] = [[], [], [], []]
        for d in (0, 1):
            for c in (0, 1, 2):
                cell = pool[(domains[pool] == d) & (labels[pool] == c)]
                cell = cell[torch.randperm(len(cell), generator=generator)]
                visits[d].append(cell[:64])
                visits[d + 2].append(cell[64:])
        ordered = []
        for visit in visits:
            joined = torch.cat(visit)
            ordered.append(joined[torch.randperm(len(joined), generator=generator)])
        arrival_indices = torch.cat(ordered)
        atomic_json(output / f"order_{stream_name}.json", {"indices": arrival_indices.tolist()})
        arrivals = images[arrival_indices]
        for seed in learning_seeds:
            torch.manual_seed(seed)
            common = copy.deepcopy(initial)
            opt = optimizer_for(common)
            selector = SteerableReplay(initial.encoder, centers, target, seed=seed)
            for step in range(max(prefixes) + 1):
                batch = arrivals[step * 32 : (step + 1) * 32]
                if step in prefixes:
                    name = f"{stream_name}_{seed}_{step}"
                    snapshot: dict[str, Any] = {
                        "method": copy.deepcopy(common.state_dict()),
                        "optimizer": copy.deepcopy(opt.state_dict()),
                        "actuator": selector.state_dict(),
                        "rng": torch.get_rng_state().clone(),
                    }
                    torch.save(snapshot, output / f"parent_{name}.pt")
                    gradients = [None if p.grad is None else p.grad.clone() for p in common.parameters()]
                    modes = [module.training for module in common.modules()]
                    _, actions = action_observations(common, selector, batch, step, monitor, warm_monitor)
                    if (
                        not identical(common.state_dict(), snapshot["method"])
                        or not identical(selector.state_dict(), snapshot["actuator"])
                        or not identical(opt.state_dict(), snapshot["optimizer"])
                        or not identical(gradients, [p.grad for p in common.parameters()])
                        or modes != [module.training for module in common.modules()]
                        or not torch.equal(snapshot["rng"], torch.get_rng_state())
                    ):
                        raise ValueError("Predecision observations changed actual learning")
                    observations.append(
                        {"state": name, "stream": stream_name, "seed": seed, "step": step, "actions": actions}
                    )
                    atomic_json(output / "observations.json", {"states": observations})
                    for repeat in repeats:
                        memory = None
                        for alpha in (0.0, 0.5, 1.0):
                            atomic_json(
                                output / "progress.json",
                                {
                                    "state": "alpha_branches",
                                    "parent": name,
                                    "repeat": repeat,
                                    "alpha": alpha,
                                    "completed_branches": len(branches),
                                },
                            )
                            trial, private = copy.deepcopy(common), copy.deepcopy(selector)
                            private.set_action(alpha)
                            optimizer = optimizer_for(trial)
                            optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
                            torch.set_rng_state(snapshot["rng"])
                            if repeat:
                                torch.manual_seed(seed * 10000 + step * 100 + repeat)
                            paired, decisions = [], []
                            for offset in range(12):
                                actual = step + offset
                                record = update(
                                    trial, optimizer, arrivals[actual * 32 : (actual + 1) * 32], actual, private
                                )
                                paired.append(memory_signature(private))
                                decisions.append({**record, **private.last})
                                if private.last["current_count"] != 16 or private.last["replay_count"] != 16:
                                    raise ValueError("Unequal branch batch composition")
                                if offset + 1 in (4, 12):
                                    z = isolated_embed(trial, images[readout_indices])
                                    score(
                                        z,
                                        "branch",
                                        name,
                                        stream=stream_name,
                                        seed=seed,
                                        step=step,
                                        repeat=repeat,
                                        alpha=alpha,
                                        horizon=offset + 1,
                                    )
                                    torch.save(z, output / f"features_{name}_{repeat}_{alpha}_{offset + 1}.pt")
                            if private.optimizer_images - selector.optimizer_images != 384:
                                raise ValueError("Unequal optimizer work")
                            if memory is not None and memory != paired:
                                raise ValueError("Action-dependent reservoir or private replay RNG")
                            memory = paired
                            torch.save(
                                {"method": trial.state_dict(), "actuator": private.state_dict()},
                                output / f"branch_{name}_{repeat}_{alpha}.pt",
                            )
                            branches.append({"state": name, "repeat": repeat, "alpha": alpha, "decisions": decisions})
                            atomic_json(output / "branches.json", {"branches": branches})
                    torch.set_rng_state(snapshot["rng"])
                if step < max(prefixes):
                    update(common, opt, batch, step, selector)
    verified = {
        **hashes,
        **image_hashes,
        str(args.audit.resolve()): inputs["audit_sha256"],
        str(Path(config.well)): inputs["well_sha256"],
    }
    for path, expected in verified.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Source/input drift: {path}")
    artifacts = {
        str(path.relative_to(output)): digest(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "progress.json"
    }
    atomic_json(
        output / "verification.json",
        {
            "all_inputs_unchanged": True,
            "checked_inputs": len(verified),
            "artifact_hashes": artifacts,
            "parents": len(observations),
            "branches": len(branches),
            "branch_updates": len(branches) * 12,
            "elapsed_seconds": time.time() - started,
        },
    )
    atomic_json(output / "progress.json", {"state": "completed", "branches": len(branches)})
    print(f"Completed {len(observations)} parents, {len(branches)} branches: {output}")


if __name__ == "__main__":
    main()
