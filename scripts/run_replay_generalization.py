"""Frozen health-gradient replay models on new order and stochastic conditions."""

# ruff: noqa: E501
# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.streams import StreamBatch
from cafl4ds.domain_replay_health import domain_sensors, sensor
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.image_utility import reconstruction_draws
from cafl4ds.reader_calibration import probe_outcomes
from cafl4ds.replay_local import gradient_signals
from cafl4ds.replay_response import ResponseModel, choose
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_fresh_alpha import decode
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import ROOT, optimizer_for, update
from scripts.run_support_population import isolated_embed

RECIPES = ("composition", "reconstruction", "health_gradients")
CONDITIONS = ("shuffled_incoming", "interleaved")


def response(row: dict[str, Any]) -> ResponseModel:
    """Restore saved parameters without fitting on replication labels."""
    return ResponseModel(
        tuple(row["fields"]),
        np.array(row["mean"]),
        np.array(row["scale"]),
        np.array(row["weight"]),
        row["regularization"],
    )


def new_order(order: list[int], domains: torch.Tensor, condition: str, seed: int) -> list[int]:
    """Preserve history and sample a novel unique future from remaining images."""
    prefix, remaining = order[:320], torch.tensor(order[320:])
    incoming = int(domains[order[320]])
    generator = torch.Generator().manual_seed(seed)
    groups = {}
    for d in (0, 1):
        group = remaining[domains[remaining] == d]
        groups[d] = group[torch.randperm(len(group), generator=generator)]
    if condition == "shuffled_incoming":
        future = groups[incoming][:192].tolist()
    elif condition == "interleaved":
        future = []
        used_counts = {0: 0, 1: 0}
        for d in (incoming, 1 - incoming, incoming, 1 - incoming, incoming, incoming):
            start = used_counts[d]
            future.extend(groups[d][start : start + 32].tolist())
            used_counts[d] += 32
    else:
        raise ValueError(condition)
    used = set(future)
    result = prefix + future + [i for i in order[320:] if i not in used]
    if len(future) != 192 or len(result) != 768 or len(set(result)) != 768 or set(result) != set(order):
        raise ValueError("Invalid arrival permutation")
    return result


def main() -> None:  # noqa: C901
    """Save every prediction before collecting all matched action outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run = parser.parse_args().output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = ROOT / "outputs/image-utility/20261008-replay-controller"
    local = ROOT / "outputs/image-utility/20261008-replay-local-v2"
    parents = ROOT / "outputs/image-utility/20261008-new-replay-parents"
    input_hashes: dict[str, str] = {}

    def verified(folder: Path, name: str, manifest: str = "verification.json") -> Path:
        path = folder / name
        expected = json.loads((folder / manifest).read_text())["artifact_hashes"][name]
        if digest(path) != expected:
            raise ValueError(f"Changed input {path}")
        input_hashes[str(path)] = expected
        return path

    roles = json.loads(verified(parents, "roles.json").read_text())
    metadata = (
        torch.load(verified(source, "readout_metadata.pt"), weights_only=False)
        if (source / "readout_metadata.pt").exists()
        else torch.load(verified(local, "readout_metadata.pt"), weights_only=False)
    )
    reader = json.loads(verified(local, "reader_frozen.json").read_text())
    old_features = json.loads(verified(source, "predecision_features.json").read_text())["rows"]
    bases = {r["state"]: r for r in old_features}
    snapshots = {s: torch.load(verified(local, f"parent_{s}.pt"), weights_only=False) for s in bases}
    orders = {s: json.loads(verified(source, f"order_{s}.json").read_text())["indices"] for s in bases}
    raw_models = json.loads(verified(source, "response_models.json", "policy_verification.json").read_text())["models"]
    raw_models = [
        m
        for m in raw_models
        if m["horizon"] == 6 and m["bag"] == 0 and m["target"] == "linear_error_balanced" and m["recipe"] in RECIPES
    ]
    models = {(m["heldout_pool"], m["recipe"]): response(m) for m in raw_models}
    old_choices = pd.read_csv(verified(source, "policy_choices.csv", "policy_verification.json"))
    constants = old_choices[(old_choices.horizon == 6)].groupby("state").accuracy_constant.first().to_dict()
    for model in raw_models:
        if any(bases[s]["pool_group"] == model["heldout_pool"] for s in model["fit_states"]):
            raise ValueError("Frozen model trained on its evaluation pool")
    original = json.loads(verified(parents, "inputs.json").read_text())
    audit = ROOT / "outputs/image-utility/20261007-fresh-alpha-allocation.json"
    if digest(audit) != original["audit_sha256"]:
        raise ValueError("Config changed")
    config = OmegaConf.create(json.loads(audit.read_text())["config"])
    if digest(Path(config.well)) != original["well_sha256"]:
        raise ValueError("Warm checkpoint changed")
    input_hashes[str(audit)] = digest(audit)
    input_hashes[str(Path(config.well).resolve())] = digest(Path(config.well))
    image_hashes = {p: digest(Path(p)) for paths in roles["paths"].values() for p in paths}
    if image_hashes != original["image_hashes"]:
        raise ValueError("Image content changed")
    source_hashes = {
        str(p): digest(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "cafl4ds/replay_local.py",
            ROOT / "cafl4ds/replay_response.py",
            ROOT / "cafl4ds/filters/replay_ratio.py",
            ROOT / "cafl4ds/reader_calibration.py",
            ROOT / "scripts/run_slow_edge_pilot.py",
            ROOT / "docs/project-plan/slow-edge-replay-generalization.md",
        )
    }
    atomic_json(
        run / "inputs.json",
        {
            "source": str(source),
            "local": str(local),
            "parents": str(parents),
            "input_hashes": input_hashes,
            "source_hashes": source_hashes,
            "image_hashes": image_hashes,
            "frozen_models": raw_models,
            "fresh_dataset_confirmation": False,
            "expected_branches": 144,
            "expected_updates": 864,
        },
    )
    atomic_json(run / "status.json", {"state": "decoding", "completed_branches": 0})
    images, labels, domains, indices = decode(roles["paths"], roles["tags"])
    if not torch.equal(labels[metadata["indices"]], metadata["labels"]) or not torch.equal(
        domains[metadata["indices"]], metadata["domains"]
    ):
        raise ValueError("Readout roles changed")
    torch.save(metadata, run / "readout_metadata.pt")
    atomic_json(run / "reader_frozen.json", reader)
    torch.manual_seed(62000)
    initial = _method(config)
    monitor, monitor_domains = images[indices["monitor"]], domains[indices["monitor"]]
    warm = isolated_embed(initial, monitor)
    torch.save(warm, run / "warm_monitor.pt")
    lookup = {tensor_hash(image): i for i, image in enumerate(images)}
    if len(lookup) != len(images):
        raise ValueError("Ambiguous image identity")
    feature_rows, choices, cases = [], [], []
    for state in sorted(bases):
        base, saved = bases[state], snapshots[state]
        method = copy.deepcopy(initial)
        method.load_state_dict(saved["method"])
        torch.set_rng_state(saved["rng"])
        memory = torch.stack(saved["actuator"]["buffer"])
        memory_indices = [lookup[tensor_hash(image)] for image in memory]
        diagnostic_indices = torch.randperm(len(memory), generator=torch.Generator().manual_seed(90003))[:32]
        before, rng, modes = (
            copy.deepcopy(method.state_dict()),
            torch.get_rng_state().clone(),
            [m.training for m in method.modules()],
        )
        z = isolated_embed(method, monitor)
        recon = reconstruction_draws(method, monitor, 2, 90001).mean(0)
        health = {f"health_{k}": v for k, v in sensor(z, warm, recon).items()}
        health.update(domain_sensors(z, warm, recon, monitor_domains))
        memory_losses = reconstruction_draws(method, memory, 2, 85003).mean(0)
        for condition in CONDITIONS:
            order_seed = 840001 + base["seed"] * 11 + CONDITIONS.index(condition)
            order = new_order(orders[state], domains, condition, order_seed)
            order_name = f"{state}__{condition}"
            atomic_json(
                run / f"order_{order_name}.json", {"indices": order, "seed": order_seed, "condition": condition}
            )
            current = images[order[320:352]]
            gradients = gradient_signals(method, current, memory[diagnostic_indices])
            current_losses = reconstruction_draws(method, current, 2, 85001).mean(0)
            for repeat in (2, 3):
                case_name = f"{order_name}__draw{repeat}"
                selection_seed = 930001 + base["seed"] * 17 + repeat
                training_seed = 950003 + base["seed"] * 19 + repeat
                attrs = {
                    "case": case_name,
                    "base_state": state,
                    "state": case_name,
                    "pool_group": base["pool_group"],
                    "stream": base["stream"],
                    "schedule": base["schedule"],
                    "condition": condition,
                    "repeat": repeat,
                    "prefix": 10,
                    "selection_seed": selection_seed,
                    "training_seed": training_seed,
                    "order_name": order_name,
                }
                observations = []
                for action in (8, 16, 24):
                    preview = ReplayRatio(action, seed=selection_seed)
                    preview.restore_memory(saved["actuator"])
                    preview.select(StreamBatch(current, era=0, step=10), FilterContext(method, 10))
                    c, r = (
                        current_losses[preview.last["current_indices"]],
                        memory_losses[preview.last["replay_indices"]],
                    )
                    observation = {
                        **attrs,
                        "action": action,
                        "features": {
                            "step": 10.0,
                            "history_replay_fraction": 0.5,
                            "current_night_fraction": float(domains[order[320:352]].float().mean()),
                            "memory_night_fraction": float(domains[memory_indices].float().mean()),
                            "domain_changed": float(domains[order[320]] != domains[order[288]]),
                            "current_reconstruction_mean": float(c.mean()),
                            "replay_reconstruction_mean": float(r.mean()),
                            "selected_reconstruction_mean": float(torch.cat((c, r)).mean()),
                            **health,
                            **gradients,
                        },
                        "preview": preview.last,
                        "diagnostic_memory_indices": diagnostic_indices.tolist(),
                        "memory_image_indices": memory_indices,
                    }
                    observations.append(observation)
                feature_rows.extend(observations)
                for recipe in RECIPES:
                    predictions = dict(
                        zip((8, 16, 24), models[(base["pool_group"], recipe)].predict(observations), strict=True)
                    )
                    choices.append(
                        {
                            **attrs,
                            "recipe": recipe,
                            "action": choose(predictions),
                            "predictions": predictions,
                            "accuracy_constant": int(constants[state]),
                        }
                    )
                for recipe, action in (("training_constant", int(constants[state])), ("constant16", 16)):
                    choices.append(
                        {
                            **attrs,
                            "recipe": recipe,
                            "action": action,
                            "predictions": {},
                            "accuracy_constant": int(constants[state]),
                        }
                    )
                cases.append(attrs)
            if (
                not identical(before, method.state_dict())
                or not torch.equal(rng, torch.get_rng_state())
                or modes != [m.training for m in method.modules()]
                or any(p.grad is not None for p in method.parameters())
            ):
                raise ValueError("Observation changed state/RNG/modes/gradients")
        atomic_json(
            run / "status.json", {"state": "observing", "completed_states": len(cases) // 4, "completed_branches": 0}
        )
    atomic_json(run / "predecision_features.json", {"rows": feature_rows})
    atomic_json(run / "frozen_choices.json", {"choices": choices})
    choice_hash = digest(run / "frozen_choices.json")
    scores: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []

    def score(z: torch.Tensor, attrs: dict[str, Any]) -> None:
        with threadpool_limits(limits=1):
            for panel, support in metadata["supports"].items():
                fitted = {}
                for recipe, c in reader["selected_C"][panel].items():
                    if c not in fitted:
                        fitted[c] = probe_outcomes(
                            z, metadata["labels"], metadata["domains"], support, metadata["roles"], c
                        )
                    for role, metrics in fitted[c].items():
                        scores.append({**attrs, "panel": panel, "role": role, "reader": recipe, "C": c, **metrics})

    for case in cases:
        saved = snapshots[case["base_state"]]
        order = json.loads((run / f"order_{case['order_name']}.json").read_text())["indices"]
        previous_signatures = None
        allocations = []
        for action in (8, 16, 24):
            method = copy.deepcopy(initial)
            method.load_state_dict(saved["method"])
            optimizer = optimizer_for(method)
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            selector = ReplayRatio(action, seed=case["selection_seed"])
            selector.restore_memory(saved["actuator"])
            torch.manual_seed(case["training_seed"])
            decisions, signatures = [], []
            for offset in range(6):
                step = 10 + offset
                record = update(method, optimizer, images[order[step * 32 : (step + 1) * 32]], step, selector)
                decisions.append({**record, **selector.last})
                signatures.append(
                    {
                        "buffer": tensor_hash(torch.stack(selector.reservoir._buffer)),
                        "rng": tensor_hash(selector.reservoir._gen.get_state()),
                        "seen": selector.reservoir._seen,
                    }
                )
            if previous_signatures is not None and previous_signatures != signatures:
                raise ValueError("Action-dependent memory or sampler")
            previous_signatures = signatures
            allocations.append(decisions)
            z = isolated_embed(method, images[metadata["indices"]])
            torch.save(z, run / f"features_{case['case']}_{action}.pt")
            score(z, {**case, "replay_count": action, "horizon": 6})
            branches.append({**case, "replay_count": action, "decisions": decisions, "memory_signatures": signatures})
            atomic_json(run / "branches.json", {"branches": branches})
            pd.DataFrame(scores).to_csv(run / "scores.csv", index=False)
            atomic_json(
                run / "status.json",
                {"state": "training", "completed_branches": len(branches), "expected_branches": 144},
            )
            print(f"Collected {len(branches)}/144 branches: {case['case']}, replay{action}", flush=True)
        for offset in range(6):
            low, mid, high = [a[offset] for a in allocations]
            if (
                not (set(high["current_indices"]) <= set(mid["current_indices"]) <= set(low["current_indices"]))
                or low["replay_indices"] != high["replay_indices"][:8]
                or mid["replay_indices"] != high["replay_indices"][:16]
            ):
                raise ValueError("Non-nested samples")
        if not identical(saved["optimizer"], snapshots[case["base_state"]]["optimizer"]) or not identical(
            saved["optimizer"], torch.load(local / f"parent_{case['base_state']}.pt", weights_only=False)["optimizer"]
        ):
            raise ValueError("Parent optimizer state mutated")
    if len(branches) != 144 or len(scores) != 3456 or digest(run / "frozen_choices.json") != choice_hash:
        raise ValueError("Incomplete collection or choices changed")
    for path, value in {**input_hashes, **source_hashes, **image_hashes}.items():
        if digest(Path(path)) != value:
            raise ValueError(f"Changed input {path}")
    atomic_json(
        run / "verification.json",
        {
            "branches": 144,
            "new_ssl_updates": 864,
            "choices_frozen_before_training_sha256": choice_hash,
            "elapsed_seconds": time.time() - started,
            "artifact_hashes": {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != "status.json"},
        },
    )
    atomic_json(run / "status.json", {"state": "complete", "completed_branches": 144})


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
