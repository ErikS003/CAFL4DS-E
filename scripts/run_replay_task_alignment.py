"""Freeze support-labeled replay proxies before new one-update outcome collection."""

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
import torch.nn.functional as F
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.image_utility import FixedProbe, balanced_losses
from cafl4ds.reader_calibration import fit_probe, probe_outcomes
from cafl4ds.replay_task_alignment import head_logits, ssl_directions, task_gradient
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_fresh_alpha import decode
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import ROOT, optimizer_for, update
from scripts.run_support_population import isolated_embed

ACTIONS = (8, 16, 24)
ROTATIONS = (("support1", "support2", "support3"), ("support2", "support3", "support1"), ("support3", "support1", "support2"))
OBSERVATION_SEEDS = (111003, 112009)


def choose_proxy(values: dict[int, float]) -> int:
    """Freeze ties toward16, then8, then24 without fitting thresholds."""
    best = max(values.values())
    return next(a for a in (16, 8, 24) if values[a] >= best - 1e-12)


def support_ce(probe: FixedProbe, z: torch.Tensor, metadata: dict[str, Any], support: str) -> float:
    """Stable balanced frozen-head CE with the same objective as the task gradient."""
    ix = metadata["supports"][support]
    classes = torch.as_tensor(probe.classifier.classes_)
    labels = metadata["labels"][ix]
    targets = (labels[:, None] == classes[None, :]).long().argmax(1)
    losses = F.cross_entropy(head_logits(probe, z[ix]), targets, reduction="none")
    return balanced_losses(losses, labels, metadata["domains"][ix])["balanced"]


def main() -> None:  # noqa: C901
    """Use labels only on recorded support sets for all predecision observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run = parser.parse_args().output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = ROOT / "outputs/image-utility/20261008-replay-mask-stability"
    local = ROOT / "outputs/image-utility/20261008-replay-local-v2"
    parents = ROOT / "outputs/image-utility/20261008-new-replay-parents"
    hashes: dict[str, str] = {}

    def verified(folder: Path, name: str) -> Path:
        path = folder / name
        expected = json.loads((folder / "verification.json").read_text())["artifact_hashes"][name]
        if digest(path) != expected:
            raise ValueError(f"Changed input {path}")
        hashes[str(path)] = expected
        return path

    old_branches = json.loads(verified(source, "branches.json").read_text())["branches"]
    branches = [b for b in old_branches if b["repeat"] in (4, 5)]
    if len(branches) != 72:
        raise ValueError("Expected72 audited six-update sources")
    old_choices = json.loads(verified(source, "frozen_choices.json").read_text())["choices"]
    old_scores = pd.read_csv(verified(source, "scores.csv"))
    metadata = torch.load(verified(source, "readout_metadata.pt"), weights_only=False)
    reader = json.loads(verified(source, "reader_frozen.json").read_text())
    roles = json.loads(verified(parents, "roles.json").read_text())
    audit_path = ROOT / "outputs/image-utility/20261007-fresh-alpha-allocation.json"
    original = json.loads(verified(parents, "inputs.json").read_text())
    if digest(audit_path) != original["audit_sha256"]:
        raise ValueError("Changed config")
    hashes[str(audit_path)] = digest(audit_path)
    config = OmegaConf.create(json.loads(audit_path.read_text())["config"])
    if digest(Path(config.well)) != original["well_sha256"]:
        raise ValueError("Changed warm checkpoint")
    hashes[str(Path(config.well).resolve())] = digest(Path(config.well))
    image_hashes = {p: digest(Path(p)) for paths in roles["paths"].values() for p in paths}
    if image_hashes != original["image_hashes"]:
        raise ValueError("Changed images")
    states = sorted({b["base_state"] for b in branches})
    snapshots, parent_z, orders = {}, {}, {}
    for state in states:
        snapshots[state] = torch.load(verified(local, f"parent_{state}.pt"), weights_only=False)
        parent_z[state] = torch.load(verified(local, f"features_parent_{state}.pt"), weights_only=True)
        order_name = next(b["order_name"] for b in branches if b["base_state"] == state)
        orders[state] = json.loads(verified(source, f"order_{order_name}.json").read_text())["indices"]
    for b in branches:
        verified(source, f"features_{b['case']}_{b['replay_count']}.pt")
    source_hashes = {
        str(p): digest(p)
        for p in (
            Path(__file__).resolve(), ROOT / "cafl4ds/replay_task_alignment.py",
            ROOT / "cafl4ds/reader_calibration.py", ROOT / "cafl4ds/filters/replay_ratio.py",
            ROOT / "scripts/run_slow_edge_pilot.py", ROOT / "docs/project-plan/slow-edge-task-aligned-replay-frozen.md",
        )
    }
    atomic_json(run / "inputs.json", {"source": str(source), "local": str(local), "parents": str(parents), "input_hashes": hashes, "source_hashes": source_hashes, "image_hashes": image_hashes, "rotations": ROTATIONS, "observation_seeds": OBSERVATION_SEEDS, "task_head_C": 0.1, "new_trajectory_updates": 72, "virtual_observation_steps": 72, "task_gradient_observations": 36, "cached_six_update_branches": 72, "fresh_dataset_confirmation": False})
    atomic_json(run / "status.json", {"state": "decoding", "completed_parents": 0})
    images, labels, domains, _ = decode(roles["paths"], roles["tags"])
    if not torch.equal(labels[metadata["indices"]], metadata["labels"]) or not torch.equal(domains[metadata["indices"]], metadata["domains"]):
        raise ValueError("Readout identity changed")
    readout_images = images[metadata["indices"]]
    torch.save(metadata, run / "readout_metadata.pt")
    atomic_json(run / "reader_frozen.json", reader)
    initial = _method(config)
    forecasts: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    observation_logs: list[dict[str, Any]] = []
    heads: dict[str, list[FixedProbe]] = {}
    # Phase1: all parent observations and all choices precede NEW outcome training.
    for position, state in enumerate(states):
        saved = snapshots[state]
        method = copy.deepcopy(initial)
        method.load_state_dict(saved["method"])
        state_before = copy.deepcopy(method.state_dict())
        modes_before = [m.training for m in method.modules()]
        grads_before = [None if p.grad is None else p.grad.clone() for p in method.parameters()]
        rng_before = torch.get_rng_state().clone()
        head_list, gradients = [], []
        for fitting, gradient_support, _ in ROTATIONS:
            with threadpool_limits(limits=1):
                probe = fit_probe(parent_z[state], metadata["labels"], metadata["supports"][fitting], 0.1)
            ix = metadata["supports"][gradient_support]
            if torch.isin(ix, metadata["supports"][fitting]).any() or torch.isin(ix, torch.cat(list(metadata["roles"].values()))).any():
                raise ValueError("Support split leakage")
            gradient, ce = task_gradient(method, probe, readout_images[ix], metadata["labels"][ix], metadata["domains"][ix])
            if abs(ce - support_ce(probe, parent_z[state], metadata, gradient_support)) > 2e-6:
                raise ValueError("Gradient objective differs from frozen-head outcome")
            head_list.append(probe)
            gradients.append(gradient)
        heads[state] = head_list
        torch.save({"heads": head_list, "task_gradients": gradients}, run / f"task_{state}.pt")
        attrs = next(b for b in branches if b["base_state"] == state)
        for action in ACTIONS:
            selector = ReplayRatio(action, seed=attrs["selection_seed"])
            selector.restore_memory(saved["actuator"])
            incoming = images[orders[state][320:352]]
            selected = selector.select(StreamBatch(incoming, era=0, step=10), FilterContext(method, 10))
            expected = next(b for b in branches if b["base_state"] == state and b["replay_count"] == action)["decisions"][0]
            if any(selector.last[k] != expected[k] for k in selector.last):
                raise ValueError("Observation selection differs from trajectory")
            sgd_draws, adam_draws = [], []
            for seed in OBSERVATION_SEEDS:
                sgd, adam, log = ssl_directions(method, saved["optimizer"], selected, seed)
                sgd_draws.append(sgd)
                adam_draws.append(adam)
                observation_logs.append({"base_state": state, "action": action, "observation_seed": seed, **log})
            vectors = {"sgd": torch.stack(sgd_draws).mean(0), "adamw": torch.stack(adam_draws).mean(0)}
            torch.save({**vectors, "sgd_draws": sgd_draws, "adamw_draws": adam_draws, "selection": selector.last}, run / f"directions_{state}_{action}.pt")
            for rotation, gradient in enumerate(gradients):
                for proxy, delta in vectors.items():
                    forecasts.append({"base_state": state, "pool_group": attrs["pool_group"], "rotation": rotation, "fitting_support": ROTATIONS[rotation][0], "gradient_support": ROTATIONS[rotation][1], "third_support": ROTATIONS[rotation][2], "proxy": proxy, "action": action, "predicted_ce_reduction": -float(torch.dot(gradient.double(), delta.double())), "task_gradient_norm": float(gradient.norm()), "direction_norm": float(delta.norm()), "negative_direction_cosine": -float(torch.dot(gradient, delta) / (gradient.norm() * delta.norm()))})
        if not identical(state_before, method.state_dict()) or modes_before != [m.training for m in method.modules()] or not torch.equal(rng_before, torch.get_rng_state()) or any(not identical(g, p.grad) for g, p in zip(grads_before, method.parameters(), strict=True)) or not identical(saved["optimizer"], torch.load(local / f"parent_{state}.pt", weights_only=False)["optimizer"]):
            raise ValueError("Read-only observations mutated parent or RNG")
        state_rows = [f for f in forecasts if f["base_state"] == state]
        for proxy in ("sgd", "adamw"):
            for rotation in (-1, 0, 1, 2):
                values = {a: float(np.mean([f["predicted_ce_reduction"] for f in state_rows if f["proxy"] == proxy and f["action"] == a and (rotation == -1 or f["rotation"] == rotation)])) for a in ACTIONS}
                choices.append({"base_state": state, "pool_group": attrs["pool_group"], "recipe": f"task_{proxy}", "rotation": rotation, "action": choose_proxy(values), "predictions": values})
        for recipe in ("composition", "reconstruction", "health_gradients"):
            c = next(c for c in old_choices if c["base_state"] == state and c["repeat"] == 4 and c["recipe"] == recipe)
            choices.append({"base_state": state, "pool_group": attrs["pool_group"], "recipe": recipe, "rotation": -1, "action": c["action"], "predictions": c["predictions"]})
        for action in ACTIONS:
            choices.append({"base_state": state, "pool_group": attrs["pool_group"], "recipe": f"constant{action}", "rotation": -1, "action": action, "predictions": {}})
        atomic_json(run / "status.json", {"state": "observing", "completed_parents": position + 1, "expected_parents": 12})
        print(f"Observed parent{position+1}/12: {state}", flush=True)
    pd.DataFrame(forecasts).to_csv(run / "forecasts.csv", index=False)
    pd.DataFrame(observation_logs).to_csv(run / "observation_work.csv", index=False)
    atomic_json(run / "frozen_choices.json", {"choices": choices, "before_new_outcome_training": True})
    frozen_hashes = {n: digest(run / n) for n in ("forecasts.csv", "frozen_choices.json")}
    atomic_json(run / "pretraining_verification.json", {"frozen_hashes": frozen_hashes, "parents_unchanged": True, "new_trajectory_updates_so_far": 0, "virtual_observation_steps": len(observation_logs), "task_gradient_observations": 36})
    query_scores: list[dict[str, Any]] = []
    fixed_scores: list[dict[str, Any]] = []
    collected: list[dict[str, Any]] = []
    # Phase2: no prediction or choice can change below this boundary.
    for position, b in enumerate(branches):
        state, action = b["base_state"], b["replay_count"]
        saved = snapshots[state]
        attrs = {k: b[k] for k in ("case", "base_state", "pool_group", "repeat", "training_seed", "selection_seed", "replay_count")}
        method = copy.deepcopy(initial)
        method.load_state_dict(saved["method"])
        optimizer = optimizer_for(method)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        selector = ReplayRatio(action, seed=b["selection_seed"])
        selector.restore_memory(saved["actuator"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(b["training_seed"])
            record = update(method, optimizer, images[orders[state][320:352]], 10, selector)
        expected = b["decisions"][0]
        if {**record, **selector.last} != expected:
            raise ValueError("One-step trajectory does not reproduce audited six-step first update")
        z1 = isolated_embed(method, readout_images)
        torch.save(z1, run / f"features_{b['case']}_{action}_h1.pt")
        z6 = torch.load(source / f"features_{b['case']}_{action}.pt", weights_only=True)
        with threadpool_limits(limits=1):
            for panel, support in metadata["supports"].items():
                fitted = {}
                for reader_name, c in reader["selected_C"][panel].items():
                    if c not in fitted:
                        fitted[c] = probe_outcomes(z1, metadata["labels"], metadata["domains"], support, metadata["roles"], c)
                    for role, metrics in fitted[c].items():
                        query_scores.append({**attrs, "horizon": 1, "panel": panel, "role": role, "reader": reader_name, "C": c, **metrics})
        cached_rows = old_scores[(old_scores.case == b["case"]) & (old_scores.replay_count == action)].to_dict("records")
        query_scores.extend({**r, **attrs, "horizon": 6} for r in cached_rows)
        for rotation, probe in enumerate(heads[state]):
            for support_role, support in (("gradient", ROTATIONS[rotation][1]), ("third", ROTATIONS[rotation][2])):
                before = support_ce(probe, parent_z[state], metadata, support)
                for horizon, z in ((1, z1), (6, z6)):
                    after = support_ce(probe, z, metadata, support)
                    fixed_scores.append({**attrs, "rotation": rotation, "support_role": support_role, "support": support, "horizon": horizon, "before_ce": before, "after_ce": after, "ce_reduction": before - after})
        collected.append({**attrs, "horizon": 1, "decision": {**record, **selector.last}, "cached_h6_case": b["case"]})
        pd.DataFrame(query_scores).to_csv(run / "scores.csv", index=False)
        pd.DataFrame(fixed_scores).to_csv(run / "fixed_support_scores.csv", index=False)
        atomic_json(run / "branches.json", {"branches": collected})
        atomic_json(run / "status.json", {"state": "training", "completed_branches": position + 1, "expected_branches": 72})
        print(f"Collected one-update branch{position+1}/72: {b['case']}, replay{action}", flush=True)
    for path, value in {**hashes, **source_hashes, **image_hashes}.items():
        if digest(Path(path)) != value:
            raise ValueError(f"Changed input/source {path}")
    if any(digest(run / n) != h for n, h in frozen_hashes.items()):
        raise ValueError("Predecision forecasts or choices changed")
    if len(collected) != 72 or len(query_scores) != 3456 or len(fixed_scores) != 864 or len(forecasts) != 216:
        raise ValueError("Incomplete study")
    atomic_json(run / "verification.json", {"branches": 72, "new_trajectory_updates": 72, "virtual_observation_steps": 72, "task_gradient_observations": 36, "cached_six_update_branches": 72, "forecasts_frozen_before_training": frozen_hashes, "elapsed_seconds": time.time() - started, "artifact_hashes": {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name not in ("status.json", "verification.json")}})
    atomic_json(run / "status.json", {"state": "complete", "completed_branches": 72})


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
