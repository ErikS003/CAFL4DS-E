"""Identify implemented constant-alpha responses on fresh, paired development trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.filters.steerable import SteerableReplay, fit_anchors
from cafl4ds.image_utility import FixedProbe, embed, split_query_roles, utility_features
from cafl4ds.utility_diagnostics import diagnostic_outcomes
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_image_utility_experiment import identical
from scripts.run_slow_edge_pilot import ROOT, load_inputs, optimizer_for, update


def memory_signature(actuator: SteerableReplay) -> dict[str, Any]:
    """Audit action-independent memory and its private sampling state after each update."""
    return {
        "buffer": tensor_hash(torch.stack(actuator.reservoir._buffer)),
        "rng": tensor_hash(actuator.reservoir._gen.get_state()),
        "seen": actuator.reservoir._seen,
    }


def main() -> None:  # noqa: C901 - explicit matched branches and provenance checks
    """Save every matched state and branch, and fail on unequal work or observation mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", choices=["bdd", "synthetic"], default="bdd")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.set_num_threads(4)
    torch.manual_seed(62000)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    atomic_json(output / "progress.json", {"state": "loading"})
    seeds = [6701] if args.smoke else [6701, 6707, 6711, 6717]
    steps = [4] if args.smoke else [4, 12, 20, 28, 36]
    repeats, horizons = (1 if args.smoke else 2), [4, 12]
    paths = [
        *sorted((ROOT / "cafl4ds").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
    ]
    hashes = {str(p): digest(p) for p in paths}
    exclusion_runs = [
        "20261006-bdd-large-gradient",
        "20261006-bdd-context-chronological",
        "20261006-bdd-utility-confirmation",
    ]
    excluded: set[str] = set()
    exclusion_hashes = {}
    if args.data == "bdd":
        for name in exclusion_runs:
            path = ROOT / "outputs/image-utility" / name / "inputs.json"
            exclusion_hashes[str(path)] = digest(path)
            excluded.update(json.loads(path.read_text())["train_paths"])
    records = []
    try:
        initial, warm, arrivals, evaluation, provenance = load_inputs(
            args.data, train_per_cell=256, stream_seed=67003, excluded_paths=excluded
        )
        if excluded & set(provenance.get("train_paths", [])):
            raise ValueError("New stream overlaps previous experiments")
        if max(steps) + 12 > len(arrivals) // 32:
            raise ValueError("Incomplete action-response windows")
        centers, target = fit_anchors(embed(initial, warm), seed=62000)
        roles = split_query_roles(evaluation)
        monitor = evaluation.images[roles["monitor"]]
        warm_monitor = embed(initial, monitor)
        fixed = FixedProbe.fit(initial, evaluation)
        atomic_json(
            output / "inputs.json",
            {
                **provenance,
                "source_hashes": hashes,
                "exclusion_manifest_hashes": exclusion_hashes,
                "seeds": seeds,
                "steps": steps,
                "repeats": repeats,
                "horizons": horizons,
                "primary_horizon": 12,
                "alphas": [0.0, 0.5, 1.0],
                "smoke": args.smoke,
                "centroids": centers.tolist(),
                "target": target.tolist(),
                "roles": {k: v.tolist() for k, v in roles.items()},
                "planned_branches": len(seeds) * len(steps) * repeats * 3,
                "planned_branch_updates": len(seeds) * len(steps) * repeats * 3 * 12,
                "prefix_policy": "implemented alpha=0.5; shared common prefix for all branches",
                "repeat_policy": "repeat 0 uses prefix RNG; repeat 1 independent training RNG; same replay state",
                "limitations": [
                    "Development only; shared well, stream and evaluation corpus across seeds",
                    "Windows overlap; seeds and steps are not independent new datasets",
                    "Constant-action responses conditional on alpha=0.5 history; no arbitrary sequence identification",
                    "Outcome labels fit support probes and evaluate queries; never control SSL or selection",
                    "No response predictor fitted or MPC qualification claim",
                ],
            },
        )
        del warm
        for seed in seeds:
            torch.manual_seed(seed)
            common = copy.deepcopy(initial)
            optimizer = optimizer_for(common)
            actuator = SteerableReplay(initial.encoder, centers, target, seed=seed)
            for step in range(max(steps) + 1):
                batch = arrivals[step * 32 : (step + 1) * 32]
                if step in steps:
                    snapshot: dict[str, Any] = {
                        "method": copy.deepcopy(common.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "actuator": actuator.state_dict(),
                        "rng": torch.get_rng_state().clone(),
                    }
                    torch.save(snapshot, output / f"snapshot_{seed}_{step}.pt")
                    _, health = utility_features(common, actuator, batch, monitor, warm_monitor)
                    before = diagnostic_outcomes(common, evaluation, roles, fixed)
                    if (
                        not identical(common.state_dict(), snapshot["method"])
                        or not identical(actuator.state_dict(), snapshot["actuator"])
                        or not torch.equal(snapshot["rng"], torch.get_rng_state())
                    ):
                        raise ValueError("Pre-decision observation mutated parent state")
                    for repeat in range(repeats):
                        paired_memory = None
                        order = torch.randperm(
                            3, generator=torch.Generator().manual_seed(seed + step * 100 + repeat)
                        ).tolist()
                        for action in order:
                            alpha = [0.0, 0.5, 1.0][action]
                            trial = copy.deepcopy(common)
                            opt = optimizer_for(trial)
                            opt.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
                            selector = copy.deepcopy(actuator)
                            selector.set_action(alpha)
                            torch.set_rng_state(snapshot["rng"])
                            if repeat:
                                torch.manual_seed(seed * 10000 + step * 100 + repeat)
                            decisions, memory, outcomes, post_health = [], [], {}, {}
                            for offset in range(12):
                                actual_step = step + offset
                                images = arrivals[actual_step * 32 : (actual_step + 1) * 32]
                                audit = update(trial, opt, images, actual_step, selector)
                                if selector.last["current_count"] != 16 or selector.last["replay_count"] != 16:
                                    raise ValueError("Expected 16 current + 16 replay images")
                                decisions.append({**copy.deepcopy(selector.last), **audit})
                                memory.append(memory_signature(selector))
                                if offset + 1 in horizons:
                                    outcomes[offset + 1] = diagnostic_outcomes(trial, evaluation, roles, fixed)
                                    _, post_health[offset + 1] = utility_features(
                                        trial, selector, images, monitor, warm_monitor
                                    )
                            if selector.optimizer_images - actuator.optimizer_images != 384:
                                raise ValueError("Unequal optimizer work")
                            if paired_memory is None:
                                paired_memory = memory
                            elif memory != paired_memory:
                                raise ValueError("Actions changed replay memory or private sampling RNG")
                            row = {
                                "seed": seed,
                                "step": step,
                                "repeat": repeat,
                                "alpha": alpha,
                                "before": before,
                                "pre_health": health,
                                "outcomes": outcomes,
                                "post_health": post_health,
                                "decisions": decisions,
                                "memory": memory,
                                "optimizer_images": 384,
                                "execution_order": order,
                            }
                            atomic_json(output / f"branch_{seed}_{step}_{repeat}_{alpha}.json", row)
                            records.append(row)
                            atomic_json(
                                output / "progress.json",
                                {
                                    "state": "collecting",
                                    "completed_branches": len(records),
                                    "planned_branches": len(seeds) * len(steps) * repeats * 3,
                                    "seed": seed,
                                    "step": step,
                                    "elapsed_seconds": time.time() - started,
                                },
                            )
                            print(
                                f"seed={seed} step={step} repeat={repeat} alpha={alpha}: {len(records)} branches",
                                flush=True,
                            )
                    torch.set_rng_state(snapshot["rng"])
                    if (
                        not identical(snapshot["method"], common.state_dict())
                        or not identical(snapshot["optimizer"], optimizer.state_dict())
                        or not identical(snapshot["actuator"], actuator.state_dict())
                    ):
                        raise ValueError("Branching mutated parent")
                update(common, optimizer, batch, step, actuator)
        for guarded_path, expected in {**hashes, **exclusion_hashes}.items():
            if digest(Path(guarded_path)) != expected:
                raise ValueError(f"Scientific source changed during run: {guarded_path}")
        atomic_json(output / "dataset.json", {"rows": records})
        atomic_json(
            output / "progress.json",
            {
                "state": "completed",
                "completed_branches": len(records),
                "branch_updates": len(records) * 12,
                "elapsed_seconds": time.time() - started,
                "matched_memory_every_update": True,
                "parent_state_unchanged": True,
            },
        )
    except Exception as error:
        atomic_json(
            output / "progress.json", {"state": "failed", "error": repr(error), "completed_branches": len(records)}
        )
        raise


if __name__ == "__main__":
    main()
