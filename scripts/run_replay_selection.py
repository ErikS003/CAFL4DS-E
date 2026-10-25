"""Freeze replay-content interventions, then collect paired outcomes and health gates."""

# ruff: noqa: E501
# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.streams import StreamBatch
from cafl4ds.domain_replay_health import sensor
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.image_utility import reconstruction_draws
from cafl4ds.replay_health_support import metrics, precise_probe
from cafl4ds.replay_local import encoder_gradient
from cafl4ds.replay_selection import RULES, FixedReplaySelection, coverage_selection, ranked_blocks
from cafl4ds.replay_task_alignment import ssl_directions, task_gradient
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_fresh_alpha import decode
from scripts.run_image_utility_experiment import identical
from scripts.run_replay_task_alignment import ROTATIONS
from scripts.run_slow_edge_pilot import ROOT, optimizer_for, update
from scripts.run_support_population import isolated_embed

OBSERVATION_SEEDS = (181003, 182009)
OUTCOME_SEEDS = (193001, 194009, 195023)
TOLERANCES = (1e-8, 1e-10)


def freeze_health_gates(
    features: list[dict[str, Any]], scores: list[dict[str, Any]], target_horizon: int = 1
) -> dict[str, Any]:
    """Fit only development draw0 on other pools, choosing unlabeled coverage/uniform."""
    rows = pd.DataFrame(scores)
    primary = rows.loc[
        (rows.draw == 0)
        & (rows.horizon == target_horizon)
        & (rows.panel == "pooled")
        & (rows.role == "development")
        & (rows.tolerance == 1e-10)
    ]
    groups = sorted({r["pool_group"] for r in features})
    if len(groups) < 3:
        return {"choices": [], "models": [], "reason": "Smoke data do not have three pool groups"}
    lookup = {(r["state"], r["rule"]): r["features"] for r in features}
    observations: list[dict[str, Any]] = []
    for state, group in primary.groupby("state"):
        values = group.set_index("rule")
        a, b = lookup[(str(state), "coverage")], lookup[(str(state), "uniform")]
        observations.append(
            {
                "state": str(state),
                "pool_group": str(values.iloc[0].pool_group),
                "features": {k: a[k] if k.startswith("health_") else a[k] - b[k] for k in a},
                "target": float(
                    values.loc["coverage", "accuracy_balanced"] - values.loc["uniform", "accuracy_balanced"]
                ),
            }
        )
    fields = {
        "constant": [],
        "reconstruction": ["selected_reconstruction", "virtual_ssl_loss"],
        "health_ssl": [
            "selected_reconstruction",
            "virtual_ssl_loss",
            "selected_coverage",
            "selected_variance",
            "ssl_gradient_norm",
            "current_ssl_cosine",
            "health_rankme",
            "health_variance",
            "health_cka_drift",
            "health_reconstruction",
        ],
    }
    choices, models = [], []
    for heldout in groups:
        training = [r for r in observations if r["pool_group"] != heldout]
        testing = [r for r in observations if r["pool_group"] == heldout]
        for recipe, names in fields.items():
            x = np.array([[r["features"][k] for k in names] for r in training])
            y = np.array([r["target"] for r in training])
            mean, scale = x.mean(0), x.std(0)
            scale[scale < 1e-12] = 1
            model = Ridge(alpha=100.0).fit((x - mean) / scale, y) if names else None
            models.append(
                {
                    "heldout_pool": heldout,
                    "recipe": recipe,
                    "fields": names,
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                    "weights": model.coef_.tolist() if model else [],
                    "intercept": float(model.intercept_) if model else float(y.mean()),
                    "training_states": [r["state"] for r in training],
                }
            )
            for row in testing:
                prediction = (
                    float(model.predict((np.array([[row["features"][k] for k in names]]) - mean) / scale)[0])
                    if model
                    else float(y.mean())
                )
                choices.append(
                    {
                        "state": row["state"],
                        "heldout_pool": heldout,
                        "recipe": recipe,
                        "predicted_gain": prediction,
                        "rule": "coverage" if prediction >= 0.001 else "uniform",
                    }
                )
    return {
        "choices": choices,
        "models": models,
        "target": f"development draw0 {target_horizon}-update pooled accuracy",
        "target_horizon": target_horizon,
        "minimum_predicted_gain": 0.001,
        "query_labels_runtime": False,
        "task_gradient_features_used": False,
    }


def main() -> None:  # noqa: C901
    """Separate observations, draw0 model fitting and later frozen-policy evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    local = ROOT / "outputs/image-utility/20261008-replay-local-v2"
    origin = ROOT / "outputs/image-utility/20261008-new-replay-parents"
    input_hashes: dict[str, str] = {}
    manifests = {p: json.loads((p / "verification.json").read_text())["artifact_hashes"] for p in (local, origin)}

    def verified(folder: Path, name: str) -> Path:
        path = folder / name
        expected = manifests[folder][name]
        if digest(path) != expected:
            raise ValueError(f"Changed historical input: {path}")
        input_hashes[str(path)] = expected
        return path

    metadata = torch.load(verified(local, "readout_metadata.pt"), weights_only=False)
    reader = json.loads(verified(local, "reader_frozen.json").read_text())
    old_rows = json.loads(verified(local, "predecision_features.json").read_text())["rows"]
    roles = json.loads(verified(origin, "roles.json").read_text())
    original = json.loads(verified(origin, "inputs.json").read_text())
    audit = ROOT / "outputs/image-utility/20261007-fresh-alpha-allocation.json"
    if digest(audit) != original["audit_sha256"]:
        raise ValueError("Configuration changed")
    input_hashes[str(audit)] = digest(audit)
    config = OmegaConf.create(json.loads(audit.read_text())["config"])
    if digest(Path(config.well)) != original["well_sha256"]:
        raise ValueError("Warm checkpoint changed")
    input_hashes[str(Path(config.well).resolve())] = digest(Path(config.well))
    for paths in roles["paths"].values():
        for name in paths:
            if digest(Path(name)) != original["image_hashes"][name]:
                raise ValueError(f"Image changed: {name}")
            input_hashes[name] = original["image_hashes"][name]
    source_files = [
        Path(__file__).resolve(),
        ROOT / "cafl4ds/replay_selection.py",
        ROOT / "cafl4ds/replay_task_alignment.py",
        ROOT / "cafl4ds/replay_health_support.py",
        ROOT / "cafl4ds/replay_local.py",
        ROOT / "cafl4ds/filters/replay_ratio.py",
        ROOT / "cafl4ds/filters/reservoir.py",
        ROOT / "cafl4ds/image_utility.py",
        ROOT / "scripts/run_slow_edge_pilot.py",
        ROOT / "scripts/run_fresh_alpha.py",
        ROOT / "scripts/run_support_population.py",
        ROOT / "scripts/report_replay_selection.py",
        ROOT / "docs/project-plan/slow-edge-replay-selection-frozen.md",
    ]
    source_hashes = {str(p): digest(p) for p in source_files}
    source_snapshots = run / "source_snapshots"
    for source_path in source_files:
        destination = source_snapshots / source_path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        if digest(destination) != source_hashes[str(source_path)]:
            raise ValueError("Frozen source snapshot differs")
    states = sorted([r for r in old_rows if r["phase"] == "transition" and r["action"] == 16], key=lambda r: r["state"])
    if len(states) != 12:
        raise ValueError("Expected12 transition parents")
    if args.smoke:
        states = states[:1]
    draws = OUTCOME_SEEDS[:2] if args.smoke else OUTCOME_SEEDS
    atomic_json(run / "status.json", {"state": "decoding", "completed_branches": 0})
    images, labels, domains, _ = decode(roles["paths"], roles["tags"])
    if not torch.equal(labels[metadata["indices"]], metadata["labels"]) or not torch.equal(
        domains[metadata["indices"]], metadata["domains"]
    ):
        raise ValueError("Readout image identities changed")
    supports = metadata["supports"]
    if torch.isin(supports["pooled"], torch.cat(list(metadata["roles"].values()))).any():
        raise ValueError("Support/query overlap")
    for a, b, _ in ROTATIONS:
        if torch.isin(supports[a], supports[b]).any():
            raise ValueError("Task-head/gradient support overlap")
    readout_images = images[metadata["indices"]]
    torch.save(metadata, run / "readout_metadata.pt")
    atomic_json(run / "reader_frozen.json", reader)
    torch.manual_seed(62000)
    initial = _method(config)
    parents, orders, plans, task_heads = {}, {}, [], {}
    features, block_rows, forecasts, solver_rows = [], [], [], []
    observation_steps = 0
    # No branch training is allowed until every selection/observation is frozen.
    for position, state in enumerate(states):
        name = state["state"]
        saved = torch.load(verified(local, f"parent_{name}.pt"), weights_only=False)
        order = json.loads(verified(local, f"order_{name}.json").read_text())["indices"]
        if torch.isin(torch.tensor(order[: (int(state["prefix"]) + 4) * 32]), metadata["indices"]).any():
            raise ValueError("Training arrivals overlap readout images")
        parents[name], orders[name] = saved, order
        method = copy.deepcopy(initial)
        method.load_state_dict(saved["method"])
        rng_before = torch.get_rng_state().clone()
        modes_before = [m.training for m in method.modules()]
        grads_before = [None if p.grad is None else p.grad.clone() for p in method.parameters()]
        parent_features = isolated_embed(method, readout_images)
        gradients, heads = [], []
        for fitting, gradient_support, _ in ROTATIONS:
            with threadpool_limits(limits=1):
                head = precise_probe(parent_features, metadata["labels"], supports[fitting], 0.1, 1e-10)
            ix = supports[gradient_support]
            gradient, _ = task_gradient(
                method, head, readout_images[ix], metadata["labels"][ix], metadata["domains"][ix]
            )
            gradients.append(gradient)
            heads.append(head)
        task_heads[name] = heads
        torch.save({"heads": heads, "gradients": gradients}, run / f"task_{name}.pt")
        prefix = int(state["prefix"])
        incoming = images[order[prefix * 32 : (prefix + 1) * 32]]
        preview = ReplayRatio(16, seed=state["seed"])
        preview.restore_memory(saved["actuator"])
        generator = torch.Generator()
        generator.set_state(saved["actuator"]["rng"].clone())
        candidates = torch.randperm(len(saved["actuator"]["buffer"]), generator=generator)[:64]
        preview.select(StreamBatch(incoming, era=0, step=prefix), FilterContext(method, prefix))
        if preview.last["replay_indices"] != candidates[:16].tolist():
            raise ValueError("Uniform baseline is not the exact ordinary draw")
        current = incoming[preview.last["current_indices"]]
        memory = torch.stack(saved["actuator"]["buffer"])[candidates]
        z = isolated_embed(method, torch.cat((current, memory)))
        warm = isolated_embed(initial, memory)
        losses = reconstruction_draws(method, memory, 2, 183001).mean(0)
        health = sensor(z[16:], warm, losses)
        current_gradients = [encoder_gradient(method, current, seed) for seed in OBSERVATION_SEEDS]
        scores_by_rotation = torch.empty(3, 8, dtype=torch.float64)
        for block in range(8):
            deltas = [
                ssl_directions(
                    method, saved["optimizer"], torch.cat((current, memory[block * 8 : (block + 1) * 8])), seed
                )[1]
                for seed in OBSERVATION_SEEDS
            ]
            observation_steps += len(deltas)
            delta = torch.stack(deltas).mean(0)
            for rotation, gradient in enumerate(gradients):
                scores_by_rotation[rotation, block] = -torch.dot(gradient.double(), delta.double())
                block_rows.append(
                    {
                        "state": name,
                        "pool_group": state["pool_group"],
                        "block": block,
                        "rotation": rotation,
                        "predicted_ce_reduction": float(scores_by_rotation[rotation, block]),
                    }
                )
        selected = {
            "uniform": list(range(16)),
            "coverage": coverage_selection(z[16:], z[:16]),
            **ranked_blocks(scores_by_rotation[0], list(range(16))),
        }
        # The action changes membership; ranking order cannot change stochastic masks
        # when two rules select the same set. Uniform retains its ordinary draw order.
        selected = {rule: sorted(ix) for rule, ix in selected.items()}
        rotations = [ranked_blocks(scores_by_rotation[i], list(range(16))) for i in range(3)]
        rotations = [{rule: sorted(ix) for rule, ix in row.items()} for row in rotations]
        selections = {rule: candidates[ix].tolist() for rule, ix in selected.items()}
        plans.append(
            {
                "state": name,
                "pool_group": state["pool_group"],
                "seed": state["seed"],
                "prefix": prefix,
                "current_indices": preview.last["current_indices"],
                "candidate_indices": candidates.tolist(),
                "selections": selections,
                "rotation_selections": [{rule: candidates[ix].tolist() for rule, ix in r.items()} for r in rotations],
            }
        )
        normalized = F.normalize(z[16:].double(), dim=1)
        for rule, ix in selected.items():
            full = torch.cat((current, memory[ix]))
            observations = [ssl_directions(method, saved["optimizer"], full, seed) for seed in OBSERVATION_SEEDS]
            observation_steps += len(observations)
            delta = torch.stack([o[1] for o in observations]).mean(0)
            gradient_vectors = [-o[0] / 1e-4 for o in observations]
            cosines = [
                float(F.cosine_similarity(a, b, dim=0))
                for a, b in zip(current_gradients, gradient_vectors, strict=True)
            ]
            features.append(
                {
                    "state": name,
                    "pool_group": state["pool_group"],
                    "rule": rule,
                    "features": {
                        "selected_reconstruction": float(losses[ix].mean()),
                        "virtual_ssl_loss": float(np.mean([o[2]["ssl_loss"] for o in observations])),
                        "selected_coverage": float(
                            (1 - normalized @ normalized[ix].T).clamp_min(0).min(dim=1).values.mean()
                        ),
                        "selected_variance": float(normalized[ix].var(dim=0).mean()),
                        "ssl_gradient_norm": float(np.mean([float(g.norm()) for g in gradient_vectors])),
                        "current_ssl_cosine": float(np.mean(cosines)),
                        **{f"health_{k}": v for k, v in health.items()},
                    },
                }
            )
            for rotation, gradient in enumerate(gradients):
                forecasts.append(
                    {
                        "state": name,
                        "rule": rule,
                        "rotation": rotation,
                        "predicted_ce_reduction": -float(torch.dot(gradient.double(), delta.double())),
                    }
                )
        if (
            not identical(saved["method"], method.state_dict())
            or not torch.equal(rng_before, torch.get_rng_state())
            or modes_before != [m.training for m in method.modules()]
            or any(not identical(g, p.grad) for g, p in zip(grads_before, method.parameters(), strict=True))
        ):
            raise ValueError("Predecision observation changed parent")
        torch.save(parent_features, run / f"features_parent_{name}.pt")
        atomic_json(
            run / "status.json", {"state": "observing", "completed_parents": position + 1, "completed_branches": 0}
        )
        print(f"Observed {position + 1}/{len(states)} parents", flush=True)
    atomic_json(run / "frozen_selections.json", {"plans": plans})
    atomic_json(run / "predecision_features.json", {"rows": features})
    pd.DataFrame(block_rows).to_csv(run / "block_forecasts.csv", index=False)
    pd.DataFrame(forecasts).to_csv(run / "batch_forecasts.csv", index=False)
    frozen = {
        n: digest(run / n)
        for n in ("frozen_selections.json", "predecision_features.json", "block_forecasts.csv", "batch_forecasts.csv")
    }
    atomic_json(
        run / "inputs.json",
        {
            "input_hashes": input_hashes,
            "source_hashes": source_hashes,
            "smoke": args.smoke,
            "parents": len(states),
            "pool_groups": sorted({s["pool_group"] for s in states}),
            "rules": RULES,
            "observation_seeds": OBSERVATION_SEEDS,
            "outcome_seeds": draws,
            "tolerances": TOLERANCES,
            "horizons": [1, 4],
            "primary_panel": "pooled",
            "primary_horizon": 4,
            "primary_tolerance": 1e-10,
            "minimum_gain": 0.001,
            "numerical_contrast_limit": 0.0001,
            "virtual_steps": observation_steps,
            "task_backward_observations": 3 * len(states),
            "current_ssl_backward_observations": 2 * len(states),
            "new_updates_expected": 4 * len(RULES) * len(draws) * len(states),
            "fresh_dataset_confirmation": False,
        },
    )
    atomic_json(run / "pretraining_verification.json", {"frozen_hashes": frozen, "completed_outcome_updates": 0})
    scores, branches = [], []

    def score(z: torch.Tensor, attrs: dict[str, Any]) -> None:
        with threadpool_limits(limits=1):
            for panel, support in supports.items():
                c = reader["selected_C"][panel]["accuracy_selected"]
                heads = {}
                for tolerance in TOLERANCES:
                    head = precise_probe(z, metadata["labels"], support, c, tolerance)
                    heads[str(tolerance)] = head
                    solver_rows.append(
                        {
                            **attrs,
                            "panel": panel,
                            "tolerance": tolerance,
                            "iterations": int(head.classifier.n_iter_.max()),
                        }
                    )
                    for role, indices in metadata["roles"].items():
                        scores.append(
                            {
                                **attrs,
                                "panel": panel,
                                "role": role,
                                "tolerance": tolerance,
                                **metrics(head, z, metadata["labels"], metadata["domains"], indices),
                            }
                        )
                torch.save(heads, run / f"heads_{attrs['case']}_{panel}.pt")
            for rotation, head in enumerate(task_heads[attrs["state"]]):
                for support in ROTATIONS[rotation][1:]:
                    scores.append(
                        {
                            **attrs,
                            "panel": f"task{rotation}",
                            "role": support,
                            "tolerance": 1e-10,
                            **metrics(head, z, metadata["labels"], metadata["domains"], supports[support]),
                        }
                    )

    for plan in plans:
        name = plan["state"]
        score(
            torch.load(run / f"features_parent_{name}.pt", weights_only=True),
            {
                "state": name,
                "pool_group": plan["pool_group"],
                "rule": "parent",
                "draw": -1,
                "horizon": 0,
                "case": f"parent_{name}",
            },
        )
    gates_frozen: dict[str, str] = {}
    for draw, seed in enumerate(draws):
        for plan in plans:
            name, saved = plan["state"], parents[plan["state"]]
            expected_memory = None
            expected_later = None
            for rule in RULES:
                method = copy.deepcopy(initial)
                method.load_state_dict(saved["method"])
                opt = optimizer_for(method)
                opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
                selector = FixedReplaySelection(plan["selections"][rule], plan["prefix"], plan["seed"])
                selector.restore_memory(saved["actuator"])
                decisions, signatures = [], []
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + plan["seed"])
                    for offset in range(4):
                        step = plan["prefix"] + offset
                        record = update(method, opt, images[orders[name][step * 32 : (step + 1) * 32]], step, selector)
                        decisions.append({**record, **selector.last})
                        signatures.append(
                            {
                                "buffer": tensor_hash(torch.stack(selector.reservoir._buffer)),
                                "rng": tensor_hash(selector.reservoir._gen.get_state()),
                                "seen": selector.reservoir._seen,
                            }
                        )
                        if offset + 1 in (1, 4):
                            case = f"{name}__draw{draw}__{rule}__h{offset + 1}"
                            z = isolated_embed(method, readout_images)
                            torch.save(z, run / f"features_{case}.pt")
                            score(
                                z,
                                {
                                    "state": name,
                                    "pool_group": plan["pool_group"],
                                    "rule": rule,
                                    "draw": draw,
                                    "horizon": offset + 1,
                                    "case": case,
                                },
                            )
                    torch.save(
                        {"method": method.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state()},
                        run / f"checkpoint_{name}__draw{draw}__{rule}.pt",
                    )
                later = [{k: r[k] for k in ("current_indices", "replay_indices")} for r in decisions[1:]]
                if expected_memory is not None and (signatures != expected_memory or later != expected_later):
                    raise ValueError("First selection altered memory or subsequent image draws")
                expected_memory, expected_later = signatures, later
                branches.append(
                    {
                        "state": name,
                        "pool_group": plan["pool_group"],
                        "rule": rule,
                        "draw": draw,
                        "decisions": decisions,
                        "memory_signatures": signatures,
                    }
                )
                pd.DataFrame(scores).to_csv(run / "scores.csv", index=False)
                pd.DataFrame(solver_rows).to_csv(run / "solver.csv", index=False)
                atomic_json(run / "branches.json", {"branches": branches})
                atomic_json(
                    run / "status.json",
                    {
                        "state": "training",
                        "draw": draw,
                        "completed_branches": len(branches),
                        "expected_branches": len(states) * len(RULES) * len(draws),
                    },
                )
                print(f"Collected {len(branches)}/{len(states) * len(RULES) * len(draws)} branches", flush=True)
        if draw == 0:
            atomic_json(run / "frozen_health_gates.json", freeze_health_gates(features, scores, target_horizon=4))
            gates_frozen = {"frozen_health_gates.json": digest(run / "frozen_health_gates.json")}
            atomic_json(
                run / "health_pretraining_verification.json",
                {"frozen_hashes": gates_frozen, "completed_draw": 0, "evaluation_branches": 0},
            )
        for filename, expected in {**frozen, **gates_frozen}.items():
            if digest(run / filename) != expected:
                raise ValueError(f"Frozen decision changed: {filename}")
    for path, expected in {**input_hashes, **source_hashes}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input/source changed during study: {path}")
    atomic_json(
        run / "verification.json",
        {
            "branches": len(branches),
            "new_optimizer_updates": 4 * len(branches),
            "virtual_steps": observation_steps,
            "frozen_hashes": {**frozen, **gates_frozen},
            "elapsed_seconds": time.time() - started,
            "input_hashes": input_hashes,
            "source_hashes": source_hashes,
            "artifact_hashes": {
                p.name: digest(p)
                for p in run.iterdir()
                if p.is_file() and p.name not in ("verification.json", "status.json")
            },
        },
    )
    atomic_json(
        run / "status.json",
        {"state": "complete", "completed_branches": len(branches), "new_optimizer_updates": 4 * len(branches)},
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
