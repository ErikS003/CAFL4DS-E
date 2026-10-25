"""Replay saved interventions with recognition readouts and pre-decision alpha diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.image_utility import FixedProbe, branch_outcomes, embed, reconstruction_draws, utility_features
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.utility_context import HISTORY, arrival_context
from cafl4ds.utility_diagnostics import alpha_geometry, diagnostic_outcomes, implemented_keep
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_image_utility_experiment import identical, restore_bdd_inputs
from scripts.run_slow_edge_pilot import ROOT, optimizer_for


def verify_fixed(actual: dict[Any, Any], expected: dict[Any, Any]) -> float:
    """Check old fixed-head outcomes without requiring newly added diagnostic keys to exist."""
    maximum = 0.0
    for key, value in expected.items():
        actual_key = int(key) if isinstance(key, str) and key.isdecimal() and key not in actual else key
        if isinstance(value, dict):
            maximum = max(maximum, verify_fixed(actual[actual_key], value))
        elif key.startswith("fixed_ce_"):
            error = abs(float(actual[key]) - float(value))
            if error > 1e-10:
                raise ValueError(f"Replayed fixed CE changed: {key}, error={error}")
            maximum = max(maximum, error)
    return maximum


def full_features(
    method: SSLMethod,
    actuator: SteerableReplay,
    batch: torch.Tensor,
    evaluation: Evaluation,
    roles: dict[str, torch.Tensor],
    warm_monitor: torch.Tensor,
    original: list[dict[str, Any]],
    step: int,
) -> list[dict[str, Any]]:
    """Keep the original eight-image observation group; add three fixed groups for other arrivals."""
    candidates = [row["candidate"] for row in original]
    order = [*candidates, *[index for index in range(32) if index not in candidates]]
    signals = []
    for start in range(0, 32, 8):
        current, _ = utility_features(
            method,
            actuator,
            batch[order[start : start + 8]],
            evaluation.images[roles["monitor"]],
            warm_monitor,
            draws=3,
        )
        for signal in current:
            signal.update(
                {
                    key: value
                    for key, value in original[0]["features"].items()
                    if key.startswith("health_") and key.endswith("_trend")
                }
            )
        signals.extend(current)
    arrival_context(method, actuator, batch, order, signals, step, None, 3)
    for signal in signals:
        signal.update({key: original[0]["features"][key] for key in HISTORY})
    for signal, row in zip(signals[:8], original, strict=True):
        if signal != row["features"]:
            raise ValueError("Full-arrival reader changed an original candidate feature")
    return sorted(
        [
            {"seed": original[0]["seed"], "step": step, "candidate": index, "features": signal}
            for index, signal in zip(order, signals, strict=True)
        ],
        key=lambda row: row["candidate"],
    )


def authority(
    method: SSLMethod,
    actuator: SteerableReplay,
    batch: torch.Tensor,
    seed: int,
    step: int,
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure reachable current diets; never train or use outcome labels to pick an action."""
    clusters = actuator._assign(batch)
    coverage = (actuator.target - actuator.occupancy).clamp_min(0)[clusters]
    score_seed = (seed + 104729 * (step + 1)) % (2**63 - 1)
    draws = reconstruction_draws(method, batch, 3, score_seed)
    geometry = alpha_geometry(coverage, draws[0], 16)
    fixed = {}
    for alpha in (0.0, 0.5, 1.0):
        clone = copy.deepcopy(actuator)
        clone.set_action(alpha)
        clone.select(StreamBatch(batch, era=0, step=step), FilterContext(method, step))
        keep = implemented_keep(coverage, draws[0], alpha, 16)
        if keep != clone.last["current_indices"]:
            raise ValueError("Alpha geometry disagrees with the implemented filter")
        fixed[str(alpha)] = keep
    single, averaged = set(fixed["0.0"]), set(implemented_keep(coverage, draws.mean(0), 0.0, 16))
    return {
        "seed": seed,
        "step": step,
        "features": signals,
        "coverage": coverage.tolist(),
        "single_draw_reconstruction": draws[0].tolist(),
        "three_draw_reconstruction": draws.mean(0).tolist(),
        "coverage_span": float(coverage.max() - coverage.min()),
        "geometry": geometry,
        "fixed_alpha_sets": fixed,
        "implemented_filter_checks": 3,
        "single_three_draw_top16_jaccard": len(single & averaged) / len(single | averaged),
        "feature_groups": "original eight candidates first; remaining arrival indices in three groups of eight",
        "history": "reuse the recorded pre-decision history/trends; recompute current measurements",
    }


def main() -> None:  # noqa: C901 - explicit replay, provenance and equal-work checks
    """Collect diagnostics in a new directory while preserving all original scientific artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Replay only the first state for engineering verification")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.set_num_threads(4)
    torch.manual_seed(62000)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    atomic_json(output / "progress.json", {"state": "loading"})
    manifest = json.loads((source / "inputs.json").read_text())
    config = manifest["configuration"]
    seeds = config["seeds"][:1] if args.smoke else config["seeds"]
    steps = config["snapshot_steps"][:1] if args.smoke else config["snapshot_steps"]
    planned = len(seeds) * len(steps) * config["repeats"] * config["candidates"]
    if config["data"] != "bdd" or config["candidates"] != 8 or config["horizons"] != [1, 4]:
        raise ValueError("This saved-state protocol requires the eight-candidate BDD confirmation")
    paths = [
        source / "inputs.json",
        source / "dataset.json",
        source / "fixed_probe.pt",
        *sorted(source.glob("seed_*/snapshot_*.pt")),
        *sorted(source.glob("seed_*/context_*.pt")),
    ]
    input_hashes = {str(path): digest(path) for path in paths}
    science = [
        *sorted((ROOT / "cafl4ds").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
    ]
    hashes = {str(path): digest(path) for path in science}
    atomic_json(
        output / "inputs.json",
        {
            "source": str(source),
            "input_hashes": input_hashes,
            "source_hashes": hashes,
            "planned_rows": planned,
            "planned_states": len(seeds) * len(steps),
            "horizons": [1, 4],
            "smoke": args.smoke,
            "fixed_ce_reproduction_tolerance": 1e-10,
            "limits": [
                "Post-confirmation diagnostics using shared saved states and evaluation corpus",
                "No new selector fit, qualification nomination, deployed action or MPC claim",
            ],
        },
    )
    try:
        initial, warm, arrivals, evaluation, _ = restore_bdd_inputs(manifest)
        del warm
        roles = {name: torch.tensor(indices) for name, indices in manifest["roles"].items()}
        warm_monitor = embed(initial, evaluation.images[roles["monitor"]])
        saved = torch.load(source / "fixed_probe.pt", weights_only=False)
        classifier = LogisticRegression(C=1, max_iter=1000, random_state=63000)
        classifier.coef_, classifier.intercept_, classifier.classes_ = [
            saved[key].numpy() for key in ("coef", "intercept", "classes")
        ]
        classifier.n_features_in_ = len(saved["mean"])
        probe = FixedProbe(saved["mean"].numpy(), saved["scale"].numpy(), classifier)
        dataset = json.loads((source / "dataset.json").read_text())["rows"]
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for row in dataset:
            grouped.setdefault((row["seed"], row["step"], row["repeat"]), []).append(row)
        records: list[dict[str, Any]] = []
        maximum_error = 0.0
        for seed in seeds:
            for step in steps:
                snapshot = torch.load(source / f"seed_{seed}/snapshot_{step}.pt", weights_only=False)
                method = copy.deepcopy(initial)
                method.load_state_dict(snapshot["method"])
                method.train()
                optimizer = optimizer_for(method)
                optimizer.load_state_dict(snapshot["optimizer"])
                actuator = SteerableReplay(
                    initial.encoder, torch.tensor(manifest["centroids"]), torch.tensor(manifest["target"]), seed=seed
                )
                actuator.load_state_dict(snapshot["actuator"])
                parent_rng = torch.get_rng_state().clone()
                batch = arrivals[step * 32 : (step + 1) * 32]
                features = full_features(
                    method, actuator, batch, evaluation, roles, warm_monitor, grouped[seed, step, 0], step
                )
                atomic_json(
                    output / f"authority_{seed}_{step}.json", authority(method, actuator, batch, seed, step, features)
                )

                def evaluate(trial: SSLMethod) -> dict[str, dict[str, float]]:
                    return diagnostic_outcomes(trial, evaluation, roles, probe)

                before = evaluate(method)
                for repeat in range(config["repeats"]):
                    originals = grouped[seed, step, repeat]
                    context = torch.load(source / f"seed_{seed}/context_{step}_{repeat}.pt", weights_only=False)
                    if tensor_hash(context["baseline"]) != originals[0]["context_sha256"]:
                        raise ValueError("Saved baseline tensor changed")
                    if [tensor_hash(t) for t in context["continuation"]] != originals[0]["continuation_sha256"]:
                        raise ValueError("Saved continuation tensors changed")
                    base = branch_outcomes(
                        method,
                        optimizer,
                        context["baseline"],
                        context["continuation"],
                        [1, 4],
                        snapshot["rng"],
                        evaluate,
                    )
                    maximum_error = max(maximum_error, verify_fixed(base, originals[0]["baseline"]))
                    context_rows = []
                    for old, choice, index in zip(
                        originals, context["choices"], context["candidate_indices"], strict=True
                    ):
                        if (
                            index != old["candidate"]
                            or not torch.equal(choice[:15], context["baseline"][:15])
                            or not torch.equal(choice[16:], context["baseline"][16:])
                        ):
                            raise ValueError("Only the candidate slot may change")
                        actual = branch_outcomes(
                            method, optimizer, choice, context["continuation"], [1, 4], snapshot["rng"], evaluate
                        )
                        maximum_error = max(maximum_error, verify_fixed(actual, old["candidate_outcomes"]))
                        maximum_error = max(maximum_error, verify_fixed(before, old["before"]))
                        gains = {
                            str(h): {
                                role: {name: base[h][role][name] - value for name, value in actual[h][role].items()}
                                for role in ("development", "holdout")
                            }
                            for h in (1, 4)
                        }
                        context_rows.append(
                            {**old, "before": before, "baseline": base, "candidate_outcomes": actual, "gain": gains}
                        )
                    atomic_json(output / f"rows_{seed}_{step}_{repeat}.json", {"rows": context_rows})
                    records.extend(context_rows)
                    atomic_json(
                        output / "progress.json",
                        {
                            "state": "collecting",
                            "seed": seed,
                            "step": step,
                            "rows": len(records),
                            "maximum_fixed_ce_error": maximum_error,
                        },
                    )
                    print(f"seed={seed} step={step} repeat={repeat}: {len(records)} rows", flush=True)
                if not torch.equal(parent_rng, torch.get_rng_state()) or not identical(
                    method.state_dict(), snapshot["method"]
                ):
                    raise ValueError("Diagnostic readers or branches changed the source state/RNG")
                if not identical(optimizer.state_dict(), snapshot["optimizer"]) or not identical(
                    actuator.state_dict(), snapshot["actuator"]
                ):
                    raise ValueError("Diagnostics changed optimizer or actuator state")
        if len(records) != planned:
            raise ValueError("Incomplete diagnostic replay")
        if any(digest(Path(path)) != expected for path, expected in {**input_hashes, **hashes}.items()):
            raise ValueError("A source input or scientific file changed during diagnostic replay")
        atomic_json(output / "dataset.json", {"rows": records})
        atomic_json(
            output / "progress.json",
            {
                "state": "completed",
                "rows": len(records),
                "maximum_fixed_ce_error": maximum_error,
                "elapsed_seconds": time.time() - started,
                "optimizer_images_per_branch_horizon": {"1": 32, "4": 128},
            },
        )
    except Exception as error:
        atomic_json(output / "progress.json", {"state": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
