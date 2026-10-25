"""Collect pre-decision signals and paired image-replacement outcomes, without fitting a selector."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import transforms

from cafl4ds.artifacts import atomic_json
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.steerable import SteerableReplay, fit_anchors
from cafl4ds.image_utility import (
    FixedProbe,
    branch_outcomes,
    embed,
    outcome_losses,
    replacement_batches,
    split_query_roles,
    utility_features,
)
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.utility_context import ArrivalHistory, arrival_context
from scripts.confirm_last_block import tensor_hash
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_adaptation_experiment import _method
from scripts.run_slow_edge_pilot import ROOT, load_inputs, optimizer_for, update


def restore_bdd_inputs(
    manifest: dict[str, Any],
) -> tuple[SSLMethod, torch.Tensor, torch.Tensor, Evaluation, dict[str, Any]]:
    """Decode the recorded image identities and verify their original tensor hashes."""
    config = OmegaConf.create(manifest["config"])
    if digest(Path(config.well)) != manifest["well_sha256"]:
        raise ValueError("Recovery warm well changed")
    initial = _method(config)
    resize = transforms.Compose(
        [transforms.Resize((config.data.img_size, config.data.img_size)), transforms.ToTensor()]
    )

    def decode(paths: list[str]) -> torch.Tensor:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(resize(image.convert("RGB")))
        return torch.stack(images)

    warm = decode(manifest["warm_paths"])
    arrivals = decode([manifest["train_paths"][i] for i in manifest["train_source_order"]])
    saved = manifest["evaluation"]
    images = decode([manifest["validation_paths"][i] for i in saved["original_indices"]])
    labels, domains = [], []
    for key, cell in saved["cell_counts"].items():
        domain, label = map(int, key.split("_"))
        count = cell["support_pool"] + cell["query"]
        labels.extend([label] * count)
        domains.extend([domain] * count)
    evaluation = Evaluation(
        images,
        torch.tensor(labels),
        torch.tensor(domains),
        torch.tensor(saved["query_positions"]),
        {int(key): torch.tensor(value) for key, value in saved["supports"].items()},
        saved,
    )
    for name, tensor in (("warm", warm), ("arrivals", arrivals), ("evaluation", images)):
        if tensor_hash(tensor) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Recovery decoded input mismatch: {name}")
    return initial, warm, arrivals, evaluation, manifest


def identical(left: object, right: object) -> bool:
    """Compare checkpoint trees exactly before reusing interrupted-run measurements."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(identical(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(identical(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def snapshot_grid(total_updates: int, horizons: list[int]) -> list[int]:
    """Cover stable periods and both sides of the three declared recurrence boundaries."""
    visit = total_updates // 4
    possible = [1, visit - 2, visit, 2 * visit - 2, 2 * visit, 3 * visit - 2, 3 * visit, total_updates - max(horizons)]
    return sorted({step for step in possible if 1 <= step <= total_updates - max(horizons)})


def continuation_batches(
    actuator: SteerableReplay, arrivals: torch.Tensor, step: int, horizon: int, seed: int, method: SSLMethod
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Draw identical replay/continuation images independent of the candidate intervention."""
    memory = copy.deepcopy(actuator.reservoir)
    memory.max_train_batch = 32
    memory._gen.manual_seed(seed)
    replay = memory._sample_replay()
    if replay is None or len(replay) != 16:
        raise ValueError("Snapshot needs 16 historical images")
    for image in arrivals[step * 32 : (step + 1) * 32]:
        memory._ingest(image)
    future = [
        memory.observe(arrivals[t * 32 : (t + 1) * 32], FilterContext(method, t))
        for t in range(step + 1, step + horizon)
    ]
    return replay, future


def main() -> None:  # noqa: C901 - explicit collection and artifact audit
    """Run configurable, local counterfactual development experiments in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["synthetic", "bdd"], default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse saved paired contexts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6301, 6307, 6311, 6317, 6323])
    parser.add_argument("--train-per-cell", type=int, default=256)
    parser.add_argument("--evaluation-per-cell", type=int, default=60)
    parser.add_argument("--snapshot-steps", type=int, nargs="+")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--observation-draws", type=int, default=3)
    parser.add_argument("--gradient-signals", action="store_true")
    parser.add_argument("--context-signals", action="store_true")
    parser.add_argument("--stream-seed", type=int, default=62003)
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--primary-only", action="store_true", help="Collect fixed-head CE outcomes; omit reconstruction outcomes"
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        raise ValueError("At least three distinct seeds required for fit/validation/test")
    if not 2 <= args.candidates <= 16 or args.repeats < 1 or args.observation_draws < 1:
        raise ValueError("Require 2..16 candidates and positive repeat/draw counts")
    horizons = sorted(set(args.horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("Horizons must be positive update counts")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.set_num_threads(4)
    torch.manual_seed(62000)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    source_files = [
        *sorted((ROOT / "cafl4ds").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "cafl4ds/configs").rglob("*.yaml")),
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    original_manifest = json.loads((out / "inputs.json").read_text()) if args.resume else None
    reference_manifest = json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    excluded_manifest = json.loads(args.exclude_manifest.read_text()) if args.exclude_manifest else None
    excluded_paths = set(excluded_manifest["train_paths"]) if excluded_manifest is not None else None
    coordinates = original_manifest or reference_manifest
    if original_manifest is not None:
        runner = "scripts/run_image_utility_experiment.py"
        for path, expected in original_manifest["source_hashes"].items():
            if Path(path).as_posix() != runner and hashes.get(path) != expected:
                raise ValueError(f"Cannot resume after scientific source changed: {path}")
    atomic_json(out / "progress.json", {"state": "loading"})
    try:
        if original_manifest is not None and args.data == "bdd":
            initial, warm, arrivals, evaluation, provenance = restore_bdd_inputs(original_manifest)
        else:
            initial, warm, arrivals, evaluation, provenance = load_inputs(
                args.data,
                train_per_cell=args.train_per_cell,
                evaluation_per_cell=args.evaluation_per_cell,
                stream_seed=args.stream_seed,
                excluded_paths=excluded_paths,
            )
        if excluded_paths and set(provenance.get("train_paths", [])) & excluded_paths:
            raise ValueError("New stream overlaps the excluded identification sample")
        if reference_manifest is not None:
            for key in ("well_sha256", "warm_sha256", "evaluation_sha256", "evaluation"):
                if json.loads(json.dumps(provenance[key])) != reference_manifest[key]:
                    raise ValueError(f"Frozen reference input mismatch: {key}")
            provenance["reference_manifest_sha256"] = digest(args.reference_manifest)
        if args.exclude_manifest:
            provenance["excluded_manifest_sha256"] = digest(args.exclude_manifest)
        roles = split_query_roles(evaluation)
        if coordinates is None:
            centers, target = fit_anchors(embed(initial, warm), seed=62000)
        else:
            centers = torch.tensor(coordinates["centroids"])
            target = torch.tensor(coordinates["target"])
        del warm
        warm_monitor = embed(initial, evaluation.images[roles["monitor"]])
        if coordinates is None:
            probe = FixedProbe.fit(initial, evaluation)
        else:
            probe_path = (
                out / "fixed_probe.pt"
                if original_manifest is not None
                else args.reference_manifest.parent / "fixed_probe.pt"
            )
            saved_probe = torch.load(probe_path, weights_only=False)
            classifier = LogisticRegression(C=1, max_iter=1000, random_state=63000)
            classifier.coef_ = saved_probe["coef"].numpy()
            classifier.intercept_ = saved_probe["intercept"].numpy()
            classifier.classes_ = saved_probe["classes"].numpy()
            classifier.n_features_in_ = len(saved_probe["mean"])
            probe = FixedProbe(saved_probe["mean"].numpy(), saved_probe["scale"].numpy(), classifier)
        total = len(arrivals) // 32
        steps = sorted(set(args.snapshot_steps or snapshot_grid(total, horizons)))
        if not steps or min(steps) < 1 or max(steps) + max(horizons) > total:
            raise ValueError("Snapshot steps must leave the complete requested horizon")
        configuration = {
            **{key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "output": str(out),
            "snapshot_steps": steps,
            "horizons": horizons,
            "total_updates": total,
            "visit_updates": total // 4,
            "planned_rows": len(args.seeds) * len(steps) * args.candidates * args.repeats,
            "planned_intervention_updates": len(args.seeds)
            * len(steps)
            * (args.candidates + 1)
            * args.repeats
            * max(horizons),
        }
        manifest = {
            **provenance,
            "configuration": configuration,
            "source_hashes": hashes,
            "torch": torch.__version__,
            "roles": {name: indices.tolist() for name, indices in roles.items()},
            "centroids": centers.tolist(),
            "target": target.tolist(),
            "predictor_seed_roles": {"fit": args.seeds[:-2], "validation": args.seeds[-2], "test": args.seeds[-1]},
            "target_definition": "baseline held-out loss minus candidate held-out loss at equal optimizer work",
            "outcome_metrics": ["fixed_ce"] if args.primary_only else ["fixed_ce", "reconstruction"],
            "gradient_reference": "full balanced label-free monitor pool" if args.gradient_signals else None,
            "prefix_policy": "alpha=0.5; all arrivals admitted to uniform replay memory",
            "continuation_policy": "precomputed uniform 16 current + 16 historical, paired across branches",
            "limitations": [
                "Conditional single-slot replacement effects, not independent intrinsic image values",
                "Shared data, initial well and arrival order; seeds vary learning/randomization only",
                "Fixed-head CE includes coordinate drift; downstream refit accuracy remains a separate check",
                "Development outcome labels may fit the offline predictor; never enter SSL or candidate features",
                "No online selector efficacy or MPC claim; longer horizons may cross a domain boundary",
            ],
        }
        if original_manifest is not None:
            for key, expected in original_manifest["configuration"].items():
                if key != "resume" and configuration.get(key) != expected:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            for key in ("roles", "centroids", "target"):
                if manifest[key] != original_manifest[key]:
                    raise ValueError(f"Resume input mismatch: {key}")
            atomic_json(
                out / "resume.json",
                {"source_hashes": hashes, "verification": "exact prefix checkpoints, contexts and features"},
            )
        else:
            atomic_json(out / "inputs.json", manifest)
        probe_state = {
            "mean": torch.from_numpy(probe.mean),
            "scale": torch.from_numpy(probe.scale),
            "coef": torch.from_numpy(probe.classifier.coef_),
            "intercept": torch.from_numpy(probe.classifier.intercept_),
            "classes": torch.from_numpy(probe.classifier.classes_),
        }
        if args.resume:
            if not identical(probe_state, torch.load(out / "fixed_probe.pt", weights_only=False)):
                raise ValueError("Resume fixed probe mismatch")
        else:
            torch.save(probe_state, out / "fixed_probe.pt")
        print(json.dumps(configuration), flush=True)
        records: list[dict[str, Any]] = []
        reused_rows = 0
        feature_seconds = branch_seconds = 0.0
        for seed in args.seeds:
            torch.manual_seed(seed)
            common = copy.deepcopy(initial)
            optimizer = optimizer_for(common)
            actuator = SteerableReplay(initial.encoder, centers, target, seed=seed)
            directory = out / f"seed_{seed}"
            directory.mkdir(exist_ok=args.resume)
            previous_health: dict[str, float] | None = None
            previous_step = 0
            previous_context: ArrivalHistory | None = None
            for step in range(max(steps) + 1):
                batch = arrivals[step * 32 : (step + 1) * 32]
                if step in steps:
                    atomic_json(
                        out / "progress.json",
                        {"state": "collecting", "seed": seed, "step": step, "completed_rows": len(records)},
                    )
                    rng = torch.get_rng_state().clone()
                    snapshot = {
                        "method": copy.deepcopy(common.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "actuator": actuator.state_dict(),
                        "rng": rng,
                        "step": step,
                    }
                    snapshot_path = directory / f"snapshot_{step}.pt"
                    if snapshot_path.exists():
                        if not identical(snapshot, torch.load(snapshot_path, weights_only=False)):
                            raise ValueError(f"Resume prefix mismatch: {snapshot_path}")
                    else:
                        torch.save(snapshot, snapshot_path)
                    generator = torch.Generator().manual_seed(63000 + step)
                    candidates = torch.randperm(32, generator=generator)[: args.candidates].tolist()
                    tick = time.time()
                    signals, health = utility_features(
                        common,
                        actuator,
                        batch[candidates],
                        evaluation.images[roles["monitor"]],
                        warm_monitor,
                        draws=args.observation_draws,
                        gradient_signals=args.gradient_signals,
                    )
                    for signal in signals:
                        signal.update(
                            {
                                f"{key}_trend": 0.0
                                if previous_health is None
                                else (value - previous_health[key]) / (step - previous_step)
                                for key, value in health.items()
                            }
                        )
                    previous_health, previous_step = health, step
                    if args.context_signals:
                        previous_context = arrival_context(
                            common,
                            actuator,
                            batch,
                            candidates,
                            signals,
                            step,
                            previous_context,
                            args.observation_draws,
                        )
                    feature_seconds += time.time() - tick

                    def evaluate(method: SSLMethod) -> dict[str, dict[str, float]]:
                        return outcome_losses(
                            method,
                            evaluation,
                            roles,
                            probe,
                            args.observation_draws,
                            reconstruction=not args.primary_only,
                        )

                    before = evaluate(common)
                    for repeat in range(args.repeats):
                        tick = time.time()
                        draw_seed = seed * 10000 + step * 100 + repeat
                        replay, continuation = continuation_batches(
                            actuator, arrivals, step, max(horizons), draw_seed, common
                        )
                        baseline, choices, reference = replacement_batches(batch, replay, candidates, draw_seed + 1)
                        rows_path = directory / f"rows_{step}_{repeat}.json"
                        if args.resume and rows_path.exists():
                            saved = json.loads(rows_path.read_text())["rows"]
                            context = torch.load(directory / f"context_{step}_{repeat}.pt", weights_only=False)
                            if not identical(
                                context,
                                {
                                    "baseline": baseline,
                                    "choices": choices,
                                    "continuation": continuation,
                                    "reference": reference,
                                    "candidate_indices": candidates,
                                },
                            ):
                                raise ValueError(f"Resume context mismatch: {rows_path}")
                            if len(saved) != len(candidates) or any(
                                row["seed"] != seed
                                or row["step"] != step
                                or row["repeat"] != repeat
                                or row["candidate"] != candidate
                                or row["features"] != signal
                                or row["before"] != before
                                for row, candidate, signal in zip(saved, candidates, signals, strict=True)
                            ):
                                raise ValueError(f"Resume feature mismatch: {rows_path}")
                            records.extend(saved)
                            reused_rows += len(saved)
                            print(f"verified seed={seed} step={step} repeat={repeat}: {len(records)} rows", flush=True)
                            continue
                        torch.save(
                            {
                                "baseline": baseline,
                                "choices": choices,
                                "continuation": continuation,
                                "reference": reference,
                                "candidate_indices": candidates,
                            },
                            directory / f"context_{step}_{repeat}.pt",
                        )
                        base = branch_outcomes(common, optimizer, baseline, continuation, horizons, rng, evaluate)
                        context_rows = []
                        for candidate, first_batch, signal in zip(candidates, choices, signals, strict=True):
                            actual = branch_outcomes(
                                common, optimizer, first_batch, continuation, horizons, rng, evaluate
                            )
                            gains = {
                                str(horizon): {
                                    role: {
                                        key: base[horizon][role][key] - value
                                        for key, value in actual[horizon][role].items()
                                    }
                                    for role in ("development", "holdout")
                                }
                                for horizon in horizons
                            }
                            row = {
                                "seed": seed,
                                "step": step,
                                "visit": step // (total // 4),
                                "repeat": repeat,
                                "candidate": candidate,
                                "arrival_id": step * 32 + candidate,
                                "reference_candidate": reference,
                                "features": signal,
                                "before": before,
                                "baseline": base,
                                "candidate_outcomes": actual,
                                "gain": gains,
                                "optimizer_images_per_horizon": {str(h): 32 * h for h in horizons},
                                "context_sha256": tensor_hash(baseline),
                                "continuation_sha256": [tensor_hash(images) for images in continuation],
                            }
                            if "train_paths" in provenance:
                                row["image_path"] = provenance["train_paths"][
                                    provenance["train_source_order"][row["arrival_id"]]
                                ]
                            context_rows.append(row)
                        atomic_json(directory / f"rows_{step}_{repeat}.json", {"rows": context_rows})
                        records.extend(context_rows)
                        atomic_json(
                            out / "progress.json",
                            {
                                "state": "collecting",
                                "seed": seed,
                                "step": step,
                                "completed_rows": len(records),
                                "reused_rows": reused_rows,
                            },
                        )
                        branch_seconds += time.time() - tick
                        print(f"seed={seed} step={step} repeat={repeat}: {len(records)} rows", flush=True)
                    if not torch.equal(torch.get_rng_state(), rng):
                        raise ValueError("Observations/interventions changed prefix training RNG")
                # Branch experiments never affect this shared prefix trajectory.
                update(common, optimizer, batch, step, actuator)
        if any(digest(ROOT / path) != expected for path, expected in hashes.items()):
            raise ValueError("Experiment sources changed during collection")
        atomic_json(out / "dataset.json", {"rows": records})
        summary = {
            "state": "completed",
            "rows": len(records),
            "reused_rows": reused_rows,
            "timing_scope": "this attempt only; reused branch work excluded",
            "elapsed_seconds": time.time() - started,
            "feature_seconds": feature_seconds,
            "intervention_seconds": branch_seconds,
            "configuration": configuration,
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        atomic_json(out / "progress.json", {"state": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
