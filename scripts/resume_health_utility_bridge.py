"""Resume the frozen bridge grid after a host crash; preserve committed branches."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import embeddings
from cafl4ds.health_metric_derivatives import health_gradients, project_displacement
from cafl4ds.health_utility_analysis import analyze, fit_models, frozen_decisions
from cafl4ds.health_utility_bridge import ACTIONS, DRAWS, candidate_menu, extrema
from cafl4ds.prediction_benchmark import health_effect, health_values
from cafl4ds.recurrence import Evaluation
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import learner, reference_images
from scripts.run_health_warning import arrivals, readout_source
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import DRAWS as SUPPORTS, evaluate_gpu
from scripts.run_retention_anchor_signal import setup
from scripts.run_scheduling_experiment import scores
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
TIME = ROOT / "outputs/prediction-health-timeseries/20261014-pilot"
IMAGE = ROOT / "outputs/prediction-image-health/20261014-pilot-device-repair"


def read(path: Path) -> Any:
    """Read preserved finite JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def private(parent: dict[str, Any], point: int, draw: int) -> dict[str, Any]:
    """Share the immutable parent and reset only the explicitly private RNG fields."""
    return {
        **parent,
        "cpu_rng": torch.Generator().manual_seed(point + draw).get_state(),
        "cuda_rng": [torch.Generator(device="cuda").manual_seed(point + draw).get_state()],
    }


def make_preview(
    initial: Any,
    config: Any,
    state: dict[str, Any],
    batch: torch.Tensor,
    bank: torch.Tensor,
    warm: np.ndarray,
    gradients: Any,
    values: dict[str, float],
    names: list[str],
) -> dict[str, Any]:
    """Paid copied update: features use derivatives, never the observed endpoint health."""
    attempts = []
    timing = {}
    for repeat in range(2):
        tick = time.perf_counter()
        model, optimizer = restore(initial, config.optim, state)
        synchronize(torch.device("cuda"))
        clone_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        loss, norm = optimize_current(model, optimizer, batch.cuda(), True)
        synchronize(torch.device("cuda"))
        update_seconds = time.perf_counter() - tick
        end = capture(model, optimizer)
        marker = fingerprint(end)
        tick = time.perf_counter()
        z = embeddings(model, bank)
        synchronize(torch.device("cuda"))
        bank_seconds = time.perf_counter() - tick
        if fingerprint(capture(model, optimizer)) != marker:
            raise ValueError("Preview bank reading changed learner")
        attempts.append((loss, norm, marker, z))
        if repeat == 0:
            timing = {"clone_seconds": clone_seconds, "update_seconds": update_seconds, "bank_seconds": bank_seconds}
    if attempts[0][:3] != attempts[1][:3] or not np.array_equal(attempts[0][3], attempts[1][3]):
        raise ValueError("Preview complete-state replay differs")
    predicted, terms = project_displacement(gradients, names, end["model"], state["model"], values)
    return {
        "predicted": predicted,
        "projection_terms": terms,
        "actual_response_diagnostic": health_effect(values, health_values(z, warm), values),
        "loss": loss,
        "gradient_norm": norm,
        "fingerprint": fingerprint(end),
        "rgb_mean": float(batch.mean()),
        "rgb_std": float(batch.std()),
        "timing": timing,
    }


def train_outcome(
    initial: Any,
    config: Any,
    state: dict[str, Any],
    batch: torch.Tensor,
    following: torch.Tensor,
    bank: torch.Tensor,
    expected: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Exact paired four-update intervention; future continuation is shared across actions."""
    batches = [batch] + list(following.split(32))
    model, optimizer = restore(initial, config.optim, state)
    trace = []
    for i, selected in enumerate(batches):
        update = list(optimize_current(model, optimizer, selected.cuda(), True))
        trace.append(update)
        if i == 0 and (
            fingerprint(capture(model, optimizer)) != expected["fingerprint"]
            or update != [expected["loss"], expected["gradient_norm"]]
        ):
            raise ValueError("Actual first update differs from preview")
    marker = fingerprint(capture(model, optimizer))
    z = embeddings(model, bank)
    if marker != fingerprint(capture(model, optimizer)):
        raise ValueError("Endpoint observer changed learner")
    repeated, opt = restore(initial, config.optim, state)
    repeated_trace = [list(optimize_current(repeated, opt, selected.cuda(), True)) for selected in batches]
    if (
        repeated_trace != trace
        or fingerprint(capture(repeated, opt)) != marker
        or not np.array_equal(embeddings(repeated, bank), z)
    ):
        raise ValueError("Four-update state/embedding replay failed")
    return model, optimizer, z, {"trace": trace, "fingerprint": marker, "replay_passed": True}


def validate_progress(choices: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> None:
    """Accept only a committed prefix of the frozen intervention grid."""
    expected = [(p, d, a) for d in (0, 1) for p in range(18) for a in ACTIONS]
    actual = [(r["parent"], r["draw"], r["action"]) for r in outcomes]
    if actual != expected[: len(actual)] or len(actual) > 180:
        raise ValueError("Recovered outcomes are not a unique ordered grid prefix")
    if [c["parent"] for c in choices] != list(range(len(choices))) or len(choices) > 18:
        raise ValueError("Recovered menus are not an ordered parent prefix")
    if actual and max(p for p, _, _ in actual) >= len(choices):
        raise ValueError("Completed branch lacks its frozen candidate menu")


def preserve_json(path: Path, value: dict[str, Any]) -> None:
    """Reuse a completed phase artifact only when its contents remain identical."""
    if path.exists():
        if read(path) != value:
            raise ValueError("Completed phase artifact differs: " + str(path))
    else:
        atomic_json(path, value)


def main() -> None:  # noqa: C901 - frozen two-draw causal collection kept explicit
    """Draw0 collects labels; models/decisions are fixed before draw1 readouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "input_lock.json").exists():
        raise ValueError("Resume requires the original immutable lock")
    if shutil.disk_usage(out).free < 512 * 1024**2:
        raise ValueError("Launch storage reserve")
    setup()
    start = time.perf_counter()
    old = read(TIME / "input_lock.json")
    if not read(TIME / "independent-verification.json")["passed"]:
        raise ValueError("Dense trajectories not audited")
    cases = old["cases"]
    series = read(TIME / "series.json")["records"]
    if any(
        (s["stream"], s["checkpoint"]) != (c["stream"], c["checkpoint"]) for s, c in zip(series, cases, strict=True)
    ):
        raise ValueError("Temporal history and parent ordering differ")
    hashes = read(out / "input_lock.json")["hashes"]
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise ValueError("Original frozen inputs changed")
    recovered_choices = read(out / "choices.json")["records"]
    recovered_outcomes = read(out / "outcomes.json")["records"]
    validate_progress(recovered_choices, recovered_outcomes)
    recovery = out / f"recovery_{len(list(out.glob('recovery_*'))) + 1:02d}"
    recovery.mkdir()
    committed_hashes = {}
    for name in ("choices.json", "outcomes.json", "input_lock.json"):
        shutil.copyfile(out / name, recovery / name)
    for row in recovered_outcomes:
        directory = out / row["directory"]
        for name in ("endpoint.npz", "predictions.npz", "readout.json"):
            path = directory / name
            committed_hashes[str(path.relative_to(out))] = digest(path)
    recovery_sources = [
        Path(__file__),
        ROOT / "scripts/verify_health_utility_bridge_precision.py",
        ROOT / "tests/unit/test_health_utility_recovery.py",
    ]
    recovery_lock = {
        "reason": "User reported a host crash and authorized resuming the frozen grid.",
        "original_input_lock_sha256": digest(out / "input_lock.json"),
        "completed_branches_reused": len(recovered_outcomes),
        "completed_parent_menus_reused": len(recovered_choices),
        "committed_artifact_hashes": committed_hashes,
        "recovery_source_hashes": {str(p): digest(p) for p in recovery_sources},
        "additional_prefix_reconstruction_updates": 8 * len(recovered_choices),
        "interrupted_uncommitted_tail_updates_bound": [
            0,
            40
            if len(recovered_choices) < 18
            else 180
            if not (out / "observations1.json").exists() and len(recovered_outcomes) >= 90
            else 8,
        ],
        "lost_costs": "Original per-stage timings were held in RAM. Saved readout/preview timings survive; total pre-crash elapsed time and exact unfinished work are unavailable.",
        "scientific_design_changed": False,
    }
    atomic_json(recovery / "recovery_lock.json", recovery_lock)
    for path in recovery_sources:
        shutil.copyfile(path, recovery / path.name)
    config = OmegaConf.create(old["config"])
    bank = reference_images(old["reference"])
    warm = np.load(IMAGE / "warm.npy")
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    ancestor = ROOT / "outputs/regime-windows/20261013-gpu"
    with np.load(ancestor / "evaluation_arrays.npz") as a:
        data = Evaluation(
            torch.load(
                ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt",
                map_location="cpu",
                weights_only=True,
            ),
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in SUPPORTS},
            read(ancestor / "data-role-manifest.json"),
        )
    labels, domains = data.labels[data.query].numpy(), data.domains[data.query].numpy()
    torch.manual_seed(11)
    initial = _method(config).cuda()
    parents, holds, observations0, costs = [], [], [], []
    choices, outcomes = recovered_choices, recovered_outcomes
    completed = {(r["parent"], r["draw"], r["action"]) for r in outcomes}
    previews_count, outcome_updates = 32 * len(choices), 8 * len(outcomes)
    new_fits = 3 * sum(r["alias"] is None for r in outcomes)
    reused_fits = 3 * sum(r["alias"] is not None for r in outcomes)
    readout_cache = {
        (r["parent"], r["draw"], r["fingerprint"]): out / r["directory"] for r in outcomes if r["alias"] is None
    }

    def outcome(parent_index: int, draw_index: int, obs: dict[str, Any], action: str) -> None:
        nonlocal outcome_updates, new_fits, reused_fits
        if (parent_index, draw_index, action) in completed:
            return
        if shutil.disk_usage(out).free < 256 * 1024**2:
            raise ValueError("Readout storage reserve")
        case, state, images = cases[parent_index], parents[parent_index], source_images[parent_index]
        subset = choices[parent_index]["menus"][choices[parent_index]["selected"][action]]
        state = private(state, case["point"], DRAWS[draw_index])
        tick = time.perf_counter()
        model, optimizer, z, audit = train_outcome(
            initial, config, state, images[128:192][subset], images[192:288], bank, obs["previews"][action]
        )
        outcome_updates += 8
        directory = out / f"runs/parent_{parent_index:02d}/draw_{draw_index}/{action}"
        if directory.exists():
            raise ValueError("Uncommitted directory requires separate preservation before retry")
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "endpoint.npz", embeddings=z)
        cache_key = (parent_index, draw_index, audit["fingerprint"])
        alias = readout_cache.get(cache_key)
        if alias is None:
            readout = evaluate_gpu(model, optimizer, data, directory)
            readout_cache[cache_key] = directory
            new_fits += 3
        else:
            shutil.copyfile(alias / "predictions.npz", directory / "predictions.npz")
            shutil.copyfile(alias / "readout.json", directory / "readout.json")
            readout = read(directory / "readout.json")
            reused_fits += 3
            with np.load(alias / "endpoint.npz") as previous:
                if not np.array_equal(z, previous["embeddings"]):
                    raise ValueError("Readout alias embedding differs")
        outcomes.append(
            {
                "parent": parent_index,
                "stream": case["stream"],
                "checkpoint": case["checkpoint"],
                "draw": draw_index,
                "action": action,
                "selected_slots": subset,
                "directory": str(directory.relative_to(out)),
                "alias": str(alias.relative_to(out)) if alias else None,
                "panels": readout["panels"],
                "scores": scores(out, directory, labels, domains),
                **audit,
            }
        )
        costs.append(
            {
                "stage": "outcome_and_readout",
                "parent": parent_index,
                "draw": draw_index,
                "action": action,
                "seconds": time.perf_counter() - tick,
                "new_primary_fits": 0 if alias else 3,
            }
        )
        completed.add((parent_index, draw_index, action))
        atomic_json(out / "outcomes.json", {"complete": False, "records": outcomes})
        atomic_json(recovery / "costs_partial.json", {"records": costs, "elapsed_seconds": time.perf_counter() - start})

    source_images = []
    for i, case in enumerate(cases):
        if shutil.disk_usage(out).free < 256 * 1024**2:
            raise ValueError("Per-parent storage reserve")
        images = arrivals(case, intake, {})
        source_images.append(images[:288])
        model, optimizer = restore(initial, config.optim, learner(Path(case["parent"])))
        prefix = [list(optimize_current(model, optimizer, b.cuda(), True)) for b in images[:128].split(32)]
        state = capture(model, optimizer)
        if prefix != case["trace"][:4] or fingerprint(state) != case["landmarks"]["4"]["fingerprint"]:
            raise ValueError("Decision parent differs from original")
        replay, opt = restore(initial, config.optim, learner(Path(case["parent"])))
        for b in images[:128].split(32):
            optimize_current(replay, opt, b.cuda(), True)
        if fingerprint(capture(replay, opt)) != fingerprint(state):
            raise ValueError("Common prefix replay differs")
        parents.append(state)
        if i < len(recovered_choices):
            previous = choices[i]
            if fingerprint(state) != previous["parent_fingerprint"]:
                raise ValueError("Recovered menu parent differs")
            previews, selected = previous["preview_records"], previous["selected"]
        else:
            marker = fingerprint(capture(model, optimizer))
            tick = time.perf_counter()
            gradients, values = health_gradients(model, bank, torch.from_numpy(warm))
            if fingerprint(capture(model, optimizer)) != marker:
                raise ValueError("Health derivative changed parent")
            costs.append({"stage": "health_gradients", "parent": i, "seconds": time.perf_counter() - tick})
            names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
            menu = candidate_menu(i)
            state0 = private(state, case["point"], DRAWS[0])
            previews = [
                make_preview(initial, config, state0, images[128:192][selected], bank, warm, gradients, values, names)
                for selected in menu
            ]
            previews_count += 32
            selected = extrema(previews)
            choices.append(
                {
                    "parent": i,
                    "menus": menu,
                    "selected": selected,
                    "preview_records": previews,
                    "parent_values": values,
                    "parent_fingerprint": marker,
                }
            )
            atomic_json(out / "choices.json", {"complete": False, "records": choices})
        source = readout_source(Path(case["landmarks"]["4"]["readout"]))
        holds.append(
            {
                "parent": i,
                "source": str(source),
                "scores": scores(Path("."), source, labels, domains),
                "panels": read(source / "readout.json")["panels"],
            }
        )
        obs = {
            "parent": i,
            "stream": case["stream"],
            "age": case["parent_updates"] + 4,
            "history": series[i]["health"][:5],
            "previews": {a: previews[selected[a]] for a in ACTIONS},
        }
        observations0.append(obs)
        for action in ACTIONS:
            outcome(i, 0, obs, action)
        print(
            json.dumps({"stage": "development", "parents": i + 1, "total": 18, "seconds": time.perf_counter() - start}),
            flush=True,
        )
    atomic_json(out / "choices.json", {"complete": True, "records": choices})
    preserve_json(out / "holds.json", {"records": holds})
    preserve_json(out / "observations0.json", {"records": observations0})
    expected_models = fit_models(observations0, [r for r in outcomes if r["draw"] == 0], holds, series)
    if (out / "models.json").exists():
        models = read(out / "models.json")
        if models != expected_models:
            raise ValueError("Existing frozen models differ")
    else:
        models = expected_models
        atomic_json(out / "models.json", models)
    observations1 = []
    cached_observations1 = read(out / "observations1.json")["records"] if (out / "observations1.json").exists() else []
    for i, case in enumerate(cases):
        if cached_observations1:
            observations1.append(cached_observations1[i])
            previews_count += 10
            continue
        model, optimizer = restore(initial, config.optim, parents[i])
        marker = fingerprint(capture(model, optimizer))
        tick = time.perf_counter()
        gradients, values = health_gradients(model, bank, torch.from_numpy(warm))
        if fingerprint(capture(model, optimizer)) != marker:
            raise ValueError("Evaluation derivative changed parent")
        costs.append({"stage": "health_gradients_draw1", "parent": i, "seconds": time.perf_counter() - tick})
        names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
        state = private(parents[i], case["point"], DRAWS[1])
        previews = {
            a: make_preview(
                initial,
                config,
                state,
                source_images[i][128:192][choices[i]["menus"][choices[i]["selected"][a]]],
                bank,
                warm,
                gradients,
                values,
                names,
            )
            for a in ACTIONS
        }
        previews_count += 10
        observations1.append({**observations0[i], "previews": previews})
    preserve_json(out / "observations1.json", {"records": observations1})
    decisions = frozen_decisions(observations1, models)
    preserve_json(
        out / "evaluation_decisions.json",
        {
            "frozen_before_draw1_outcomes": True,
            "records": decisions,
            "model_sha256": digest(out / "models.json"),
            "observation_sha256": digest(out / "observations1.json"),
        },
    )
    for i, obs in enumerate(observations1):
        for action in ACTIONS:
            outcome(i, 1, obs, action)
        print(
            json.dumps({"stage": "evaluation", "parents": i + 1, "total": 18, "seconds": time.perf_counter() - start}),
            flush=True,
        )
    atomic_json(out / "outcomes.json", {"complete": True, "records": outcomes})
    summary = analyze(outcomes, holds, decisions)
    atomic_json(out / "summary.json", summary)
    total = 144 + previews_count + outcome_updates
    atomic_json(
        out / "costs.json",
        {
            "records": costs,
            "common_prefix_updates": 144,
            "preview_and_replay_updates": previews_count,
            "outcome_and_replay_updates": outcome_updates,
            "total_experiment_updates": total,
            "new_primary_finetune_fits": new_fits,
            "reused_duplicate_fits": reused_fits,
            "reused_hold_fits": 54,
            "offline_finetune_optimizer_steps": new_fits * 100,
            "elapsed_seconds": time.perf_counter() - start,
            "elapsed_seconds_scope": "This recovery process only; pre-crash total unavailable",
            "recovery_additional_prefix_updates": recovery_lock["additional_prefix_reconstruction_updates"],
            "interrupted_tail_updates_bound": recovery_lock["interrupted_uncommitted_tail_updates_bound"],
            "total_update_lower_bound_including_recovery": total
            + recovery_lock["additional_prefix_reconstruction_updates"],
            "reused_committed_branches": recovery_lock["completed_branches_reused"],
            "efficiency_advantage_claimed": False,
        },
    )
    if total != 2340 or new_fits + reused_fits != 540 or len(outcomes) != 180:
        raise ValueError("Scientific work/grid mismatch")
    if any(digest(Path(p)) != expected for p, expected in hashes.items()):
        raise ValueError("Locked sources changed")
    if any(digest(out / p) != h for p, h in committed_hashes.items()):
        raise ValueError("Previously committed results changed during recovery")
    if any(digest(Path(p)) != h for p, h in recovery_lock["recovery_source_hashes"].items()):
        raise ValueError("Recovery sources changed during collection")
    atomic_json(
        recovery / "verification.json",
        {
            "passed": True,
            "committed_results_unchanged": len(recovered_outcomes[: recovery_lock["completed_branches_reused"]]),
            "original_hashes_unchanged": True,
        },
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 96 * 1024**2:
        raise ValueError("Output cap")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "paired_branches": 180,
            "online_preview_replay_updates": total,
            "decision_freeze_verified": True,
            "output_mib": size / 1024**2,
            "original_parent_matches": 18,
        },
    )
    print(json.dumps({"complete": True, "bridge_qualified": summary["bridge_qualified"]}), flush=True)


if __name__ == "__main__":
    main()
