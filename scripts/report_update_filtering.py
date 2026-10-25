"""Independently audit arrival masks, saved states, predictions and half-rate decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure

from cafl4ds.artifacts import atomic_json
from scripts.audit_recurrence_retention import close, digest, interval, metrics, order_hash
from scripts.report_memory_admission import read_trajectory

SNAPSHOTS = (36, 108, 180, 252, 324, 396, 468, 540)
KINDS = ("knn", "linear", "finetune")


def independent_mask(policy: str, seed: int) -> list[bool]:
    """Reconstruct the frozen masks without importing the runtime selector."""
    if policy == "always":
        return [True for _ in range(576)]
    if policy == "fixed_half":
        return [step % 2 == 0 for step in range(1, 577)]
    if policy != "random_half":
        raise ValueError("Unknown policy")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(62003 + seed)
    result: list[bool] = []
    for _ in range(4):
        positions = set(torch.randperm(144, generator=rng).tolist()[:72])
        result.extend(position in positions for position in range(144))
    return result


def audit_ledger(  # noqa: C901 - complete independent identity/state/budget checks
    directory: Path, result: dict[str, Any], order: list[int], policy: str, seed: int
) -> None:
    """Reject noncausal identity reuse, wrong budgets, missing updates and fabricated clocks."""
    expected = independent_mask(policy, seed)
    mask = json.loads((directory / "mask.json").read_text())
    if mask != {"policy": policy, "seed": seed, "accepted": expected}:
        raise ValueError("Mask identity or timing differs")
    ledger = json.loads((directory / "ledger.json").read_text())["steps"]
    losses = json.loads((directory / "losses.json").read_text())["losses"]
    if len(ledger) != 576 or len(order) != 18432 or len(set(order)) != 18432:
        raise ValueError("Single-pass arrival budget differs")
    if result["arrival_order_sha256"] != order_hash(order):
        raise ValueError("Arrival order differs")
    count = 0
    for index, row in enumerate(ledger):
        count += int(expected[index])
        if (
            row["step"] != index + 1
            or row["current"] != order[32 * index : 32 * (index + 1)]
            or row["accepted"] != expected[index]
            or row["optimizer_steps"] != count
            or row["optimizer_images"] != count * 32
        ):
            raise ValueError("Causal arrival/update ledger differs")
        if not np.isfinite(row["online_seconds"]) or row["online_seconds"] < 0:
            raise ValueError("Invalid online timing")
    wanted = [index + 1 for index, accepted in enumerate(expected) if accepted]
    if [row["step"] for row in losses] != wanted or [row["optimizer_step"] for row in losses] != list(
        range(1, count + 1)
    ):
        raise ValueError("Skipped or missing optimizer observations")
    if not np.isfinite([[row["loss"], row["grad_norm"]] for row in losses]).all():
        raise ValueError("Nonfinite optimizer observations")
    budget = {
        "arrivals": 18432,
        "updates": count,
        "optimizer_images": count * 32,
        "reservoir_seen": 0,
        "reservoir_size": 0,
    }
    if result["budget"] != budget or not np.isfinite(result["encoder_delta_l2"]) or result["encoder_delta_l2"] <= 0:
        raise ValueError("Model/update budget differs")
    blocks = [sum(row["online_seconds"] for row in ledger[start : start + 144]) for start in range(0, 576, 144)]
    close(result["block_online_seconds"], blocks, "online block timings")
    close(result["online_seconds"], sum(blocks), "online total timing")
    if result["online_seconds"] <= 0:
        raise ValueError("No measured online work")
    for step in SNAPSHOTS:
        state = torch.load(
            directory / "decision_states" / f"arrival_{step:06d}.pt", map_location="cpu", weights_only=False
        )
        updates = sum(expected[:step])
        if (
            state["arrival_step"] != step
            or state["optimizer_steps"] != updates
            or state["mask"] != expected
            or state["arrival_order_sha256"] != order_hash(order)
            or state["torch_rng"].dtype != torch.uint8
            or state["torch_rng"].ndim != 1
        ):
            raise ValueError("Decision-state identity/RNG missing")
        if not state["optimizer"]["state"] or not state["model"]:
            raise ValueError("Incomplete decision state")
        for parameter in state["optimizer"]["state"].values():
            if int(parameter["step"]) != updates:
                raise ValueError("Saved optimizer clock differs")
            for value in parameter.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError("Nonfinite saved optimizer")
        if any(not torch.isfinite(value).all() for value in state["model"].values()):
            raise ValueError("Nonfinite saved model")
    if any(not (directory / "checkpoints" / f"boundary_{i}.pt").is_file() for i in range(1, 5)):
        raise ValueError("Missing boundary checkpoint")


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - independent complete study and frozen conjunction
    """Rescore every arm and expose every frozen gate without selecting outcomes."""
    locked = json.loads((out / "input_lock.json").read_text())
    for name, expected in locked["hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed scientific input: {name}")
    preflight = json.loads((out / "preflight.json").read_text())
    compatibility = json.loads((out / "compatibility.json").read_text())
    if not preflight["passed"] or not compatibility["passed"] or compatibility["optimizer_updates"] != 576:
        raise ValueError("Controls not qualified")
    prior = Path(locked["prior"])
    data = json.loads((out / "data_lock.json").read_text())
    seeds = locked["config"]["seeds"]
    draws = locked["config"]["recurrence"]["draw_seeds"]
    manifests = json.loads((out / "mask_manifest.json").read_text())
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        query = arrays["query"]
        labels, domains = arrays["labels"][query], arrays["domains"][query]
        if query.tolist() != data["evaluation"]["query_positions"]:
            raise ValueError("Query identities differ")
        for support in data["evaluation"]["supports"].values():
            if set(support) & set(query.tolist()) or np.bincount(arrays["labels"][support]).tolist() != [25, 25, 25]:
                raise ValueError("Support/query information budget differs")
    control_dir = out / "compatibility/recurrence_always" / f"seed_{seeds[0]}"
    control = read_trajectory(control_dir, out / "frozen", labels, domains, draws)
    audit_ledger(control_dir, control, data["orders"][str(seeds[0])]["recurrence"], "always", seeds[0])
    close(control["online_seconds"], compatibility["online_seconds"], "compatibility timing")
    old_dir = prior / "recurrence_no_replay" / f"seed_{seeds[0]}"
    old_control = json.loads((old_dir / "result.json").read_text())
    # Recheck exact compatibility from the original saved values before independent rescore tolerance.
    original_control = json.loads((control_dir / "result.json").read_text())
    if any(
        original_control[k] != old_control[k]
        for k in ("trajectory", "budget", "encoder_delta_l2", "arrival_order_sha256")
    ):
        raise ValueError("Complete compatibility values changed")
    for original in old_dir.glob("predictions_*.npz"):
        with np.load(original, allow_pickle=False) as a, np.load(control_dir / original.name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("Compatibility predictions changed")
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    timing: dict[str, list[dict[str, Any]]] = {}
    for schedule in ("recurrence", "shuffled"):
        rows[schedule] = {policy: [] for policy in ("fixed_half", "random_half", "always")}
        for seed in seeds:
            order = data["orders"][str(seed)][schedule]
            if set(order) != set(data["orders"][str(seed)]["shuffled"]):
                raise ValueError("Schedule populations differ")
            for policy in ("fixed_half", "random_half"):
                if manifests[str(seed)][policy] != independent_mask(policy, seed):
                    raise ValueError("Precollection mask manifest differs")
                directory = out / f"{schedule}_{policy}" / f"seed_{seed}"
                result = read_trajectory(directory, out / "frozen", labels, domains, draws)
                audit_ledger(directory, result, order, policy, seed)
                rows[schedule][policy].append(result)
                timing.setdefault(f"{schedule}_{policy}", []).append(
                    {
                        "seed": seed,
                        "online_seconds": result["online_seconds"],
                        "block_online_seconds": result["block_online_seconds"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
            rows[schedule]["always"].append(
                read_trajectory(
                    prior / f"{schedule}_no_replay" / f"seed_{seed}", prior / "frozen", labels, domains, draws
                )
            )
    contrasts: dict[str, Any] = {}
    guards: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    for schedule, policies in rows.items():
        contrasts[schedule] = {}
        guards[schedule] = {}
        quality[schedule] = {}
        for kind in KINDS:
            comparisons = {}
            for left, right in (("fixed_half", "always"), ("random_half", "always"), ("fixed_half", "random_half")):
                comparisons[f"{left}_minus_{right}"] = {
                    metric: interval(
                        [
                            metrics(a["trajectory"], 144, kind)[metric] - metrics(b["trajectory"], 144, kind)[metric]
                            for a, b in zip(policies[left], policies[right], strict=True)
                        ]
                    )
                    for metric in ("final_balanced", "interference", "bwt", "forgetting")
                }
            for policy in ("fixed_half", "random_half"):
                comparisons[f"{policy}_minus_frozen"] = interval(
                    [metrics(a["trajectory"], 144, kind)["final_vs_warm"] for a in policies[policy]]
                )
            contrasts[schedule][kind] = comparisons
        for policy in ("fixed_half", "random_half"):
            endpoints = []
            for step, domain in ((144, "A"), (288, "B"), (432, "A"), (576, "B")):
                values = []
                for a, b in zip(policies[policy], policies["always"], strict=True):
                    left = next(p for p in a["trajectory"] if p["step"] == step)["scores"]["finetune"][domain]
                    right = next(p for p in b["trajectory"] if p["step"] == step)["scores"]["finetune"][domain]
                    values.append(left - right)
                endpoints.append({"step": step, "domain": domain, **interval(values)})
            guards[schedule][policy] = endpoints
            primary = contrasts[schedule]["finetune"]
            quality[schedule][policy] = {
                "final_noninferior": primary[f"{policy}_minus_always"]["final_balanced"]["ci95_pp"][0] > -1,
                "current_domain_noninferior": all(g["ci95_pp"][0] > -1 for g in endpoints),
                "positive_advantage_over_frozen": primary[f"{policy}_minus_frozen"]["ci95_pp"][0] > 0,
            }
    matched = rows["recurrence"]["fixed_half"][0]["online_seconds"] / control["online_seconds"]
    gates = {
        **quality["recurrence"]["fixed_half"],
        "exact_half_update_budget": True,
        "online_time_reduced_at_least_25pct": matched <= 0.75,
    }
    result = {
        "decision": "half_rate_efficiency_lead" if all(gates.values()) else "no_primary_efficiency_lead",
        "primary_gates": gates,
        "quality_gates_by_policy": quality,
        "contrasts": contrasts,
        "current_domain_guards": guards,
        "timing_by_arm": timing,
        "matched_seed5003_timing": {
            "always_online_seconds": control["online_seconds"],
            "fixed_half_online_seconds": rows["recurrence"]["fixed_half"][0]["online_seconds"],
            "ratio": matched,
            "reduction_percent": 100 * (1 - matched),
        },
        "fixed_timing_advantage_established": contrasts["recurrence"]["finetune"]["fixed_half_minus_random_half"][
            "final_balanced"
        ]["ci95_pp"][0]
        > 0,
        "work": {
            "collection_updates": 5760,
            "collection_optimizer_images": 184320,
            "collection_arriving_images": 368640,
            "compatibility_updates": 576,
            "discarded_preflight_updates": 8,
            "copied_finetune_updates": 25500,
        },
        "scope": "Reused-data development; resident-data CPU timing, single matched control; no health-policy claim.",
    }
    atomic_json(out / "report.json", result)
    lines = [
        "# Half-rate SSL update filtering",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "Primary: recurrence/fixed-half, copied fine-tuning25 labels/class, five paired seeds.",
        "Reused images/evaluation/common warm model; development only.",
        "",
        "| Schedule | Readout | Contrast | Final gain pp [95% interval] |",
        "|---|---|---|---|",
    ]
    for schedule, readouts in contrasts.items():
        for kind, comparisons in readouts.items():
            for name, comparison in comparisons.items():
                value = comparison if name.endswith("frozen") else comparison["final_balanced"]
                lines.append(
                    f"| {schedule} | {kind} | {name} | {value['mean_pp']:+.3f} "
                    f"[{value['ci95_pp'][0]:+.3f}, {value['ci95_pp'][1]:+.3f}] |"
                )
    lines.extend(
        [
            "",
            f"Frozen primary gates: `{gates}`.",
            "",
            f"Matched seed5003 online-time reduction: {100 * (1 - matched):.1f}%.",
            "Timing excludes load/evaluation/research serialization; resident-data CPU work only.",
            "Single control timing is descriptive; block/arm variability is retained in report.json.",
            "",
            f"Fixed timing advantage over random established: {result['fixed_timing_advantage_established']}.",
            "A simple efficiency result does not establish health control or independent population transfer.",
            "Nominal five-seed intervals and endpoint guards are not simultaneous95% coverage.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, domain in zip(axes, ("A", "B"), strict=True):
        for policy, runs in rows["recurrence"].items():
            curves = (
                np.array(
                    [[p["scores"]["finetune"][domain] for p in r["trajectory"] if p["step"] % 144 == 0] for r in runs]
                )
                * 100
            )
            axis.plot([0, 144, 288, 432, 576], curves.mean(axis=0), marker="o", label=policy)
        warm = rows["recurrence"]["fixed_half"][0]["trajectory"][0]["scores"]["finetune"][domain]
        axis.axhline(100 * warm, color="black", linestyle="--", label="warm frozen")
        axis.set(title=f"{domain}: downstream fine-tuning", xlabel="Arriving batches", ylabel="Balanced accuracy (%)")
        axis.legend()
    figure.savefig(out / "trajectories.png", dpi=150)
    atomic_json(
        out / "independent_verification.json",
        {
            "passed": True,
            "new_arms_checked": 20,
            "prior_controls_rescored": 10,
            "compatibility_arms_checked": 1,
            "arrival_decisions_reconstructed": 11520,
            "decision_states_checked": 168,
            "input_hashes_unchanged": True,
            "report_sha256": digest(out / "report.json"),
            "scope": "Independent masks/ledgers, saved-output scores and inference; not model retraining.",
            "artifact_hashes": {
                str(p.relative_to(out)): digest(p)
                for p in out.rglob("*")
                if p.is_file()
                and "source_snapshot" not in p.parts
                and p.suffix in {".json", ".npz", ".pt", ".png", ".md"}
                and p.name not in {"status.json", "independent_verification.json"}
            },
        },
    )
    root = Path(__file__).resolve().parents[1]
    for folder in ("phase1", "latest-results"):
        (root / "docs/experiments" / folder / "slow-edge-update-filtering.md").write_text(
            (out / "report.md").read_text() + f"\nArtifacts: `{out.relative_to(root).as_posix()}`.\n", encoding="utf-8"
        )
    return result


def main() -> None:
    """Audit a complete collection without recipe changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(report(parser.parse_args().run.resolve()), indent=2))


if __name__ == "__main__":
    main()
