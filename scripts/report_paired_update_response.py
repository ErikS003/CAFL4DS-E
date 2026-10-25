"""Independently audit paired arrival windows and classify prespecified phase preferences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.audit_recurrence_retention import close, digest, order_hash, scores

OFFSETS = (36, 108, 180, 252, 324, 396, 468, 540)
KINDS = ("knn", "linear", "finetune")


def state_hash(value: Any) -> str:  # noqa: ANN401 - independent nested state encoding
    """Reconstruct restoration fingerprints, including scalar optimizer tensors."""

    def encode(item: Any) -> Any:  # noqa: ANN401 - recursive JSON/tensor state
        if isinstance(item, torch.Tensor):
            h = hashlib.sha256(str((tuple(item.shape), item.dtype)).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
            return {"tensor": h.hexdigest()}
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return list(map(encode, item))
        return item

    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


def bands(values: list[float]) -> dict[str, Any]:
    """Compute nominal and 24-contrast simultaneous conditional five-seed intervals."""
    sample = np.asarray(values, dtype=float) * 100
    if sample.shape != (5,) or not np.isfinite(sample).all():
        raise ValueError("Expected five finite paired seed differences")
    mean, se = float(sample.mean()), float(sample.std(ddof=1) / np.sqrt(5))
    width = float(t.ppf(0.975, 4)) * se
    adjusted = float(t.ppf(1 - 0.05 / 48, 4)) * se
    return {
        "mean_pp": mean,
        "ci95_pp": [mean - width, mean + width],
        "simultaneous95_pp": [mean - adjusted, mean + adjusted],
        "paired_seed_pp": sample.tolist(),
    }


def preference(primary: dict[str, Any], timing_ratio: float) -> str:
    """Require all adjusted quality guardrails and real cost savings for half preference."""
    safe = all(primary[name]["simultaneous95_pp"][0] > -1 for name in ("balanced", "A", "B"))
    required = any(primary[name]["simultaneous95_pp"][1] < -1 for name in ("balanced", "A", "B"))
    if safe and timing_ratio <= 0.75:
        return "half_efficiency_preferred"
    if required:
        return "full_quality_required"
    return "unresolved"


def verify_terminal(
    directory: Path, point: dict[str, Any], labels: np.ndarray, domains: np.ndarray, draws: list[int]
) -> None:
    """Rescore raw predictions without trusting the collector's accuracy implementation."""
    with np.load(directory / f"predictions_{point['step']:06d}.npz", allow_pickle=False) as predicted:
        for kind in KINDS:
            panels = []
            for draw in draws:
                computed = scores(predicted[f"{draw}_{kind}"], labels, domains)
                panels.append(computed)
                for metric, value in computed.items():
                    close(point["panels"][str(draw)][kind][metric], value, "per-support terminal scoring")
            for metric in ("A", "B", "balanced", "micro"):
                actual = float(np.mean([p[metric] for p in panels]))
                close(point["scores"][kind][metric], actual, "averaged terminal scoring")
                point["scores"][kind][metric] = actual
    if set(point["fine_tuning_audits"]) != set(map(str, draws)) or any(
        record["steps"] != 100 or not np.isfinite(record["backbone_delta_l2"]) or record["backbone_delta_l2"] <= 0
        for record in point["fine_tuning_audits"].values()
    ):
        raise ValueError("Copied fine-tuning work failed")


def verify_window(window: dict[str, Any], order: list[int], offset: int, policy: str, horizon: int = 36) -> None:
    """Reconstruct masks, arrival identities and two optimizer clocks independently."""
    ledger, losses = window["ledger"], window["losses"]
    if len(ledger) != horizon:
        raise ValueError("Window arrival count differs")
    accepted_steps = [a for a in range(offset + 1, offset + horizon + 1) if policy == "full" or a % 2 == 0]
    updates = 0
    for index, row in enumerate(ledger):
        arrival = offset + index + 1
        accept = arrival in accepted_steps
        updates += int(accept)
        if (
            row["step"] != arrival
            or row["current"] != order[(arrival - 1) * 32 : arrival * 32]
            or row["accepted"] != accept
            or row["branch_updates"] != updates
            or row["optimizer_steps"] != offset // 2 + updates
        ):
            raise ValueError("Causal mask/arrival/update ledger differs")
        if not np.isfinite(row["online_seconds"]) or row["online_seconds"] < 0:
            raise ValueError("Invalid online time")
    if (
        window["updates"] != len(accepted_steps)
        or window["optimizer_images"] != len(accepted_steps) * 32
        or window["arrivals"] != horizon * 32
        or [p["step"] for p in losses] != accepted_steps
        or [p["optimizer_step"] for p in losses] != list(range(offset // 2 + 1, offset // 2 + updates + 1))
    ):
        raise ValueError("Accepted optimizer exposure differs")
    if not np.isfinite([[p["loss"], p["grad_norm"]] for p in losses]).all():
        raise ValueError("Nonfinite optimizer observation")
    close(window["online_seconds"], sum(row["online_seconds"] for row in ledger), "online timing")
    if window["online_seconds"] <= 0:
        raise ValueError("Missing online work")


def verify_final(state: dict[str, Any], offset: int, updates: int) -> None:
    """Require complete finite learning state and cumulative Adam clocks."""
    if (
        state["arrival_step"] != offset + 36
        or state["optimizer_steps"] != offset // 2 + updates
        or not state["model_training"]
        or state["torch_rng"].dtype != torch.uint8
        or state["torch_rng"].ndim != 1
        or not state["model"]
        or not state["optimizer"]["state"]
    ):
        raise ValueError("Final learning state incomplete")
    if any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Nonfinite model state")
    for row in state["optimizer"]["state"].values():
        if int(row["step"]) != offset // 2 + updates or any(
            isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in row.values()
        ):
            raise ValueError("Final Adam clock/values differ")


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - explicit full grid, provenance and prospective inference
    """Audit all 80 paired branches before reporting opportunity and uncertainty."""
    locked = json.loads((out / "input_lock.json").read_text())
    for name, sha in locked["hashes"].items():
        if digest(Path(name)) != sha:
            raise ValueError(f"Changed sealed input: {name}")
    parent = Path(locked["parent"])
    data = json.loads((out / "data_lock.json").read_text())
    seeds = locked["config"]["seeds"]
    draws = locked["config"]["recurrence"]["draw_seeds"]
    preflight = json.loads((out / "preflight.json").read_text())
    if (
        not preflight["passed"]
        or not preflight["restored_continuation_exact"]
        or preflight["continuation_updates"] != 36
    ):
        raise ValueError("Restoration not qualified")
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        query = arrays["query"]
        labels, domains = arrays["labels"][query], arrays["domains"][query]
        if query.tolist() != data["evaluation"]["query_positions"]:
            raise ValueError("Query identity differs")
        for support in data["evaluation"]["supports"].values():
            if set(support) & set(query.tolist()) or np.bincount(arrays["labels"][support]).tolist() != [25, 25, 25]:
                raise ValueError("Support/query information budget differs")
    compatibility = json.loads((out / "compatibility/result.json").read_text())
    first = parent / "recurrence_fixed_half" / f"seed_{seeds[0]}"
    next_state = torch.load(first / "decision_states/arrival_000108.pt", map_location="cpu", weights_only=False)
    continued_state = torch.load(out / "compatibility/final_state.pt", map_location="cpu", weights_only=False)
    if state_hash(continued_state) != state_hash({k: next_state[k] for k in continued_state}):
        raise ValueError("Restored continuation state differs")
    verify_window(compatibility["window"], data["orders"][str(seeds[0])]["recurrence"], 36, "half", 72)
    old_losses = json.loads((first / "losses.json").read_text())["losses"]
    if compatibility["window"]["losses"] != [p for p in old_losses if 36 < p["step"] <= 108]:
        raise ValueError("Continued optimizer observations differ")
    verify_terminal(out / "compatibility", compatibility["terminal"], labels, domains, draws)
    with (
        np.load(first / "predictions_000108.npz", allow_pickle=False) as a,
        np.load(out / "compatibility/predictions_000108.npz", allow_pickle=False) as b,
    ):
        if any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Continued prediction panel differs")
    phases = []
    branch_costs = []
    for offset in OFFSETS:
        rows: dict[str, list[dict[str, Any]]] = {"full": [], "half": []}
        ratios = []
        for seed in seeds:
            parent_file = parent / f"recurrence_fixed_half/seed_{seed}/decision_states/arrival_{offset:06d}.pt"
            parent_state = torch.load(parent_file, map_location="cpu", weights_only=False)
            order = data["orders"][str(seed)]["recurrence"]
            if (
                len(order) != 18432
                or len(set(order)) != 18432
                or parent_state["arrival_order_sha256"] != order_hash(order)
            ):
                raise ValueError("Parent/order identity differs")
            if parent_state["arrival_step"] != offset or parent_state["optimizer_steps"] != offset // 2:
                raise ValueError("Parent clock differs")
            pair = {}
            for policy in ("full", "half"):
                directory = out / f"seed_{seed}/arrival_{offset:06d}" / policy
                result = json.loads((directory / "result.json").read_text())
                if (
                    result["seed"] != seed
                    or result["offset"] != offset
                    or result["policy"] != policy
                    or result["parent_sha256"] != digest(parent_file)
                    or not result["parent_unchanged"]
                    or result["arrival_order_sha256"] != order_hash(order)
                ):
                    raise ValueError("Branch parent binding differs")
                learning_keys = ("model", "optimizer", "torch_rng", "model_training", "arrival_step", "optimizer_steps")
                if result["restored_fingerprint"] != state_hash({k: parent_state[k] for k in learning_keys}):
                    raise ValueError("Restored optimizer/model/RNG fingerprint differs")
                verify_window(result["window"], order, offset, policy)
                final = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
                verify_final(final, offset, result["window"]["updates"])
                if state_hash(final) != result["final_state_fingerprint"] or result["terminal"]["step"] != offset + 36:
                    raise ValueError("Terminal learning state/horizon differs")
                verify_terminal(directory, result["terminal"], labels, domains, draws)
                rows[policy].append(result)
                pair[policy] = result
                branch_costs.append(
                    {
                        "seed": seed,
                        "offset": offset,
                        "policy": policy,
                        "online_seconds": result["window"]["online_seconds"],
                        "restoration_seconds": result["restoration_seconds"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    }
                )
            ratios.append(pair["half"]["window"]["online_seconds"] / pair["full"]["window"]["online_seconds"])
        comparisons = {
            kind: {
                metric: bands(
                    [
                        a["terminal"]["scores"][kind][metric] - b["terminal"]["scores"][kind][metric]
                        for a, b in zip(rows["half"], rows["full"], strict=True)
                    ]
                )
                for metric in ("balanced", "A", "B")
            }
            for kind in KINDS
        }
        primary = comparisons["finetune"]
        mean_ratio = float(np.mean(ratios))
        phases.append(
            {
                "offset": offset,
                "terminal_arrival": offset + 36,
                "half_minus_full": comparisons,
                "mean_half_full_online_ratio": mean_ratio,
                "paired_seed_timing_ratios": ratios,
                "preference": preference(primary, mean_ratio),
                "half_accuracy_advantage": primary["balanced"]["simultaneous95_pp"][0] > 0,
            }
        )
    half_phases = [p["offset"] for p in phases if p["preference"] == "half_efficiency_preferred"]
    full_phases = [p["offset"] for p in phases if p["preference"] == "full_quality_required"]
    decision = (
        "candidate_state_dependent_opportunity"
        if half_phases and full_phases
        else "local_half_rate_candidate"
        if half_phases
        else "no_qualified_local_half_rate_opportunity"
    )
    result = {
        "decision": decision,
        "half_preferred_offsets": half_phases,
        "full_required_offsets": full_phases,
        "phases": phases,
        "branch_costs": branch_costs,
        "work": {
            "branches": 80,
            "collection_updates": 2160,
            "collection_optimizer_images": 69120,
            "collection_arrivals": 92160,
            "continuation_updates": 36,
            "discarded_full_updates": 4,
            "copied_finetune_updates": 24600,
        },
        "scope": "Half-rate parent histories, five dependent trajectories, reused data; no policy fitting.",
        "primary_interval_rule": "24-contrast Bonferroni Student-t simultaneous bands, df4",
    }
    atomic_json(out / "report.json", result)
    lines = [
        "# Paired update-rate responses",
        "",
        f"Decision: **{decision}**.",
        "",
        "Half-rate efficiency preference requires all adjusted quality lower bounds >-1pp and >=25% online savings.",
        "Eight prespecified phases across five seeds; conditional half-rate histories and reused data.",
        "",
        "| Parent arrival | Half minus full fine-tuning pp | Nominal95% | Adjusted95% | Online saving | Preference |",
        "|---|---|---|---|---|---|",
    ]
    for phase in phases:
        value = phase["half_minus_full"]["finetune"]["balanced"]
        lo, hi = value["ci95_pp"]
        alo, ahi = value["simultaneous95_pp"]
        lines.append(
            f"| {phase['offset']} | {value['mean_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"[{alo:+.3f}, {ahi:+.3f}] | {100 * (1 - phase['mean_half_full_online_ratio']):.1f}% | "
            f"{phase['preference']} |"
        )
    lines.extend(
        [
            "",
            f"Half efficiency-preferred offsets: {half_phases}.",
            f"Full quality-required offsets: {full_phases}.",
            "",
            "Domain guardrails, all readers, individual paired seeds and costs are retained in report.json.",
            "State-specific query-selected actions are diagnostic envelopes, not deployable policies.",
            "New/full-rate histories and prospective repeated rollouts are required before controller claims.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = Figure(figsize=(10, 5), constrained_layout=True)
    ax = fig.subplots()
    values = [p["half_minus_full"]["finetune"]["balanced"] for p in phases]
    means = np.array([v["mean_pp"] for v in values])
    error = np.array(
        [
            [v["mean_pp"] - v["simultaneous95_pp"][0] for v in values],
            [v["simultaneous95_pp"][1] - v["mean_pp"] for v in values],
        ]
    )
    ax.errorbar(list(OFFSETS), means, yerr=error, fmt="o", capsize=4)
    ax.axhline(-1, color="red", linestyle="--", label="quality tolerance -1pp")
    ax.axhline(0, color="black", linestyle=":")
    ax.set(
        xlabel="Parent arrival step",
        ylabel="Half minus full fine-tuning accuracy (pp)",
        title="Fixed36-arrival responses; adjusted24-contrast bands",
    )
    ax.legend()
    fig.savefig(out / "responses.png", dpi=150)
    atomic_json(
        out / "independent_verification.json",
        {
            "passed": True,
            "branches_checked": 80,
            "parent_states_checked": 40,
            "arrival_decisions_reconstructed": 2880,
            "restored_continuation_exact": True,
            "input_hashes_unchanged": True,
            "report_sha256": digest(out / "report.json"),
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
        (root / "docs/experiments" / folder / "slow-edge-paired-update-response.md").write_text(
            (out / "report.md").read_text() + f"\nArtifacts: `{out.relative_to(root).as_posix()}`.\n", encoding="utf-8"
        )
    return result


def main() -> None:
    """Audit one completed sealed collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(report(parser.parse_args().run.resolve()), indent=2))


if __name__ == "__main__":
    main()
