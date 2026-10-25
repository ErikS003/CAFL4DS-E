"""Audit temporal parent reproduction and paired actions without fitting a controller."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from scripts.decode_bdd_temporal import sha256
from scripts.report_paired_update_response import bands, preference, state_hash, verify_terminal
from scripts.run_temporal_action_response import HISTORIES, HORIZON, OFFSETS, SEEDS


def verify_rows(rows: list[dict[str, Any]], offset: int, action: str, horizon: int) -> int:
    """Reconstruct uniform/full positions directly, without the runtime selector."""
    if len(rows) != horizon or action not in {"full", "half"}:
        raise ValueError("invalid branch window count/action")
    positions = list(range(64)) if action == "full" else list(range(1, 64, 2))
    updates = 2 if action == "full" else 1
    for number, row in enumerate(rows):
        start = (offset + number) * 64
        if (row["start_frame"], row["end_frame_exclusive"]) != (start, start + 64):
            raise ValueError("arrival clock mismatch")
        if row["selected_positions"] != positions or row["selected_global_indices"] != [start + p for p in positions]:
            raise ValueError("selected source positions mismatch")
        if row["optimizer_updates"] != updates or row["accepted_frames"] != len(positions):
            raise ValueError("action work mismatch")
        if len(row["loss_gradients"]) != updates or any(
            pair is None or len(pair) != 2 or not all(math.isfinite(v) for v in pair) for pair in row["loss_gradients"]
        ):
            raise ValueError("invalid loss/gradient observations")
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in ("online_seconds", "selection_seconds", "ssl_seconds")):
            raise ValueError("invalid timing")
        if row["selection_seconds"] + row["ssl_seconds"] > row["online_seconds"] + 1e-6:
            raise ValueError("timing components exceed total")
    return updates * horizon


def verify_state(state: dict[str, Any], arrivals: int, updates: int) -> None:
    """Require actual Adam counters and finite tensors to match recorded clocks."""
    if state["arrivals"] != arrivals or state["optimizer_updates"] != updates:
        raise ValueError("learning clocks differ")
    if any(not torch.isfinite(v).all() for v in state["model"].values()):
        raise ValueError("nonfinite model")
    if not state["optimizer"]["state"] or any(
        int(v["step"]) != updates or any(not torch.isfinite(tensor).all() for tensor in v.values())
        for v in state["optimizer"]["state"].values()
    ):
        raise ValueError("Adam moments/counters differ")


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - independent state and work guards
    """Validate all parents and branches before deriving prespecified preferences."""
    inputs = json.loads((out / "input_lock.json").read_text())
    parent_lock = json.loads((out / "parent_lock.json").read_text())
    for name, expected in {**inputs["hashes"], **parent_lock["hashes"]}.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"locked evidence changed: {name}")
    if (inputs["seeds"], inputs["histories"], inputs["offsets"], inputs["horizon"]) != (
        list(SEEDS),
        list(HISTORIES),
        list(OFFSETS),
        HORIZON,
    ):
        raise ValueError("frozen design differs")
    status = json.loads((out / "status.json").read_text())
    if status["stage"] != "collection_complete" or status["completed_branches"] != 80:
        raise ValueError("collection incomplete")
    prior = Path(inputs["prior"])
    data = json.loads((out / "data_lock.json").read_text())
    prior_data = json.loads((prior / "data_lock.json").read_text())
    if data != prior_data:
        raise ValueError("data allocation changed")
    with np.load(prior / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        query = arrays["query"]
        labels, domains = arrays["labels"][query], arrays["domains"][query]
        if query.tolist() != data["evaluation"]["query_positions"] or len(query) != 600:
            raise ValueError("query allocation differs")
        for support in data["evaluation"]["supports"].values():
            if set(support) & set(query.tolist()) or np.bincount(arrays["labels"][support]).tolist() != [25, 25, 25]:
                raise ValueError("support budget differs")
    draws = list(map(int, data["evaluation"]["supports"]))
    parent_updates = 0
    for history in HISTORIES:
        for seed in SEEDS:
            directory = out / "parents" / history / f"seed_{seed}"
            reproduction = json.loads((directory / "reproduction.json").read_text())
            updates = verify_rows(reproduction["windows"], 0, "full" if history == "full" else "half", 96)
            old_rows = json.loads((prior / history / f"seed_{seed}" / "windows.json").read_text())["windows"]
            if (
                not reproduction["passed"]
                or updates != reproduction["optimizer_updates"]
                or any(
                    a["loss_gradients"] != b["loss_gradients"]
                    for a, b in zip(reproduction["windows"], old_rows, strict=True)
                )
            ):
                raise ValueError("parent exact-loss reproduction differs")
            final = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
            expected = torch.load(
                prior / history / f"seed_{seed}" / "final_state.pt", map_location="cpu", weights_only=False
            )
            verify_state(final, 6144, updates)
            if state_hash(final) != state_hash(expected):
                raise ValueError("parent final state does not reproduce")
            parent_updates += updates
    if (
        parent_updates != 1440
        or parent_lock["reconstruction_updates"] != parent_updates
        or len(parent_lock["hashes"]) != 41
    ):
        raise ValueError("parent collection budget differs")
    preflight = json.loads((out / "preflight.json").read_text())
    if not preflight["passed"] or not preflight["restored_continuation_exact"]:
        raise ValueError("restoration preflight failed")
    continued = torch.load(out / "compatibility_final_state.pt", map_location="cpu", weights_only=False)
    expected_continued = torch.load(
        out / "parents/full/seed_5003/window_016.pt", map_location="cpu", weights_only=False
    )
    if state_hash(continued) != state_hash(expected_continued):
        raise ValueError("complete restored continuation differs")
    reproduction = json.loads((out / "parents/full/seed_5003/reproduction.json").read_text())
    if [r["loss_gradients"] for r in preflight["continuation"]] != [
        r["loss_gradients"] for r in reproduction["windows"][8:16]
    ]:
        raise ValueError("restored continuation losses differ")
    if (
        verify_rows(preflight["continuation"], 8, "full", 8) != 16
        or verify_rows(preflight["half_smoke"], 8, "half", 1) != 1
    ):
        raise ValueError("preflight budget differs")
    frozen = json.loads((out / "frozen/result.json").read_text())
    if frozen != json.loads((prior / "frozen/result.json").read_text()):
        raise ValueError("frozen reproduction differs")
    verify_terminal(out / "frozen", frozen, labels, domains, draws)
    phases: list[dict[str, Any]] = []
    branches = 0
    branch_updates = 0
    for history in HISTORIES:
        for offset in OFFSETS:
            results: dict[str, list[dict[str, Any]]] = {"full": [], "half": []}
            ratios = []
            for seed in SEEDS:
                parent_file = out / "parents" / history / f"seed_{seed}" / f"window_{offset:03d}.pt"
                parent = torch.load(parent_file, map_location="cpu", weights_only=False)
                nparent = offset * (2 if history == "full" else 1)
                verify_state(parent, offset * 64, nparent)
                pair = {}
                for action in ("full", "half"):
                    directory = out / "branches" / history / f"seed_{seed}" / f"window_{offset:03d}" / action
                    result = json.loads((directory / "result.json").read_text())
                    if (result["history"], result["seed"], result["offset"], result["action"]) != (
                        history,
                        seed,
                        offset,
                        action,
                    ):
                        raise ValueError("branch identity differs")
                    initial = torch.load(directory / "initial_state.pt", map_location="cpu", weights_only=False)
                    if state_hash(initial) != state_hash(parent) or result["restored_fingerprint"] != state_hash(
                        parent
                    ):
                        raise ValueError("complete initial model/Adam/RNG restoration differs")
                    if result["parent_sha256"] != sha256(parent_file) or not result["parent_and_readout_unchanged"]:
                        raise ValueError("parent binding or readout preservation differs")
                    updates = verify_rows(result["windows"], offset, action, HORIZON)
                    if updates != result["optimizer_updates"] or not math.isclose(
                        sum(r["online_seconds"] for r in result["windows"]), result["online_seconds"], abs_tol=1e-8
                    ):
                        raise ValueError("branch totals differ")
                    final = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
                    verify_state(final, (offset + HORIZON) * 64, nparent + updates)
                    if (
                        state_hash(final) != result["final_fingerprint"]
                        or result["outcome"]["step"] != (offset + HORIZON) * 2
                    ):
                        raise ValueError("terminal state or horizon differs")
                    with np.load(
                        directory / f"predictions_{(offset + HORIZON) * 2:06d}.npz", allow_pickle=False
                    ) as predictions:
                        if any(v.shape != (600,) or not np.isin(v, [0, 1, 2]).all() for v in predictions.values()):
                            raise ValueError("invalid terminal predictions")
                    verify_terminal(directory, result["outcome"], labels, domains, draws)
                    results[action].append(result)
                    pair[action] = result
                    branches += 1
                    branch_updates += updates
                ratios.append(pair["half"]["online_seconds"] / pair["full"]["online_seconds"])
            comparisons = {
                kind: {
                    metric: bands(
                        [
                            a["outcome"]["scores"][kind][metric] - b["outcome"]["scores"][kind][metric]
                            for a, b in zip(results["half"], results["full"], strict=True)
                        ]
                    )
                    for metric in ("balanced", "A", "B")
                }
                for kind in ("finetune", "linear", "knn")
            }
            ratio = statistics.fmean(ratios)
            phases.append(
                {
                    "history": history,
                    "offset": offset,
                    "terminal_window": offset + HORIZON,
                    "half_minus_full": comparisons,
                    "paired_timing_ratios": ratios,
                    "mean_online_savings": 1 - ratio,
                    "preference": preference(comparisons["finetune"], ratio),
                }
            )
    if branches != 80 or branch_updates != 960:
        raise ValueError("branch work differs")
    half = [(p["history"], p["offset"]) for p in phases if p["preference"] == "half_efficiency_preferred"]
    full = [(p["history"], p["offset"]) for p in phases if p["preference"] == "full_quality_required"]
    verdict = (
        "candidate_state_dependent_opportunity"
        if half and full
        else ("local_half_rate_candidate" if half else "no_qualified_local_half_rate_opportunity")
    )
    result = {
        "decision": verdict,
        "half_preferred_strata": half,
        "full_required_strata": full,
        "phases": phases,
        "work": {
            "branches": branches,
            "parent_states": 40,
            "parent_updates": parent_updates,
            "branch_updates": branch_updates,
            "compatibility_updates": 16,
            "discarded_half_updates": 1,
            "copied_finetune_updates": 24300,
        },
        "primary_intervals": "24-contrast Bonferroni Student-t bands, df4; eight strata, five paired seeds",
        "scope": "reused temporal BDD development, dependent states, two reconstructed histories",
        "controller_qualified": False,
        "independent_confirmation": False,
    }
    atomic_json(out / "report.json", result)
    lines = [
        "# Paired temporal full/half action responses",
        "",
        f"Decision: **{verdict}**.",
        "",
        "80 audited branches; 40 parents; 960 branch and 1,440 parent-reconstruction SSL updates.",
        "",
        "| History | Parent window | Half minus full balanced pp | Adjusted 95% | Savings | Preference |",
        "|---|---:|---:|---|---:|---|",
    ]
    for phase in phases:
        value = phase["half_minus_full"]["finetune"]["balanced"]
        lines.append(
            f"| {phase['history']} | {phase['offset']} | {value['mean_pp']:+.3f} | "
            f"{value['simultaneous95_pp']} | {100 * phase['mean_online_savings']:.1f}% | {phase['preference']} |"
        )
    lines.extend(
        [
            "",
            "Domain guardrails, nominal intervals, paired seeds and secondary readers are in report.json.",
            "These are fixed-position diagnostic preferences, not an evaluated deployable policy.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(
        out / "independent_verification.json",
        {
            "passed": True,
            "branches_checked": branches,
            "parent_states_checked": 40,
            "parent_reproductions_checked": 10,
            "branch_windows_reconstructed": 640,
            "parent_windows_reconstructed": 960,
            "raw_predictions_rescored": True,
            "audit_source_sha256": sha256(Path(__file__)),
            "artifact_hashes": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and "source_snapshot" not in p.parts
                and p.suffix in {".json", ".pt", ".npz", ".md"}
                and p.name != "independent_verification.json"
            },
        },
    )
    return result


def main() -> None:
    """Audit a completed collection and print its complete scientific report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(report(parser.parse_args().output.resolve()), indent=2))


if __name__ == "__main__":
    main()
