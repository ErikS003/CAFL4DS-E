"""Independently reconstruct content selections, raw readouts and the frozen joint gate."""

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
from cafl4ds.recurrence import prediction_scores
from cafl4ds.temporal_anchored_filter import RANDOM_SEED_OFFSET, select_window
from scripts.confirm_replay_advantage import paired_interval
from scripts.decode_bdd_temporal import sha256
from scripts.run_bdd_anchored_filter import POLICIES, verify_control
from scripts.run_paired_update_response import fingerprint


def decision(contrasts: dict[str, Any], savings: dict[str, float]) -> dict[str, Any]:
    """Separate static efficiency from added content value; never qualify control."""
    checks = {}
    for candidate in ("uniform", "anchored"):
        checks[candidate] = {
            "noninferior_to_full_all_domains": all(
                contrasts[f"{candidate}_minus_full"][metric]["ci95_pp"][0] > -1
                for metric in ("balanced", "A", "B")),
            "beats_frozen": contrasts[f"{candidate}_minus_frozen"]["balanced"]["ci95_pp"][0] > 0,
            "cost_savings_at_least_25_percent": savings[candidate] >= .25,
        }
    efficiency = {name: all(fields.values()) for name, fields in checks.items()}
    content = {
        "beats_uniform": contrasts["anchored_minus_uniform"]["balanced"]["ci95_pp"][0] > 0,
        "beats_random": contrasts["anchored_minus_random"]["balanced"]["ci95_pp"][0] > 0,
    }
    return {"decision": "static_efficiency_candidate" if any(efficiency.values()) else
            "no_qualified_static_efficiency", "efficiency_checks": checks,
            "efficiency_qualified": efficiency, "content_checks": content,
            "added_content_value_qualified": all(content.values()),
            "controller_qualified": False, "independent_confirmation": False}


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - separate audit guards
    """Audit all twenty-five arms and then compute primary and secondary comparisons."""
    inputs = json.loads((out / "input_lock.json").read_text())
    for name, expected in inputs["hashes"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"locked source/input changed: {name}")
    data = json.loads((out / "data_lock.json").read_text())
    intake = json.loads(Path(data["intake_path"]).read_text())
    if sha256(Path(data["intake_path"])) != data["intake_sha256"]:
        raise ValueError("intake changed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    arrays = [np.load(clip["frames_path"], allow_pickle=False) for clip in intake["clips"]]
    images = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
    del arrays
    if images.shape != (6144, 3, 64, 64) or data["clip_ids"] != [c["clip_id"] for c in intake["clips"]]:
        raise ValueError("source clip/arrival identity mismatch")
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as arrays_np:
        query = arrays_np["query"]
        labels = torch.from_numpy(arrays_np["labels"][query])
        domains = torch.from_numpy(arrays_np["domains"][query])
        if query.tolist() != data["evaluation"]["query_positions"] or len(query) != 600:
            raise ValueError("query identity mismatch")
        for support in data["evaluation"]["supports"].values():
            if set(support) & set(query.tolist()) or np.bincount(arrays_np["labels"][support]).tolist() != [25, 25, 25]:
                raise ValueError("support/query information budget mismatch")
    frozen = json.loads((out / "frozen/result.json").read_text())
    scores: dict[str, list[dict[str, float]]] = {p: [] for p in POLICIES}
    readers: dict[str, dict[str, list[dict[str, float]]]] = {
        kind: {p: [] for p in POLICIES} for kind in ("linear", "knn")}
    costs: dict[str, list[float]] = {p: [] for p in POLICIES}
    selection_costs: dict[str, list[float]] = {p: [] for p in POLICIES}
    starts: dict[int, str] = {}
    total_updates = 0
    arms = 0
    for seed in inputs["seeds"]:
        for policy in POLICIES:
            directory = out / policy / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text())
            windows = json.loads((directory / "windows.json").read_text())["windows"]
            generator = torch.Generator().manual_seed(seed + RANDOM_SEED_OFFSET)
            if len(windows) != 96:
                raise ValueError("incomplete window trace")
            for number, row in enumerate(windows):
                start = number * 64
                wanted = select_window(policy, images[start:start + 64], generator)
                if (row["window"], row["start_frame"], row["end_frame_exclusive"]) != (number + 1, start, start + 64):
                    raise ValueError("window arrival clock mismatch")
                matches = row["selected_positions"] == wanted.tolist()
                if not matches or row["selected_global_indices"] != (wanted + start).tolist():
                    raise ValueError("content/random/fixed decision differs from independent reconstruction")
                if row["clip_id"] != intake["clips"][number // 2]["clip_id"]:
                    raise ValueError("window crosses or mislabels a source clip")
                nupdates = 2 if policy == "full" else 1
                if row["accepted_frames"] != len(wanted) or row["optimizer_updates"] != nupdates:
                    raise ValueError("accepted-frame/optimizer budget mismatch")
                if len(row["loss_gradients"]) != nupdates or not all(
                    value is not None and len(value) == 2 and all(math.isfinite(v) for v in value)
                    for value in row["loss_gradients"]
                ):
                    raise ValueError("invalid SSL updates")
                if not all(math.isfinite(row[k]) and row[k] >= 0 for k in
                           ("online_seconds", "selection_seconds", "ssl_seconds")):
                    raise ValueError("invalid online timings")
                if row["selection_seconds"] + row["ssl_seconds"] > row["online_seconds"] + 1e-6:
                    raise ValueError("monitoring/training time exceeds measured online total")
            expected_updates = 192 if policy == "full" else 96
            for field in ("accepted_frames", "optimizer_updates", "online_seconds", "selection_seconds", "ssl_seconds"):
                if not math.isclose(sum(r[field] for r in windows), result[field], abs_tol=1e-8):
                    raise ValueError(f"trace total differs: {field}")
            initial = torch.load(directory / "initial_state.pt", map_location="cpu", weights_only=False)
            final = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
            marker = fingerprint(initial)
            snapshot_hash = sha256(directory / "initial_state.pt")
            if marker != result["initial_fingerprint"] or snapshot_hash != result["initial_file_sha256"]:
                raise ValueError("initial state snapshot mismatch")
            starts.setdefault(seed, marker)
            if starts[seed] != marker or initial["arrivals"] != 0 or initial["optimizer_updates"] != 0:
                raise ValueError("paired initial states differ")
            if final["arrivals"] != 6144 or final["optimizer_updates"] != expected_updates:
                raise ValueError("final learning clocks differ")
            if not final["optimizer"]["state"] or any(
                int(v["step"]) != expected_updates for v in final["optimizer"]["state"].values()
            ):
                raise ValueError("optimizer moment counters differ from actual work")
            if not result["readout_preserved_learning_state"]:
                raise ValueError("readout changed learning state")
            if seed == 5003 and policy in {"full", "fixed_half"}:
                verify_control(directory, policy)
            point = result["outcome"]
            with np.load(directory / "predictions_000192.npz", allow_pickle=False) as predictions:
                for kind in ("finetune", "linear", "knn"):
                    panels = []
                    for draw, _support in data["evaluation"]["supports"].items():
                        predicted = predictions[f"{draw}_{kind}"]
                        if predicted.shape != (600,) or not np.isin(predicted, [0, 1, 2]).all():
                            raise ValueError("invalid terminal predictions")
                        computed = prediction_scores(torch.from_numpy(predicted), labels, domains)
                        if any(not math.isclose(value, point["panels"][draw][kind][metric], abs_tol=1e-12)
                               for metric, value in computed.items()):
                            raise ValueError("per-panel score differs from raw predictions")
                        panels.append(computed)
                    reconstructed = {metric: statistics.fmean(p[metric] for p in panels)
                                     for metric in ("balanced", "A", "B", "micro")}
                    if any(not math.isclose(value, point["scores"][kind][metric], abs_tol=1e-12)
                           for metric, value in reconstructed.items()):
                        raise ValueError("averaged terminal scores differ")
                    if kind == "finetune":
                        scores[policy].append(reconstructed)
                    else:
                        readers[kind][policy].append(reconstructed)
            costs[policy].append(result["online_seconds"])
            selection_costs[policy].append(result["selection_seconds"])
            total_updates += expected_updates
            arms += 1
    if arms != 25 or total_updates != 2880:
        raise ValueError("incomplete collection budget")
    comparisons = {
        "anchored_minus_random": ("anchored_half", "random_half"),
        "anchored_minus_fixed": ("anchored_half", "fixed_half"),
        "anchored_minus_uniform": ("anchored_half", "uniform_half"),
        "anchored_minus_full": ("anchored_half", "full"),
        "uniform_minus_full": ("uniform_half", "full"),
        "uniform_minus_random": ("uniform_half", "random_half"),
        "uniform_minus_fixed": ("uniform_half", "fixed_half"),
        "random_minus_fixed": ("random_half", "fixed_half"),
        "random_minus_full": ("random_half", "full"),
        "fixed_minus_full": ("fixed_half", "full"),
    }
    contrasts = {name: {metric: paired_interval([
        a[metric] - b[metric] for a, b in zip(scores[left], scores[right], strict=True)])
                       for metric in ("balanced", "A", "B")} for name, (left, right) in comparisons.items()}
    for candidate in ("uniform", "anchored"):
        contrasts[f"{candidate}_minus_frozen"] = {
            metric: paired_interval([p[metric] - frozen["scores"]["finetune"][metric]
                                     for p in scores[f"{candidate}_half"]])
            for metric in ("balanced", "A", "B")}
    savings = {candidate: [1 - c / f for c, f in
               zip(costs[f"{candidate}_half"], costs["full"], strict=True)]
               for candidate in ("uniform", "anchored")}
    means = {candidate: statistics.fmean(values) for candidate, values in savings.items()}
    result = {**decision(contrasts, means), "contrasts": contrasts,
              "online_savings_per_seed": savings, "mean_online_savings": means,
              "online_seconds": costs, "selection_seconds": selection_costs,
              "primary_reader_scores": scores, "secondary_reader_scores": readers,
              "scope": "conditional reused BDD development; two prespecified candidates, no family confirmation"}
    atomic_json(out / "report.json", result)
    lines = ["# Temporal BDD anchored-change filtering", "", f"Decision: **{result['decision']}**.", "",
             "Audited 25 arms, 2,880 updates, 6,144 arrivals per arm; selection cost is included.", "",
             "| Fine-tuning contrast | Balanced pp (95% interval) | Daytime pp | Nighttime pp |",
             "|---|---:|---:|---:|"]
    for name, row in contrasts.items():
        c = row["balanced"]
        lines.append(f"| {name} | {c['mean_pp']:.3f} {c['ci95_pp']} | "
                     f"{row['A']['mean_pp']:.3f} | {row['B']['mean_pp']:.3f} |")
    lines.extend(["", f"Mean paired processing savings: uniform {100 * means['uniform']:.1f}%, "
                  f"anchored {100 * means['anchored']:.1f}%.",
                  "", "This static-window filter cannot qualify a health controller or independent source transfer."])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
                 if p.is_file() and p.suffix in {".json", ".npz", ".pt"} and p.name != "independent_verification.json"}
    atomic_json(out / "independent_verification.json", {"passed": True, "arms_checked": arms,
                "optimizer_updates": total_updates, "windows_reconstructed": 2400,
                "raw_predictions_rescored": True, "artifact_hashes": artifacts,
                "audit_source_sha256": sha256(Path(__file__))})
    return result


def main() -> None:
    """Audit a complete fixed collection and write its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
