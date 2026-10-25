"""Independently rescore terminal predictions and audit the temporal feasibility budget."""

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
from scripts.confirm_replay_advantage import paired_interval
from scripts.decode_bdd_temporal import sha256
from scripts.run_paired_update_response import fingerprint


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - independent scoring and budget checks
    """Verify all arms before applying the frozen primary decision."""
    inputs = json.loads((out / "input_lock.json").read_text())
    for name, expected in inputs["hashes"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"locked input changed: {name}")
    seeds = inputs["seeds"]
    data = json.loads((out / "data_lock.json").read_text())
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        query = arrays["query"]
        labels = torch.from_numpy(arrays["labels"][query])
        domains = torch.from_numpy(arrays["domains"][query])
    if query.tolist() != data["evaluation"]["query_positions"] or data["arrivals"] != 6144:
        raise ValueError("evaluation or arrival identity mismatch")
    orders = json.loads((out / "orders.json").read_text())
    if orders["chronological"] != list(range(6144)) or sorted(orders["shuffled"]) != list(range(6144)):
        raise ValueError("order is not a permutation of the same arrivals")
    frozen = json.loads((out / "frozen/result.json").read_text())
    contrasts = {}
    starts: dict[int, str] = {}
    checked = 0
    total_updates = 0
    for order_name in ("chronological", "shuffled"):
        scores: dict[str, list[dict[str, float]]] = {"full": [], "half": []}
        times: dict[str, list[float]] = {"full": [], "half": []}
        for seed in seeds:
            for policy in ("full", "half"):
                directory = out / order_name / policy / f"seed_{seed}"
                result = json.loads((directory / "result.json").read_text())
                records = json.loads((directory / "arrivals.json").read_text())["records"]
                if len(records) != 192 or [r["arrival_batch"] for r in records] != list(range(1, 193)):
                    raise ValueError("arrival clock mismatch")
                expected = [policy == "full" or step % 2 == 0 for step in range(1, 193)]
                if [r["accepted"] for r in records] != expected:
                    raise ValueError("acceptance mask mismatch")
                count = 192 if policy == "full" else 96
                if sum(expected) != result["optimizer_updates"]:
                    raise ValueError("optimizer budget mismatch")
                for row in records:
                    values = row["loss_gradient"]
                    if row["accepted"]:
                        if len(values) != 2 or not all(math.isfinite(v) for v in values):
                            raise ValueError("nonfinite accepted update")
                    elif values is not None:
                        raise ValueError("rejected batch has a training result")
                if not math.isclose(sum(r["online_seconds"] for r in records), result["online_seconds"], abs_tol=1e-8):
                    raise ValueError("online timing total mismatch")
                initial = torch.load(directory / "initial_state.pt", map_location="cpu", weights_only=False)
                final = torch.load(directory / "final_state.pt", map_location="cpu", weights_only=False)
                if fingerprint(initial) != result["initial_fingerprint"]:
                    raise ValueError("immutable initial snapshot fingerprint mismatch")
                starts.setdefault(seed, result["initial_fingerprint"])
                if starts[seed] != result["initial_fingerprint"]:
                    raise ValueError("paired policies or orders did not start from identical learning state")
                if sha256(directory / "initial_state.pt") != result["initial_file_sha256"]:
                    raise ValueError("initial snapshot file changed")
                if final["arrivals"] != 6144 or final["optimizer_updates"] != count:
                    raise ValueError("final checkpoint clock mismatch")
                if any(int(value["step"]) != count for value in final["optimizer"]["state"].values()):
                    raise ValueError("optimizer moment step counters differ from declared work")
                if not result["readout_preserved_learning_state"]:
                    raise ValueError("readout state preservation failed")
                point = result["outcome"]
                panels: dict[str, list[dict[str, float]]] = {kind: [] for kind in ("finetune", "linear", "knn")}
                with np.load(directory / "predictions_000192.npz", allow_pickle=False) as predictions:
                    for draw, _support in data["evaluation"]["supports"].items():
                        for kind, collected in panels.items():
                            predicted = predictions[f"{draw}_{kind}"]
                            if predicted.shape != (len(query),) or not np.isin(predicted, [0, 1, 2]).all():
                                raise ValueError("prediction shape or class mismatch")
                            computed = prediction_scores(torch.from_numpy(predicted), labels, domains)
                            for metric, value in computed.items():
                                if not math.isclose(value, point["panels"][draw][kind][metric], abs_tol=1e-12):
                                    raise ValueError("per-panel score differs from raw predictions")
                            collected.append(computed)
                for kind, panel in panels.items():
                    for metric in ("A", "B", "balanced", "micro"):
                        value = statistics.fmean(p[metric] for p in panel)
                        if not math.isclose(value, point["scores"][kind][metric], abs_tol=1e-12):
                            raise ValueError("pooled terminal score mismatch")
                scores[policy].append(point["scores"]["finetune"])
                times[policy].append(result["online_seconds"])
                checked += 1
                total_updates += count
        floor = paired_interval([
            p["balanced"] - frozen["scores"]["finetune"]["balanced"] for p in scores["full"]
        ])
        half_deltas = {
            metric: paired_interval([a[metric] - b[metric] for a, b in zip(
                scores["half"], scores["full"], strict=True)]) for metric in ("balanced", "A", "B")
        }
        savings = [1 - h / f for h, f in zip(times["half"], times["full"], strict=True)]
        useful = floor["ci95_pp"][0] > 0
        retained = all(c["ci95_pp"][0] > -1 for c in half_deltas.values())
        cost_pass = statistics.fmean(savings) >= .25
        contrasts[order_name] = {"full_minus_frozen": floor, "half_minus_full": half_deltas,
                                "online_savings_per_seed": savings, "mean_online_savings": statistics.fmean(savings),
                                "full_establishes_ssl": useful, "half_noninferior_all_guardrails": retained,
                                "cost_pass": cost_pass,
                                "qualifies_static_half_point": useful and retained and cost_pass}
    if checked != 20 or total_updates != 2880:
        raise ValueError("incomplete collection")
    primary = contrasts["chronological"]
    decision = ("useful_static_half_rate_point" if primary["qualifies_static_half_point"] else
                "no_qualified_efficiency_point" if primary["full_establishes_ssl"] else
                "temporal_ssl_floor_not_established")
    result = {"decision": decision, "primary": "chronological", "contrasts": contrasts,
              "scope": "reused BDD development; conditional on fixed clips, warm model and support/query panels",
              "controller_qualified": False, "source_independence_established": False}
    atomic_json(out / "report.json", result)
    text = ["# Temporal BDD SSL feasibility", "", f"Decision: **{decision}**.", "",
            "Audited 20 learning arms, 2,880 updates and 6,144 arrivals per arm.", "",
            "| Order | Full minus frozen, pp (95% interval) | Half minus full, pp (95% interval) | Online savings |",
            "|---|---:|---:|---:|"]
    for order, row in contrasts.items():
        floor, half = row["full_minus_frozen"], row["half_minus_full"]["balanced"]
        text.append(f"| {order} | {floor['mean_pp']:.3f} {floor['ci95_pp']} | "
                    f"{half['mean_pp']:.3f} {half['ci95_pp']} | {100 * row['mean_online_savings']:.1f}% |")
    text.extend(["", "Chronological results are primary; global shuffle is an offline diagnostic.",
                 "Intervals describe training-seed uncertainty conditional on reused data. They do not",
                 "measure independence over driving sessions. No content filter or controller is qualified."])
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    artifacts = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
                 if p.is_file() and p.suffix in {".json", ".npz", ".pt"} and p.name != "independent_verification.json"}
    atomic_json(out / "independent_verification.json", {"passed": True, "arms_checked": checked,
                "optimizer_updates": total_updates, "raw_predictions_rescored": True, "artifact_hashes": artifacts,
                "audit_source_sha256": sha256(Path(__file__))})
    return result


def main() -> None:
    """Audit one completed collection without overwriting its scientific evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
