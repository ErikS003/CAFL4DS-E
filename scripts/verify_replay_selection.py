"""Independently recompute saved probe readouts and audit fixed replay exposure."""

# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cafl4ds.artifacts import atomic_json
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_image_utility_experiment import identical
from scripts.run_replay_task_alignment import ROTATIONS


def main() -> None:  # noqa: C901
    """Verify all scores through direct numpy probabilities and six balanced cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    verification = json.loads((run / "verification.json").read_text())
    for name, expected in verification["artifact_hashes"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Artifact changed: {name}")
    for path, expected in verification["input_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    root = Path(__file__).resolve().parents[1]
    for path, expected in verification["source_hashes"].items():
        snapshot = run / "source_snapshots" / Path(path).relative_to(root)
        if digest(snapshot) != expected:
            raise ValueError(f"Frozen scientific source changed: {snapshot}")
    for name, expected in verification["frozen_hashes"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Selection changed: {name}")
    metadata = torch.load(run / "readout_metadata.pt", weights_only=False)
    labels, domains = metadata["labels"].numpy(), metadata["domains"].numpy()
    scores = pd.read_csv(run / "scores.csv")
    maximum_error, count = 0.0, 0
    for case, frame in scores.groupby("case"):
        z = torch.load(run / f"features_{case}.pt", weights_only=True).numpy().astype(np.float64)
        for panel, subframe in frame.groupby("panel"):
            if str(panel).startswith("task"):
                rotation = int(str(panel)[4:])
                state = str(frame.iloc[0].state)
                head = torch.load(run / f"task_{state}.pt", weights_only=False)["heads"][rotation]
                heads = {str(1e-10): head}
                roles = {s: metadata["supports"][s] for s in ROTATIONS[rotation][1:]}
            else:
                heads = torch.load(run / f"heads_{case}_{panel}.pt", weights_only=False)
                roles = metadata["roles"]
            for row in subframe.to_dict("records"):
                head = heads[str(row["tolerance"])]
                ix = roles[row["role"]].numpy()
                x = (z[ix] - head.mean) / head.scale
                probability = head.classifier.predict_proba(x)
                classes = head.classifier.classes_
                targets = np.searchsorted(classes, labels[ix])
                accuracy = (classes[probability.argmax(1)] == labels[ix]).astype(np.float64)
                ce = -np.log(np.clip(probability[np.arange(len(ix)), targets], 1e-12, 1))
                for metric, values in (("accuracy", accuracy), ("ce", ce)):
                    zone_values = []
                    for domain, zone in ((0, "A"), (1, "B")):
                        mean = float(
                            np.mean([values[(domains[ix] == domain) & (labels[ix] == c)].mean() for c in classes])
                        )
                        maximum_error = max(maximum_error, abs(mean - row[f"{metric}_{zone}"]))
                        zone_values.append(mean)
                    maximum_error = max(maximum_error, abs(float(np.mean(zone_values)) - row[f"{metric}_balanced"]))
                count += 1
    if maximum_error > 1e-12:
        raise ValueError(f"Readout disagreement: {maximum_error}")
    plans = {p["state"]: p for p in json.loads((run / "frozen_selections.json").read_text())["plans"]}
    branches = json.loads((run / "branches.json").read_text())["branches"]
    for branch in branches:
        plan = plans[branch["state"]]
        rows = branch["decisions"]
        if (
            rows[0]["current_indices"] != plan["current_indices"]
            or rows[0]["replay_indices"] != plan["selections"][branch["rule"]]
            or len(rows) != 4
        ):
            raise ValueError("First selection or update count differs from protocol")
        for row in rows:
            if row["optimizer_images"] != 32 or row["current_count"] != 16 or row["replay_count"] != 16:
                raise ValueError("Unequal training exposure")
            if len(set(row["replay_indices"])) != 16:
                raise ValueError("Duplicate replay exposure")
        baseline = next(
            b
            for b in branches
            if b["state"] == branch["state"] and b["draw"] == branch["draw"] and b["rule"] == "uniform"
        )
        if branch["memory_signatures"] != baseline["memory_signatures"]:
            raise ValueError("Action-dependent reservoir contents/RNG")
        for a, b in zip(rows[1:], baseline["decisions"][1:], strict=True):
            if a["current_indices"] != b["current_indices"] or a["replay_indices"] != b["replay_indices"]:
                raise ValueError("Action-dependent later image selection")
    equivalent_pairs = 0
    for name, plan in plans.items():
        rules = list(plan["selections"])
        for position, rule in enumerate(rules):
            for other in rules[position + 1 :]:
                a, b = plan["selections"][rule], plan["selections"][other]
                if set(a) != set(b):
                    continue
                if a != b:
                    raise ValueError("Identical replay sets have different ordered batches")
                for draw in sorted({branch["draw"] for branch in branches}):
                    checkpoint_a = torch.load(run / f"checkpoint_{name}__draw{draw}__{rule}.pt", weights_only=False)
                    checkpoint_b = torch.load(run / f"checkpoint_{name}__draw{draw}__{other}.pt", weights_only=False)
                    if not identical(checkpoint_a, checkpoint_b):
                        raise ValueError("Identical ordered batches produce different trajectories")
                    equivalent_pairs += 1
    result = {
        "raw_verification_sha256": digest(run / "verification.json"),
        "verifier_sha256": digest(Path(__file__)),
        "readouts_recomputed": count,
        "maximum_readout_error": maximum_error,
        "branches_audited": len(branches),
        "optimizer_images_per_branch": 128,
        "memory_and_later_image_draws_equal": True,
        "frozen_hashes_verified": True,
        "equivalent_rule_pairs_verified": equivalent_pairs,
    }
    atomic_json(run / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
