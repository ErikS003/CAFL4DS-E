"""Assess two frozen hypotheses without promoting winners from secondary comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cafl4ds.artifacts import atomic_json

SEEDS = [6601, 6607, 6611, 6617]
HYPOTHESES = {
    "raw_context": ("raw/loss_coverage_context", "raw/loss_coverage", "raw/context_total"),
    "centered_health": ("centered/image_health_interactions", "centered/image_only", "centered/full_health"),
}


def assess(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared mean-gain gate and expose state/seed/transition variation."""
    if report["horizon"] != 4 or report["primary_origin"] != 36:
        raise ValueError("Confirmation requires four-update outcomes at origin 36")
    fold = report["folds"]["36"]
    hypotheses = {}
    for name, (model, baseline, contrast) in HYPOTHESES.items():
        roles = {}
        for role in ("test_seed", "test_time"):
            metrics = fold["tests"][role][model]
            difference = fold["matched_contrasts"][role][contrast]
            roles[role] = {
                "gain_over_random": metrics["gain_over_random"],
                "gain_over_baseline": difference["mean_gain_difference"],
                "passes": metrics["gain_over_random"] > 0 and difference["mean_gain_difference"] > 0,
                "positive_snapshot_fraction_over_baseline": difference["positive_snapshot_fraction"],
                "added_value_by_seed": difference["by_seed"],
            }
        time = fold["tests"]["test_time"][model]["by_snapshot"]
        by_step = {
            str(step): sum(row["gain_over_random"] for row in time if row["step"] == step)
            / sum(row["step"] == step for row in time)
            for step in sorted({row["step"] for row in time})
        }
        hypotheses[name] = {
            "model": model,
            "baseline": baseline,
            "qualifies_for_batch_development": all(value["passes"] for value in roles.values()),
            "tests": roles,
            "time_gain_over_random_by_step": by_step,
            "transition_gain_over_random": by_step["36"],
            "frozen_domain_gains": {
                role: {domain: scores[model]["gain_over_random"] for domain, scores in domains.items()}
                for role, domains in fold["frozen_domain_tests"].items()
            },
        }
    return {
        "status": "completed_confirmation_development",
        "horizon": 4,
        "origin": 36,
        "hypotheses": hypotheses,
        "qualified_hypotheses": [
            name for name, value in hypotheses.items() if value["qualifies_for_batch_development"]
        ],
        "limits": [
            "Two fixed descriptive hypotheses; no significance claim or secondary-winner substitution",
            "Shared reference well and evaluation corpus; disjoint training streams remain development",
            "Single-slot effects do not establish full-batch or MPC efficacy",
            "Transition and domain results are diagnostics, not post-hoc gating substitutions",
        ],
    }


def main() -> None:
    """Verify the complete collection and exclusion contributors, then write the frozen verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "inputs.json").read_text())
    config = manifest["configuration"]
    if config["seeds"] != SEEDS or config["stream_seed"] != 66003:
        raise ValueError("Confirmation seeds/order differ from the frozen specification")
    audit = json.loads((directory / "collection_audit.json").read_text())
    if audit["state"] != "completed" or audit["rows"] != 960:
        raise ValueError("Confirmation requires all 960 audited measurements")
    exclusions = json.loads(Path(config["exclude_manifest"]).read_text())
    union: set[str] = set()
    for source in exclusions["contributors"]:
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("An exclusion contributor changed")
        union.update(json.loads(path.read_text())["train_paths"])
    if len(exclusions["contributors"]) != 2 or union != set(exclusions["train_paths"]):
        raise ValueError("Exclusion union must exactly cover the two completed studies")
    if union & set(manifest["train_paths"]):
        raise ValueError("Confirmation stream overlaps a previous study")
    result = assess(json.loads((directory / "context_prediction_h4.json").read_text()))
    result["provenance"] = {
        "previous_study_unique_images": len(union),
        "fresh_unique_images": len(set(manifest["train_paths"])),
        "previous_overlap": 0,
        "contributors": exclusions["contributors"],
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(directory / "confirmation.json", result)
    lines = [
        "# Frozen image-utility confirmation",
        "",
        "Four-update, origin-36 held-out CE gain.",
        "",
        "| Hypothesis | Seed above random | Seed added value | Time above random | Time added value | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, value in result["hypotheses"].items():
        seed, time = [value["tests"][role] for role in ("test_seed", "test_time")]
        gate = "PASS" if value["qualifies_for_batch_development"] else "FAIL"
        lines.append(
            f"| {name} | {seed['gain_over_random']:.6g} | {seed['gain_over_baseline']:.6g} | "
            f"{time['gain_over_random']:.6g} | {time['gain_over_baseline']:.6g} | {gate} |"
        )
    lines += ["", "## Transition diagnostic", ""]
    for name, value in result["hypotheses"].items():
        lines.append(f"- {name}: {value['time_gain_over_random_by_step']}")
    lines += ["", "## Limits", "", *[f"- {limit}." for limit in result["limits"]]]
    (directory / "confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["status"], "qualified_hypotheses": result["qualified_hypotheses"]}))


if __name__ == "__main__":
    main()
