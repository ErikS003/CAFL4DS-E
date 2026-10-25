"""Recalibrate and test cached linear probes using support labels only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.fresh_alpha import support_folds
from cafl4ds.image_utility import balanced_losses
from cafl4ds.reader_calibration import C_GRID, choose_c, fit_probe, probe_outcomes
from scripts.evaluate_checkpoint_robustness import digest

METRICS = [f"{m}_{z}" for m in ("refit_ce", "linear_error") for z in ("A", "B", "balanced")]


def calibrate(warm: torch.Tensor, metadata: dict[str, Any], original: dict[str, float], output: Path) -> dict[str, Any]:
    """Choose C on repeated support folds and save it before scoring any queries."""
    labels, domains = metadata["labels"], metadata["domains"]
    rows = []
    chosen = {}
    for panel, support in metadata["supports"].items():
        for seed in (77017, 78011, 78017):
            for fold, (fitting, validation) in enumerate(support_folds(labels, domains, support, seed)):
                for c in C_GRID:
                    probe = fit_probe(warm, labels, fitting, c)
                    prediction = probe.classifier.predict((warm[validation].numpy() - probe.mean) / probe.scale)
                    error = (torch.from_numpy(prediction) != labels[validation]).double()
                    ce = probe.losses(warm[validation], labels[validation])
                    rows.append(
                        {
                            "panel": panel,
                            "seed": seed,
                            "fold": fold,
                            "C": c,
                            "error": balanced_losses(error, labels[validation], domains[validation])["balanced"],
                            "ce": balanced_losses(ce, labels[validation], domains[validation])["balanced"],
                        }
                    )
        frame = pd.DataFrame(rows)
        means = frame[frame.panel == panel].groupby("C").error.mean().to_dict()
        chosen[panel] = {"accuracy_selected": choose_c(means), "ce_selected": original[panel], "fixed_0.1": 0.1}
    pd.DataFrame(rows).to_csv(output / "support_cv.csv", index=False)
    atomic_json(output / "reader_frozen.json", {"selected_C": chosen, "before_query_scoring": True})
    return chosen


def summarize(scores: pd.DataFrame, output: Path) -> None:
    """Positive gains are baseline error minus candidate error, for both metrics."""
    branches = scores[scores.kind == "branch"]
    keys = ["state", "panel", "role", "recipe", "repeat", "horizon"]
    reference = branches[branches.alpha == 1].set_index(keys)
    paired = pd.DataFrame(
        [
            {
                **{k: row[k] for k in keys},
                "alpha": row.alpha,
                **{m: float(reference.loc[tuple(row[k] for k in keys), m] - row[m]) for m in METRICS},
            }
            for _, row in branches[branches.alpha != 1].iterrows()
        ]
    )
    paired.to_csv(output / "paired_gains.csv", index=False)
    mean_keys = [k for k in keys if k != "repeat"] + ["alpha"]
    means = paired.groupby(mean_keys)[METRICS].mean().reset_index()
    means.to_csv(output / "repeat_averaged_gains.csv", index=False)
    rows = []
    for (recipe, role, horizon, alpha), group in means.groupby(["recipe", "role", "horizon", "alpha"]):
        pooled = group[group.panel == "pooled"]
        record: dict[str, Any] = {
            "recipe": recipe,
            "role": role,
            "horizon": int(horizon),
            "alpha": alpha,
            "parents": len(pooled),
        }
        for metric in ("refit_ce", "linear_error"):
            col = f"{metric}_balanced"
            signs = np.sign(group[group.panel != "pooled"].pivot(index="state", columns="panel", values=col))
            draws = paired[
                (paired.recipe == recipe)
                & (paired.role == role)
                & (paired.horizon == horizon)
                & (paired.alpha == alpha)
                & (paired.panel == "pooled")
            ]
            repeats = draws.pivot(index="state", columns="repeat", values=col)
            record.update(
                {
                    f"{metric}_gain": float(pooled[col].mean()),
                    f"{metric}_mean_abs_gain": float(pooled[col].abs().mean()),
                    f"{metric}_support_sign_agreement": int((signs.nunique(axis=1) == 1).sum()),
                    f"{metric}_support_unanimous_nonzero": int(
                        ((signs.nunique(axis=1) == 1) & (signs.iloc[:, 0] != 0)).sum()
                    ),
                    f"{metric}_support_tie_cells": int((signs == 0).sum().sum()),
                    f"{metric}_repeat_sign_agreement": int((np.sign(repeats[0]) == np.sign(repeats[1])).sum()),
                    f"{metric}_repeat_unanimous_nonzero": int(
                        ((np.sign(repeats[0]) == np.sign(repeats[1])) & (repeats[0] != 0)).sum()
                    ),
                    f"{metric}_mean_abs_repeat_difference": float((repeats[0] - repeats[1]).abs().mean()),
                    f"{metric}_positive_parents": int((pooled[col] > 0).sum()),
                    f"{metric}_zero_parents": int((pooled[col] == 0).sum()),
                    f"{metric}_domain_opposition": int((pooled[f"{metric}_A"] * pooled[f"{metric}_B"] < 0).sum()),
                }
            )
        record["accuracy_ce_sign_agreement"] = int(
            (np.sign(pooled.refit_ce_balanced) == np.sign(pooled.linear_error_balanced)).sum()
        )
        rows.append(record)
    pd.DataFrame(rows).to_csv(output / "variability.csv", index=False)
    parents = scores[scores.kind == "parent"].set_index(["state", "panel", "role", "recipe"])
    learning = pd.DataFrame(
        [
            {
                **{k: row[k] for k in keys},
                "alpha": row.alpha,
                **{m: float(parents.loc[(row.state, row.panel, row.role, row.recipe), m] - row[m]) for m in METRICS},
            }
            for _, row in branches.iterrows()
        ]
    )
    learning.to_csv(output / "parent_relative_gains.csv", index=False)
    overall = (
        learning[learning.panel == "pooled"]
        .groupby(["recipe", "role", "horizon", "alpha"])[METRICS]
        .mean()
        .reset_index()
    )
    overall.to_csv(output / "learning_summary.csv", index=False)
    controls = []
    for (recipe, panel, role), group in scores.groupby(["recipe", "panel", "role"]):
        warm = group[group.kind == "warm"].iloc[0]
        for seed in (77021, 77027):
            real = group[group.state == f"real_{seed}"].iloc[0]
            shuffled = group[group.state == f"shuffled_{seed}"].iloc[0]
            controls.append(
                {
                    "recipe": recipe,
                    "panel": panel,
                    "role": role,
                    "seed": seed,
                    **{f"warm_gain_{m}": float(warm[m] - real[m]) for m in METRICS},
                    **{f"shuffled_gain_{m}": float(shuffled[m] - real[m]) for m in METRICS},
                }
            )
    control_frame = pd.DataFrame(controls)
    control_frame.to_csv(output / "control_gains.csv", index=False)
    control_means = (
        control_frame[control_frame.panel == "pooled"].groupby(["recipe", "role"]).mean(numeric_only=True).reset_index()
    )
    atomic_json(
        output / "summary.json",
        {
            "controls": control_means.to_dict("records"),
            "variability": rows,
            "learning": overall.to_dict("records"),
            "encoder_training_performed": False,
        },
    )


def main() -> None:  # noqa: C901
    """Verify all original caches, calibrate, score, reproduce, and summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("long", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fresh, long, output = args.fresh.resolve(), args.long.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    hashes = {}
    for source in (fresh, long):
        verification = json.loads((source / "verification.json").read_text())
        for name, expected in verification["artifact_hashes"].items():
            path = source / name
            if digest(path) != expected:
                raise ValueError(f"Changed original artifact: {path}")
            hashes[str(path)] = expected
    sources = [
        Path(__file__).resolve(),
        Path("docs/project-plan/slow-edge-linear-probe-reliability.md").resolve(),
        Path("cafl4ds/reader_calibration.py").resolve(),
        Path("cafl4ds/fresh_alpha.py").resolve(),
        Path("cafl4ds/image_utility.py").resolve(),
    ]
    source_hashes = {str(p): digest(p) for p in sources}
    atomic_json(
        output / "inputs.json",
        {"cache_hashes": hashes, "source_hashes": source_hashes, "fresh": str(fresh), "long": str(long)},
    )
    metadata = torch.load(fresh / "readout_metadata.pt", weights_only=False)
    other = torch.load(long / "readout_metadata.pt", weights_only=False)
    for key in ("labels", "domains", "indices"):
        if not torch.equal(metadata[key], other[key]):
            raise ValueError(f"Readout mapping differs: {key}")
    warm = torch.load(fresh / "warm_features.pt", weights_only=True)[metadata["indices"]]
    original = json.loads((fresh / "reader_frozen.json").read_text())["selected_C"]
    old_long, old_fresh = pd.read_csv(long / "scores.csv"), pd.read_csv(fresh / "scores.csv")
    identifiers = ["kind", "state", "stream", "seed", "repeat", "alpha", "horizon"]
    states = old_long[identifiers].drop_duplicates().to_dict("records")
    for state in ("real_77021", "shuffled_77021", "real_77027", "shuffled_77027"):
        row = old_fresh[old_fresh.state == state].iloc[0]
        states.append(
            {
                "kind": "control",
                "state": state,
                "stream": "",
                "seed": int(row.seed),
                "repeat": -1,
                "alpha": -1,
                "horizon": 0,
            }
        )
    if len(states) != 119:
        raise ValueError(f"Expected 119 states, got {len(states)}")
    records = []
    max_error = 0.0
    with threadpool_limits(limits=1):
        chosen = calibrate(warm, metadata, original, output)
        print(f"Frozen C: {chosen}", flush=True)
        for number, state in enumerate(states):
            kind, name = state["kind"], state["state"]
            if kind == "warm":
                features = warm
            elif kind == "control":
                features = torch.load(fresh / f"features_{name}.pt", weights_only=True)[metadata["indices"]]
            elif kind == "parent":
                features = torch.load(long / f"features_parent_{name}.pt", weights_only=True)
            else:
                filename = f"features_{name}_{int(state['repeat'])}_{state['alpha']}_{int(state['horizon'])}.pt"
                features = torch.load(long / filename, weights_only=True)
            old = (
                old_fresh[old_fresh.state == name]
                if kind == "control"
                else old_long[
                    (old_long.kind == kind) & (old_long.state == name) & (old_long.horizon == state["horizon"])
                ]
            )
            if kind == "branch":
                old = old[(old.repeat == state["repeat"]) & (old.alpha == state["alpha"])]
            for panel, support in metadata["supports"].items():
                fitted = {}
                for recipe, c in chosen[panel].items():
                    if c not in fitted:
                        fitted[c] = probe_outcomes(
                            features, metadata["labels"], metadata["domains"], support, metadata["roles"], c
                        )
                    for role, metrics in fitted[c].items():
                        records.append({**state, "panel": panel, "role": role, "recipe": recipe, "C": c, **metrics})
                        if recipe == "ce_selected":
                            baseline = old[(old.panel == panel) & (old.role == role)]
                            if len(baseline) != 1:
                                raise ValueError(f"Ambiguous original match: {state}, {panel}, {role}")
                            max_error = max(max_error, *(abs(metrics[m] - float(baseline.iloc[0][m])) for m in METRICS))
            atomic_json(output / "status.json", {"state": "scoring", "completed": number + 1, "total": len(states)})
            if (number + 1) % 10 == 0:
                print(f"Scored {number + 1}/119 states", flush=True)
    scores = pd.DataFrame(records)
    if len(scores) != 2856 or max_error > 1e-10:
        raise ValueError(f"Reproduction failed: rows={len(scores)}, error={max_error}")
    scores.to_csv(output / "scores.csv", index=False)
    summarize(scores, output)
    for path, expected in {**hashes, **source_hashes}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Input changed during scoring: {path}")
    atomic_json(
        output / "verification.json",
        {
            "states": len(states),
            "rows": len(scores),
            "original_max_error": max_error,
            "elapsed_seconds": time.time() - started,
            "artifact_hashes": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"},
        },
    )
    atomic_json(output / "status.json", {"state": "complete", "states": len(states), "rows": len(scores)})
    print(f"Complete: 119 states, 2856 rows, original error {max_error:.3g}", flush=True)


if __name__ == "__main__":
    main()
