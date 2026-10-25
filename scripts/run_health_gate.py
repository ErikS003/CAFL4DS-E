"""Cross-stream selective alpha steering with development-only empirical margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cafl4ds.alpha_health_policy import ACTIONS, fit
from cafl4ds.artifacts import atomic_json
from scripts.evaluate_checkpoint_robustness import digest

RECIPES = ("reconstruction_coverage", "reconstruction_health", "health_only", "reconstruction_gradients")
METRICS = ("linear_error_balanced", "linear_error_B", "refit_ce_balanced")
TARGETS = ("mean_accuracy", "mean_night", "robust_accuracy", "robust_night")


def select(pred: dict[str, dict[float, float]], margins: dict[str, float], gate: str, threshold: float) -> float:
    """Return reference unless an action passes the frozen gate; labels never enter."""
    accuracy, night = (
        ("robust_accuracy", "robust_night") if gate == "empirical_margin" else ("mean_accuracy", "mean_night")
    )
    eligible = []
    for action in ACTIONS[1:]:
        lower = pred[accuracy][action] - (margins[accuracy] if gate == "empirical_margin" else 0)
        night_lower = pred[night][action] - (margins[night] if gate == "empirical_margin" else 0)
        if lower > threshold and (gate == "positive_accuracy" or night_lower >= 0):
            eligible.append(action)
    return max(eligible, key=lambda a: pred[accuracy][a]) if eligible else 1.0


def main() -> None:  # noqa: C901
    """Fit, freeze, then score; all reused outcomes are explicitly exploratory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/image-utility/20261008-alpha-health-short"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    verification = json.loads((source / "verification.json").read_text())
    input_hashes = {}
    for name in ("scores.csv", "predecision_features.json"):
        path = source / name
        if digest(path) != verification["artifact_hashes"][name]:
            raise ValueError(f"Changed input: {path}")
        input_hashes[str(path)] = digest(path)
    protocol = root / "docs/project-plan/slow-edge-health-gate.md"
    input_hashes[str(protocol)] = digest(protocol)
    input_hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    input_hashes[str(root / "cafl4ds/alpha_health_policy.py")] = digest(root / "cafl4ds/alpha_health_policy.py")
    features = json.loads((source / "predecision_features.json").read_text())["rows"]
    scores = pd.read_csv(source / "scores.csv")
    # Only development outcomes are consulted before the decision-freeze boundary.
    development = scores[(scores.role == "development") & (scores.recipe == "accuracy_selected")]
    indexed = development.set_index(["state", "horizon", "alpha", "panel", "repeat"]).sort_index()
    responses = {}
    for horizon in (4, 12):
        for row in features:
            if row["alpha"] == 1:
                continue
            values: dict[str, list[float]] = {k: [] for k in TARGETS}
            for panel in ("pooled", "support1", "support2", "support3"):
                for repeat in (0, 1):
                    key = (row["state"], horizon)
                    reference = indexed.loc[(*key, 1.0, panel, repeat)]
                    candidate = indexed.loc[(*key, row["alpha"], panel, repeat)]
                    prefix = "mean" if panel == "pooled" else "robust"
                    for suffix, metric in (("accuracy", "linear_error_balanced"), ("night", "linear_error_B")):
                        values[f"{prefix}_{suffix}"].append(float(reference[metric] - candidate[metric]))
            responses[(horizon, row["state"], row["alpha"])] = {
                k: float(np.mean(v) if k.startswith("mean") else np.min(v)) for k, v in values.items()
            }
    decisions, model_records, calibration, prediction_rows = [], [], [], []
    streams = sorted({r["stream"] for r in features})
    for horizon in (4, 12):
        for heldout in streams:
            train = [r for r in features if r["stream"] != heldout and r["alpha"] != 1]
            test = [r for r in features if r["stream"] == heldout]
            for recipe in RECIPES:
                margins, models = {}, {}
                for target in TARGETS:
                    fitting = [{**r, "target": responses[(horizon, r["state"], r["alpha"])][target]} for r in train]
                    errors = []
                    for inner in sorted({r["stream"] for r in fitting}):
                        inner_fit = [r for r in fitting if r["stream"] != inner]
                        validation = [r for r in fitting if r["stream"] == inner]
                        model = fit(inner_fit, recipe, 1000.0)
                        for row, predicted in zip(validation, model.predict(validation), strict=True):
                            error = float(predicted - row["target"])
                            errors.append(error)
                            calibration.append(
                                {
                                    "horizon": horizon,
                                    "heldout_stream": heldout,
                                    "recipe": recipe,
                                    "target": target,
                                    "validation_stream": inner,
                                    "state": row["state"],
                                    "alpha": row["alpha"],
                                    "prediction": float(predicted),
                                    "observed": row["target"],
                                    "optimism": error,
                                }
                            )
                    margins[target] = max(0.0, *errors)
                    model = fit(fitting, recipe, 1000.0)
                    models[target] = model
                    model_records.append(
                        {
                            "horizon": horizon,
                            "heldout_stream": heldout,
                            "recipe": recipe,
                            "target": target,
                            "fit_states": sorted({r["state"] for r in fitting}),
                            "margin": margins[target],
                            **model.as_dict(),
                        }
                    )
                for state in sorted({r["state"] for r in test}):
                    rows = [r for r in test if r["state"] == state]
                    pred = {
                        t: dict(zip([r["alpha"] for r in rows], models[t].predict(rows), strict=True)) for t in TARGETS
                    }
                    for target in TARGETS:
                        for action in ACTIONS:
                            prediction_rows.append(
                                {
                                    "horizon": horizon,
                                    "heldout_stream": heldout,
                                    "recipe": recipe,
                                    "state": state,
                                    "target": target,
                                    "alpha": action,
                                    "prediction": float(pred[target][action]),
                                    "margin": margins[target],
                                }
                            )
                    gates = [("positive_accuracy", 0.0), ("night_guard", 0.0)] + [
                        ("empirical_margin", v / 100) for v in (0.0, 0.05, 0.1, 0.25, 0.5)
                    ]
                    for gate, threshold in gates:
                        decisions.append(
                            {
                                "horizon": horizon,
                                "heldout_stream": heldout,
                                "recipe": recipe,
                                "state": state,
                                "gate": gate,
                                "threshold_pp": threshold * 100,
                                "alpha": select(pred, margins, gate, threshold),
                            }
                        )
    pd.DataFrame(decisions).to_csv(output / "choices.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    atomic_json(output / "models.json", {"models": model_records})
    choice_hash = digest(output / "choices.csv")
    # Frozen choices: replication and held-out support/draw/C outcomes enter only here.
    outcome = scores.set_index(["state", "horizon", "alpha"]).sort_index()
    evaluations = []
    for decision in decisions:
        key = (decision["state"], decision["horizon"])
        reference = outcome.loc[(*key, 1.0)].set_index(["panel", "role", "recipe", "repeat"])
        candidate = outcome.loc[(*key, decision["alpha"])].set_index(["panel", "role", "recipe", "repeat"])
        for index, row in (reference[list(METRICS)] - candidate[list(METRICS)]).iterrows():
            evaluations.append(
                {
                    **decision,
                    "panel": index[0],
                    "role": index[1],
                    "reader": index[2],
                    "repeat": index[3],
                    "switched": decision["alpha"] != 1.0,
                    **row.to_dict(),
                }
            )
    frame = pd.DataFrame(evaluations)
    frame.to_csv(output / "gains.csv", index=False)
    primary = frame[(frame.panel == "pooled") & (frame.reader == "accuracy_selected")]
    keys = ["horizon", "recipe", "gate", "threshold_pp", "role"]
    summary = primary.groupby(keys)[list(METRICS)].mean().reset_index()
    coverage = primary.groupby(keys).switched.mean().rename("coverage")
    conditional = primary[primary.switched].groupby(keys).linear_error_balanced.mean().rename("conditional_accuracy")
    summary = summary.merge(coverage, on=keys).merge(conditional, on=keys, how="left")
    summary.to_csv(output / "summary.csv", index=False)
    primary.groupby([*keys, "heldout_stream"])[list(METRICS)].mean().reset_index().to_csv(
        output / "streams.csv", index=False
    )
    oracle = []
    for horizon in (4, 12):
        for state in sorted({r["state"] for r in features}):
            for action in (0.5, 0.0):
                contrast = (
                    outcome.loc[(state, horizon, 1.0)].set_index(["panel", "role", "recipe", "repeat"])[list(METRICS)]
                    - outcome.loc[(state, horizon, action)].set_index(["panel", "role", "recipe", "repeat"])[
                        list(METRICS)
                    ]
                )
                for role in ("development", "holdout"):
                    selected = contrast.xs(role, level="role")
                    pooled = selected.xs(("pooled", "accuracy_selected"), level=("panel", "recipe"))
                    disjoint = selected[selected.index.get_level_values("panel") != "pooled"]
                    disjoint = disjoint[disjoint.index.get_level_values("recipe") == "accuracy_selected"]
                    oracle.append(
                        {
                            "horizon": horizon,
                            "state": state,
                            "alpha": action,
                            "role": role,
                            "pooled_accuracy_pp": pooled.linear_error_balanced.mean() * 100,
                            "worst_disjoint_accuracy_pp": disjoint.linear_error_balanced.min() * 100,
                            "worst_disjoint_night_pp": disjoint.linear_error_B.min() * 100,
                            "worst_all_accuracy_pp": selected.linear_error_balanced.min() * 100,
                            "worst_all_night_pp": selected.linear_error_B.min() * 100,
                        }
                    )
    pd.DataFrame(oracle).to_csv(output / "opportunities.csv", index=False)
    for input_path, value in input_hashes.items():
        if digest(Path(input_path)) != value:
            raise ValueError(f"Input changed: {input_path}")
    if digest(output / "choices.csv") != choice_hash:
        raise ValueError("Choices changed after freeze")
    atomic_json(
        output / "verification.json",
        {
            "exploratory": True,
            "fresh_query_confirmation": False,
            "distinct_streams": len(streams),
            "new_ssl_updates": 0,
            "decisions": len(decisions),
            "models": len(model_records),
            "input_hashes": input_hashes,
            "choice_hash_before_evaluation": choice_hash,
            "artifact_hashes": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
