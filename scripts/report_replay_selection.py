"""Report replay-content effects, numerical sensitivity and cross-draw opportunity."""

# ruff: noqa: E501
# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from cafl4ds.artifacts import atomic_json
from cafl4ds.replay_selection import RULES, cross_draw_opportunities
from scripts.evaluate_checkpoint_robustness import digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:  # noqa: C901
    """Keep solver stability and future-label diagnostics separate from policy claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    raw = json.loads((run / "verification.json").read_text())
    for name, expected in raw["artifact_hashes"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Changed collected artifact: {name}")
    inputs = json.loads((run / "inputs.json").read_text())
    primary_horizon = inputs.get("primary_horizon", 1)
    scores = pd.read_csv(run / "scores.csv")
    queries = scores.loc[(scores.draw >= 0) & scores.panel.isin(["pooled", "support1", "support2", "support3"])].copy()
    keys = ["state", "pool_group", "draw", "horizon", "panel", "role", "tolerance"]
    reference = queries.loc[
        queries.rule == "uniform", keys + ["accuracy_balanced", "accuracy_A", "accuracy_B", "ce_balanced"]
    ]
    paired = queries.merge(reference, on=keys, suffixes=("", "_uniform"), validate="many_to_one")
    for metric in ("accuracy_balanced", "accuracy_A", "accuracy_B", "ce_balanced"):
        paired[f"{metric}_gain"] = (paired[metric] - paired[f"{metric}_uniform"]) * (
            -1 if metric == "ce_balanced" else 1
        )
    paired.to_csv(run / "paired_gains.csv", index=False)
    low = paired.loc[paired.tolerance == 1e-8]
    high = paired.loc[paired.tolerance == 1e-10]
    joined = low.merge(high, on=keys[:-1] + ["rule"], suffixes=("_low", "_high"), validate="one_to_one")
    maximum_change = float((joined.accuracy_balanced_gain_low - joined.accuracy_balanced_gain_high).abs().max())
    primary = high.loc[(high.panel == "pooled") & (high.role == "holdout")].copy()
    primary["gain_pp"] = primary.accuracy_balanced_gain * 100
    summary = primary.groupby(["horizon", "rule"], as_index=False)["gain_pp"].agg(["mean", "min", "max"])
    summary.to_csv(run / "rule_summary.csv", index=False)
    primary.groupby(["horizon", "pool_group", "draw", "rule"], as_index=False)["gain_pp"].mean().to_csv(
        run / "pool_draw_summary.csv", index=False
    )
    opportunity = []
    for (state, horizon), group in primary.groupby(["state", "horizon"]):
        values = group.pivot(index="draw", columns="rule", values="accuracy_balanced").loc[:, list(RULES)].to_numpy()
        result = cross_draw_opportunities(values, inputs["minimum_gain"])
        for fold in result["folds"]:
            opportunity.append({"state": state, "horizon": horizon, "pool_group": group.iloc[0].pool_group, **fold})
    pd.DataFrame(opportunity).to_csv(run / "cross_draw_opportunity.csv", index=False)
    gates = json.loads((run / "frozen_health_gates.json").read_text())
    gate_rows = []
    for choice in gates["choices"]:
        evaluated = primary.loc[
            (primary.state == choice["state"]) & (primary.rule == choice["rule"]) & (primary.draw > 0)
        ]
        for row in evaluated.to_dict("records"):
            gate_rows.append({**choice, "draw": row["draw"], "horizon": row["horizon"], "gain_pp": row["gain_pp"]})
    pd.DataFrame(
        gate_rows, columns=["state", "heldout_pool", "recipe", "predicted_gain", "rule", "draw", "horizon", "gain_pp"]
    ).to_csv(run / "health_gate_evaluation.csv", index=False)
    forecasts = pd.read_csv(run / "batch_forecasts.csv")
    task = scores.loc[(scores.draw >= 0) & (scores.tolerance == 1e-10) & (scores.panel == "task0")]
    task_reference = task.loc[task.rule == "uniform", ["state", "draw", "horizon", "role", "ce_balanced"]]
    task = task.merge(task_reference, on=["state", "draw", "horizon", "role"], suffixes=("", "_uniform"))
    task["ce_gain"] = task.ce_balanced_uniform - task.ce_balanced
    forecast_ref = forecasts.loc[
        (forecasts.rotation == 0) & (forecasts.rule == "uniform"), ["state", "predicted_ce_reduction"]
    ]
    forecasts = forecasts.loc[forecasts.rotation == 0].merge(forecast_ref, on="state", suffixes=("", "_uniform"))
    forecasts["predicted_gain"] = forecasts.predicted_ce_reduction - forecasts.predicted_ce_reduction_uniform
    task = task.merge(forecasts[["state", "rule", "predicted_gain"]], on=["state", "rule"])
    task.to_csv(run / "forecast_outcomes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = ["#2b7bba", "#d66a38", "#525866"]
    for ax, horizon in zip(axes, (1, 4), strict=True):
        subset = primary.loc[(primary.horizon == horizon) & (primary.rule != "uniform")]
        for position, rule in enumerate(RULES[1:]):
            means = subset.loc[subset.rule == rule].groupby("draw").gain_pp.mean()
            ax.bar(position, means.mean(), color=colors[position], alpha=0.6)
            offsets = np.linspace(-0.12, 0.12, len(means))
            ax.scatter(position + offsets, means, color=colors[position], edgecolor="black", zorder=3)
        ax.axhline(0, color="black", linewidth=0.8)
        lower, upper = ax.get_ylim()
        span = max(upper - lower, 0.01)
        ax.set_ylim(min(lower, -0.05 * span), max(upper, 0.05 * span))
        ax.set_xticks(range(3), ["Coverage", "Task gradient*", "Mixed*"])
        ax.set_title(f"After {horizon} update{'s' if horizon > 1 else ''}")
        ax.set_ylabel("Balanced accuracy gain vs uniform (pp)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Replay selection: bars = mean; dots = training-draw means\n*Uses labeled task support; smoke is execution validation only"
        if inputs["smoke"]
        else "Replay selection: bars = mean; dots = training-draw means\n*Uses labeled task support",
        fontsize=12,
    )
    fig.savefig(run / "selection_gains.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    for ax, horizon in zip(axes, (1, 4), strict=True):
        rows = [r for r in opportunity if r["horizon"] == horizon]
        realized = np.array([r["gain"] for r in rows]) * 100
        data = primary.loc[primary.horizon == horizon]
        hindsight = data.groupby(["state", "draw"]).accuracy_balanced_gain.max().mean() * 100
        bars = ax.bar([0, 1], [hindsight, realized.mean()], color=["#b9bec6", "#2b7bba"])
        ax.bar_label(bars, fmt="%+.3f", padding=4)
        ax.axhline(0, color="black", linewidth=0.8)
        lower, upper = ax.get_ylim()
        span = max(upper - lower, 0.01)
        ax.set_ylim(min(lower, -0.05 * span), max(upper, 0.05 * span) + 0.1 * span)
        ax.set_xticks([0, 1], ["Same-draw\nhindsight", "Other-draw\nselection"])
        ax.set_ylabel("Accuracy gain vs uniform (pp)")
        ax.set_title(f"After {horizon} update{'s' if horizon > 1 else ''}")
    fig.suptitle("Future-label diagnostic: does the apparent opportunity transfer across draws?")
    fig.savefig(run / "selection_opportunity.png", dpi=160)
    plt.close(fig)
    numeric_pass = maximum_change <= inputs["numerical_contrast_limit"]
    primary_means = primary.loc[primary.horizon == primary_horizon].groupby("rule").gain_pp.mean().to_dict()
    state = "SMOKE: execution validation only" if inputs["smoke"] else "Exploratory study on reused pools"
    report = f"""# Fixed-allocation replay selection

**{state}.** This compares which16 old images are replayed alongside the same16 current images. Only the first update changes; the next three use common uniform replay. Task-gradient and mixed selection use support labels. Coverage/uniform generation and health-gate features are unlabeled.

## Primary: {primary_horizon}-update held-out pooled accuracy

Rule-minus-uniform mean gains (percentage points): {json.dumps(primary_means, sort_keys=True)}.

![Paired selection gains](selection_gains.png)

Dots are mean gains for each training draw, not confidence intervals. See `pool_draw_summary.csv` and `paired_gains.csv` for pool, reader, domain and CE contrasts. The minimum interesting gain is .1pp; three original pool groups cannot support a generalization claim.

## Numerical gate and opportunity

Maximum paired accuracy-contrast change from tolerance1e-8 to1e-10: **{maximum_change * 100:.6f}pp**. Predeclared <=.01pp gate: **{"PASS" if numeric_pass else "FAIL"}**. This two-tolerance check does not prove universal solver convergence.

![Cross-draw opportunity](selection_opportunity.png)

Same-draw hindsight uses scored outcomes to choose and is optimistic. Other-draw selection chooses using the remaining draws and evaluates an untouched draw, with uniform fallback below .1pp. Both use future query labels and are diagnostic, not deployable controllers or proven upper bounds.

## Gradient and health interpretation

`forecast_outcomes.csv` compares final-batch AdamW CE forecasts with frozen-head outcomes on gradient support2 and third support3. Block ranking used24-image surrogate steps; final-batch forecasts use32. Local CE agreement must not be called a query accuracy improvement.

`health_gate_evaluation.csv` evaluates frozen Ridge coverage/uniform gates fitted to draw0 development accuracy at the declared target horizon from other pools. All coefficients/choices are frozen before evaluation draws. Constant and reconstruction recipes are controls. Smoke deliberately fits no response model because it lacks three pool groups. Even the full study remains small and reuses previously inspected data.

## Reproduction and budget

{raw["new_optimizer_updates"]} actual optimizer updates; {raw["virtual_steps"]} extra copied observation steps. Every real branch uses128 optimizer-image exposures. Source/input/frozen-selection hashes are in `verification.json`; heads, features, decisions and checkpoints are retained. `pretraining_verification.json` and `health_pretraining_verification.json` record phase boundaries. No closed-loop MPC claim.
"""
    (run / "report.md").write_text(report, encoding="utf-8")
    atomic_json(
        run / "report_verification.json",
        {
            "source_sha256": digest(Path(__file__)),
            "raw_verification_sha256": digest(run / "verification.json"),
            "numerical_gate_pass": numeric_pass,
            "maximum_accuracy_contrast_change_pp": maximum_change * 100,
            "primary_mean_gains_pp": primary_means,
            "smoke": inputs["smoke"],
            "artifact_hashes": {
                name: digest(run / name)
                for name in (
                    "paired_gains.csv",
                    "rule_summary.csv",
                    "pool_draw_summary.csv",
                    "cross_draw_opportunity.csv",
                    "health_gate_evaluation.csv",
                    "forecast_outcomes.csv",
                    "selection_gains.png",
                    "selection_opportunity.png",
                    "report.md",
                )
            },
        },
    )
    print(report[:1200])


if __name__ == "__main__":
    main()
