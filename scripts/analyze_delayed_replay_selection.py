"""Answer whether coverage repeats and frozen health adds value over always using it."""

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
from scripts.evaluate_checkpoint_robustness import digest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:  # noqa: C901
    """Use all draws for coverage, evaluation draws only for frozen health choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    report = json.loads((run / "report_verification.json").read_text())
    for name, expected in report["artifact_hashes"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Changed report input: {name}")
    gates = json.loads((run / "frozen_health_gates.json").read_text())
    if gates.get("target_horizon") != 4:
        raise ValueError("This analysis requires frozen delayed-benefit gates")
    gains = pd.read_csv(run / "paired_gains.csv")
    query = gains.loc[
        (gains.horizon == 4) & (gains.panel == "pooled") & (gains.role == "holdout") & (gains.tolerance == 1e-10)
    ].copy()
    coverage = query.loc[query.rule == "coverage"].copy()
    coverage["gain_pp"] = coverage.accuracy_balanced_gain * 100
    coverage["retention_gain_pp"] = (
        np.where(coverage.state.str.contains("day_then_night"), coverage.accuracy_A_gain, coverage.accuracy_B_gain)
        * 100
    )
    pool_draw = coverage.groupby(["pool_group", "draw"], as_index=False)[["gain_pp", "retention_gain_pp"]].mean()
    pool_draw.to_csv(run / "coverage_delayed_pool_draw.csv", index=False)
    state_draw = coverage.pivot(index="state", columns="draw", values="gain_pp")
    state_draw.to_csv(run / "coverage_delayed_states.csv")
    readers = (
        gains.loc[
            (gains.horizon == 4) & (gains.role == "holdout") & (gains.tolerance == 1e-10) & (gains.rule == "coverage")
        ]
        .groupby("panel")["accuracy_balanced_gain"]
        .mean()
        .mul(100)
    )
    predictors, policy_rows = [], []
    for recipe in sorted({c["recipe"] for c in gates["choices"]}):
        for choice in [c for c in gates["choices"] if c["recipe"] == recipe]:
            rows = query.loc[(query.state == choice["state"]) & (query.draw > 0)]
            for draw, frame in rows.groupby("draw"):
                values = frame.set_index("rule")
                realized = float(values.loc["coverage", "accuracy_balanced_gain"])
                actual = float(values.loc[choice["rule"], "accuracy_balanced_gain"])
                constant = next(
                    c for c in gates["choices"] if c["state"] == choice["state"] and c["recipe"] == "constant"
                )
                predictors.append(
                    {
                        "state": choice["state"],
                        "pool_group": choice["heldout_pool"],
                        "draw": draw,
                        "recipe": recipe,
                        "prediction": choice["predicted_gain"],
                        "actual_coverage_gain": realized,
                        "constant_prediction": constant["predicted_gain"],
                    }
                )
                policy_rows.append(
                    {
                        "state": choice["state"],
                        "pool_group": choice["heldout_pool"],
                        "draw": draw,
                        "recipe": recipe,
                        "selected_rule": choice["rule"],
                        "gain_vs_uniform_pp": actual * 100,
                        "gain_vs_always_coverage_pp": (actual - realized) * 100,
                    }
                )
    policies = pd.DataFrame(policy_rows)
    policies.to_csv(run / "delayed_health_policy.csv", index=False)
    forecasts = pd.DataFrame(predictors)
    forecasts.to_csv(run / "delayed_health_predictions.csv", index=False)
    summary = []
    for recipe, rows in forecasts.groupby("recipe"):
        y = rows.actual_coverage_gain.to_numpy()
        prediction = rows.prediction.to_numpy()
        mse = float(np.mean((prediction - y) ** 2))
        reference_mse = float(np.mean((rows.constant_prediction.to_numpy() - y) ** 2))
        corr = float(np.corrcoef(prediction, y)[0, 1]) if np.std(prediction) > 1e-14 and np.std(y) > 1e-14 else None
        chosen = policies.loc[policies.recipe == recipe]
        summary.append(
            {
                "recipe": recipe,
                "forecast_correlation": corr,
                "mse_skill_vs_constant": 1 - mse / reference_mse if reference_mse else None,
                "gain_vs_uniform_pp": float(chosen.gain_vs_uniform_pp.mean()),
                "gain_vs_always_coverage_pp": float(chosen.gain_vs_always_coverage_pp.mean()),
                "coverage_choices": sum(c["rule"] == "coverage" for c in gates["choices"] if c["recipe"] == recipe),
                "states": len(gates["choices"]) // 3,
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    matrix = pool_draw.pivot(index="pool_group", columns="draw", values="gain_pp")
    matrix.plot.bar(ax=axes[0], color=["#777d87", "#2b7bba", "#d66a38"], rot=0)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Coverage vs uniform after four updates")
    axes[0].set_ylabel("Balanced linear-probe accuracy gain (pp)")
    axes[0].set_xlabel("Original image pool")
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, ["Fitting draw 0", "Evaluation draw 1", "Evaluation draw 2"], frameon=False)
    policy_summary = policies.groupby("recipe")[["gain_vs_uniform_pp", "gain_vs_always_coverage_pp"]].mean()
    policy_summary.rename(
        columns={"gain_vs_uniform_pp": "Vs uniform", "gain_vs_always_coverage_pp": "Vs always coverage"}
    ).plot.bar(ax=axes[1], color=["#2b7bba", "#d66a38"], rot=0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Frozen gates: evaluation draws only")
    axes[1].set_ylabel("Balanced linear-probe accuracy gain (pp)")
    axes[1].set_xlabel("Gate recipe")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    axes[1].set_ylim(-0.002, 0.018)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        lower, upper = ax.get_ylim()
        span = max(upper - lower, 0.01)
        ax.set_ylim(min(lower, -0.05 * span), max(upper, 0.05 * span))
    fig.savefig(run / "delayed_coverage_health.png", dpi=160)
    plt.close(fig)
    result = {
        "coverage_mean_gain_pp": float(coverage.gain_pp.mean()),
        "coverage_fitting_draw_gain_pp": float(coverage.loc[coverage.draw == 0, "gain_pp"].mean()),
        "coverage_evaluation_draws_gain_pp": float(coverage.loc[coverage.draw > 0, "gain_pp"].mean()),
        "coverage_retention_mean_gain_pp": float(coverage.retention_gain_pp.mean()),
        "states_positive_in_all_draws": int((state_draw.min(axis=1) > 0).sum()),
        "states_negative_in_all_draws": int((state_draw.max(axis=1) < 0).sum()),
        "states": len(state_draw),
        "pool_draw_means": pool_draw.to_dict("records"),
        "reader_means_pp": readers.to_dict(),
        "numerical_gate_pass": report["numerical_gate_pass"],
        "health": summary,
        "fresh_dataset_confirmation": False,
    }
    atomic_json(run / "delayed_result.json", result)
    lines = [
        "# Does coverage's delayed benefit repeat, and can health select it?",
        "",
        "Primary: four-update pooled held-out balanced linear-probe accuracy. One selection differs; the next three updates are common uniform replay. Three original pools and previously inspected panels are reused.",
        "",
        f"Coverage mean gain vs uniform: **{result['coverage_mean_gain_pp']:+.4f}pp**. Retained-domain mean gain: **{result['coverage_retention_mean_gain_pp']:+.4f}pp**.",
        f"The fitting draw averages **{result['coverage_fitting_draw_gain_pp']:+.4f}pp**; the two evaluation draws average **{result['coverage_evaluation_draws_gain_pp']:+.4f}pp**. The overall mean is below the predefined 0.1pp meaningful-gain threshold and is driven by the fitting draw.",
        f"States positive on all training draws: **{result['states_positive_in_all_draws']}/{len(state_draw)}**; negative on all draws: **{result['states_negative_in_all_draws']}/{len(state_draw)}**.",
        f"Numerical contrast gate: **{'PASS' if result['numerical_gate_pass'] else 'FAIL'}**. See every pool/draw and alternate reader before inferring reliability.",
        "",
        "![Coverage and frozen health gates](delayed_coverage_health.png)",
        "",
        "| Gate | Gain vs uniform (pp) | Gain vs always coverage (pp) | Coverage choices | Forecast correlation | MSE skill vs constant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        correlation = f"{row['forecast_correlation']:.3f}" if row["forecast_correlation"] is not None else "undefined"
        skill = f"{row['mse_skill_vs_constant']:.1%}" if row["mse_skill_vs_constant"] is not None else "undefined"
        lines.append(
            f"| {row['recipe']} | {row['gain_vs_uniform_pp']:+.4f} | {row['gain_vs_always_coverage_pp']:+.4f} | {row['coverage_choices']}/{row['states']} | {correlation} | {skill} |"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        "Coverage's pilot benefit did not repeat broadly. Only one of twelve states was positive on every draw, and that is the original pilot state. Alternate support panels also change the mean benefit's sign. This is weak evidence for a reliably useful selection action at this horizon.",
        "",
        "Every frozen gate chose uniform in every state. Health therefore added no decision benefit over the constant gate or uniform replay. Its small gain over always using coverage comes entirely from avoiding coverage, and the constant gate achieved the same result. Health-plus-SSL forecasts were worse than the constant baseline. These findings do not justify a health-driven controller with this recipe.",
        "",
        "Health must improve on a constant gate and always using coverage to demonstrate useful steering. Beating uniform alone can reflect a useful static filter. Gate fitting uses other pools' draw0 development four-update accuracy; evaluation uses pooled holdout on draws1/2. Inputs and choices were frozen before those branches. No evaluation labels tune the gate.",
        "",
        "## Limits and next step",
        "",
        "The original pilot state's first two draw seeds overlap the previously inspected smoke-v2 study; those trajectories were rerun, not independent new evidence. Its third draw and the other eleven states extend the pilot. All image pools and query panels were already used in earlier research.",
        "",
        "This changes only the first replay selection. It does not test sustained coverage or a controller that observes again after each update. Keep uniform replay as the baseline. Before adding health-model complexity, test whether sustained coverage over multiple updates creates a larger, repeatable benefit. Freeze that experiment's target, threshold and evaluation draws in advance; do not lower this gate's threshold using these results.",
        "",
        "With twelve related states and three reused pools, these are exploratory local results, not independent task confirmation or a repeated-decision MPC test. Prediction rows repeat the same parent prediction across two draws; those rows are not independent states. Any numerical failure blocks interpretation of tiny gains.",
    ]
    (run / "delayed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    names = (
        "coverage_delayed_pool_draw.csv",
        "coverage_delayed_states.csv",
        "delayed_health_policy.csv",
        "delayed_health_predictions.csv",
        "delayed_coverage_health.png",
        "delayed_result.json",
        "delayed_report.md",
    )
    atomic_json(
        run / "delayed_analysis_verification.json",
        {
            "source_sha256": digest(Path(__file__)),
            "report_verification_sha256": digest(run / "report_verification.json"),
            "frozen_health_gates_sha256": digest(run / "frozen_health_gates.json"),
            "artifact_hashes": {name: digest(run / name) for name in names},
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
