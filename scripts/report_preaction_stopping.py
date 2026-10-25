"""Present audited held-history forecasts without changing scientific artifacts."""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.run_preaction_stopping import CONFIG, ROOT, check, digest, read, write


def main():
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert read(out / "independent-verification.json")["passed"]
    check(read(out / "complete.json")["hashes"])
    summary, fitted = read(out / "summary.json"), read(out / "forecasts.json")
    with np.load(out / "offline_targets.npz") as a:
        targets = a["gain"].reshape(24, 2)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), layout="constrained")
    for j, name in enumerate(("horizon_mean", "controls", "health")):
        for domain in (0, 1):
            axis = axes[domain, j]
            pred = np.array(fitted["predictions"][name])[:, domain]
            actual = targets[:, domain]
            scatter = axis.scatter(
                actual, pred, c=np.tile([8, 16, 24, 32], 6), cmap="viridis", s=30, vmin=8, vmax=32, alpha=0.85
            )
            extent = [min(actual.min(), pred.min()) - 0.3, max(actual.max(), pred.max()) + 0.3]
            axis.plot(extent, extent, color="0.5", linewidth=0.8)
            axis.axvline(-1, color="0.65", linestyle="--", linewidth=0.7)
            axis.axhline(-1, color="0.65", linestyle="--", linewidth=0.7)
            axis.set_xlim(extent)
            axis.set_ylim(extent)
            rmse = summary["metrics"][name]["rmse_pp"][domain]
            axis.set_title(f"{name.replace('_', ' ')} · RMSE {rmse:.2f} pp", fontsize=11)
            axis.set_xlabel("Measured gain (pp)")
            if j == 0:
                axis.set_ylabel(f"{('Day', 'Night')[domain]} forecast gain (pp)")
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Parent-only forecasts: each history excluded from its fitted model", fontsize=13)
    fig.colorbar(scatter, ax=axes, ticks=[8, 16, 24, 32], label="Proposed training updates", shrink=0.7)
    fig.savefig(figures / "held-history-forecasts.png", dpi=170)
    fig.savefig(figures / "held-history-forecasts.svg")
    plt.close(fig)

    names = [
        "always_train",
        "acquired_oracle",
        "horizon_mean_point",
        "controls_point",
        "health_point",
        "health_envelope",
    ]
    labels = ["Always train", "Measured oracle", "Horizon mean", "Recent loss", "Parent health", "Health + envelope"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained", width_ratios=[1.1, 1])
    values = np.array([summary["rules"][n]["updates_by_history"] for n in names])
    axes[0].imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=32)
    axes[0].set_xticks(range(6), [f"{i:02d}" for i in range(6)])
    axes[0].set_yticks(range(len(names)), labels)
    axes[0].set_xlabel("History fresh_00–fresh_05")
    axes[0].set_title("Selected training updates")
    for i in range(len(names)):
        for j in range(6):
            axes[0].text(
                j, i, str(values[i, j]), ha="center", va="center", color="white" if values[i, j] >= 24 else "black"
            )
    positions = np.arange(len(names))
    for domain, label, color, offset in ((1, "Day", "#2874a6", -0.18), (2, "Night", "#b96929", 0.18)):
        gains = [summary["rules"][n]["mean_gain_pp"][domain] for n in names]
        axes[1].barh(positions + offset, gains, height=0.32, label=label, color=color)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="0.5", linewidth=0.7)
    axes[1].set_xlabel("Mean measured gain across six histories (pp)")
    axes[1].set_title("Actual outcomes of forecast choices")
    axes[1].legend(loc="lower right")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory stopping diagnostics; no original-pretrained protection claim", fontsize=12)
    fig.savefig(figures / "forecast-stopping-outcomes.png", dpi=170)
    fig.savefig(figures / "forecast-stopping-outcomes.svg")
    plt.close(fig)

    lines = [
        "# Parent-only cumulative-effect prediction",
        "",
        "**Exploratory fitting on six previously inspected histories. Whole-history exclusion prevents fitting leakage; it does not supply fresh confirmation. No online controller is qualified.**",
        "",
        "Forecast inputs are fixed episode-start geometry, the eight strictly preceding SSL losses, and the proposed duration. No future geometry, future losses, task-labelled probe score or future image content enters the predictor. Day/night monitor metadata are assumed available. Offline task labels provide training targets and scoring.",
        "",
        "Twenty-four forecasts cover four training durations in each history. The outcomes are changes in class-balanced city/highway accuracy on day and night queries, relative to parent 128. Every schedule holds after its chosen prefix until the same endpoint. The original pretrained reference is outside this pilot's protection promise.",
        "",
        "## Prediction accuracy",
        "",
        "| Model | Day RMSE, pp | Night RMSE, pp | Day / night bias, pp |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in ("zero", "horizon_mean", "controls", "health"):
        m = summary["metrics"][name]
        lines.append(
            f"| {name} | {m['rmse_pp'][0]:.3f} | {m['rmse_pp'][1]:.3f} | {m['bias_pp'][0]:+.3f} / {m['bias_pp'][1]:+.3f} |"
        )
    lines += [
        "",
        "| Health compared with | Day / night MSE reduction | Histories improved, day / night |",
        "| --- | --- | --- |",
    ]
    for name, c in summary["comparisons"].items():
        lines.append(
            f"| {name} | {c['fractional_mse_reduction'][0]:+.1%} / {c['fractional_mse_reduction'][1]:+.1%} | {c['histories_improved'][0]} / {c['histories_improved'][1]} |"
        )
    lines += [
        "",
        f"Exploratory prediction gate: **{summary['exploratory_prediction_gate']}**. It requires improvement in both domains against every declared baseline, source consistency, positive adjusted source ranges and absolute error/bias bounds. Best geometry shams are selected separately per domain as a stringent specificity diagnostic. No model tuning follows this result.",
        "",
        "![Held-history forecasts](figures/held-history-forecasts.png)",
        "",
        "## Forecast-only stopping",
        "",
        "Point rules require predicted cumulative day/night gains >=-1pp at every visited boundary and final balanced gain >=0.25pp. They maximize tanh(balanced_gain/0.5pp), breaking ties toward fewer updates. Envelope rules use forecast minus the largest inner-held-history residual per horizon/domain. These training-only sensitivity envelopes have no nominal or simultaneous coverage guarantee.",
        "",
        "| Rule | Updates by history 00–05 | Total updates | Actual balanced / day / night gain, pp | Within / loss / unknown | Same-work random balanced gain |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for name, r in summary["rules"].items():
        gains = " / ".join(f"{v:+.3f}" for v in r["mean_gain_pp"])
        counts = f"{r['within_histories']} / {r['violation_histories']} / {r['unknown_histories']}"
        lines.append(
            f"| {name} | {' / '.join(map(str, r['updates_by_history']))} | {r['total_updates']} | {gains} | {counts} | {r['matched_random']['mean_gain_pp'][0]:+.3f} |"
        )
    lines += [
        "",
        "Budget status uses the measured conditional acquired-reference ranges across the whole selected path. Unknown is not certified protection. Exact random comparisons enumerate all 720 assignments of the same six durations, preserving training work; their choices are not screened for safety.",
        "",
        "![Stopping outcomes](figures/forecast-stopping-outcomes.png)",
        "",
        "## Limits and verification",
        "",
        "The inherited longer-horizon combined precision gate remains failed. Fixed-pool measurement ranges are not population safety intervals. Four correlated horizons per source do not create new histories; there are only six source units overall. Repeated reuse also prevents a fresh-confirmation claim. Geometry is predictive input here, not an independently calibrated protection constraint.",
        "",
        "Independent verification reconstructs parent geometry and loss inputs, all regressions using augmented least squares, nested residual envelopes, horizon exclusions, specificity shams, stopping choices and work-matched random scores. No new SSL updates or canonical classifier fits are executed. Focused tests check held-out-label poisoning, future-loss exclusion, sham donors, normalization and cumulative budgets.",
        "",
        "Artifacts: [summary](summary.json), [all forecasts and folds](forecasts.json), [causal inputs](causal_inputs.npz), [offline targets](offline_targets.npz), [frozen inputs](input_lock.json), [independent verification](independent-verification.json), [protocol](../../../docs/project-plan/preaction-stopping-frozen.md).",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        assert (out / target).resolve().is_file(), target
    check(read(out / "complete.json")["hashes"])
    write(
        out / "presentation-verification.json",
        {
            "passed": True,
            "primary_hashes_unchanged": True,
            "report_links_resolved": True,
            "hashes": {str(p): digest(p) for p in [report, *figures.glob("*")]},
            "source_sha256": digest(__file__),
        },
    )
    print({"report": str(report)}, flush=True)


if __name__ == "__main__":
    main()
