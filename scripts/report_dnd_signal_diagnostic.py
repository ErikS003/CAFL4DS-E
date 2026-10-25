"""Present the frozen cached diagnostic with saved scientific figures and notebook."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.run_dnd_signal_diagnostic import CONFIG, ROOT, check, digest, read, write

LABELS = {
    "constant": "Constant",
    "age": "Age only",
    "rank": "Age + rank",
    "variance": "Age + variance",
    "cka": "Age + CKA",
    "all": "Age + all",
}
COLORS = {
    "constant": "#888888",
    "age": "#2878a3",
    "rank": "#b18b2e",
    "variance": "#8b5b9e",
    "cka": "#30917a",
    "all": "#b64c35",
    "combined": "#b64c35",
}


def percent(value: float | None) -> str:
    """Format undefined warning precision explicitly."""
    return "undefined" if value is None else f"{100 * value:.1f}%"


def number(value: float | None) -> str:
    """Format an available point estimate."""
    return "undefined" if value is None else f"{value:+.3f}"


def save_figure(fig: plt.Figure, figures: Path, name: str) -> None:
    """Keep portable raster and vector versions."""
    fig.savefig(figures / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:  # noqa: C901 - sequential plot and notebook delivery checks
    """Render only after independent numerical verification succeeds."""
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    if not read(out / "independent-verification.json")["passed"]:
        raise ValueError("Independent verification is required")
    check(read(out / "input_lock.json")["hashes"])
    check(read(out / "complete.json")["hashes"])
    notebook = ROOT / config.notebook
    if notebook.exists() or (out / "report.md").exists():
        raise FileExistsError("Preserve the completed presentation")
    summary = read(out / "summary.json")
    primary = summary["horizons"][str(config.primary_horizon)]
    models = list(LABELS)
    instruments = ("rank", "variance", "cka", "combined")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True, layout="constrained")
    for ax, horizon in zip(axes, config.horizons, strict=True):
        for j, model in enumerate(models):
            stat = summary["horizons"][str(horizon)]["models"][model]["rmse_pp"]
            mean, bounds = stat["mean"][0], np.array(stat["ci95"])[0]
            ax.errorbar(
                mean,
                j,
                xerr=np.array([[mean - bounds[0]], [bounds[1] - mean]]),
                fmt="o",
                color=COLORS[model],
                capsize=3,
            )
        ax.set_title(f"{horizon} updates" + (" (primary)" if horizon == 32 else ""))
        ax.set_xlabel("Mean held-history loss RMSE (pp)")
        ax.set_yticks(range(len(models)), [LABELS[name] for name in models])
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    fig.suptitle("Does health improve on training age? Bars are descriptive 95% history intervals")
    save_figure(fig, figures, "forecast-errors")

    improvement = np.array(
        [primary["models"][name]["risk_rmse_improvement_vs_age"]["history_values"] for name in models[2:]]
    ).T
    limit = max(float(np.abs(improvement).max()), 0.01)
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    shown = ax.imshow(improvement, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f"{improvement[i, j]:+.2f}", ha="center", va="center")
    ax.set_xticks(range(4), [LABELS[name] for name in models[2:]])
    ax.set_yticks(range(8), summary["history_names"])
    ax.set_title(
        "32-update primary horizon: RMSE improvement over age only\nPositive means health predicts observed loss better"
    )
    fig.colorbar(shown, ax=ax, label="Age RMSE minus health RMSE (pp)")
    save_figure(fig, figures, "history-improvements")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    keys = ("actual", "matched_age", "matched_uniform_expected")
    names = ("Health alarm", "Same-count age", "Same-count uniform expectation")
    for j, (key, label) in enumerate(zip(keys, names, strict=True)):
        for ax, metric in zip(axes, ("harm_precision", "material_learning_fraction"), strict=True):
            values = [primary["alarms"][name][key][metric] for name in instruments]
            ax.bar(
                np.arange(4) + (j - 1) * 0.25,
                [100 * x if x is not None else np.nan for x in values],
                width=0.24,
                label=label,
            )
    labels = [
        f"{name.title()}\n{int(primary['alarms'][name]['actual']['warning_count'])} warnings" for name in instruments
    ]
    for ax in axes:
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percent of warned opportunities")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("Continuing loses at least 1 pp")
    axes[1].set_title("Holding sacrifices material endpoint learning")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    fig.suptitle("Fixed 10% alarms: warning quality and learning cost at 32 updates")
    save_figure(fig, figures, "warning-costs")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, horizon in zip(axes, config.horizons, strict=True):
        rows = [row for row in summary["actual_first_alarm_outcomes"] if row["horizon"] == horizon]
        censored = []
        for row in rows:
            if not row["full_horizon_available"]:
                censored.append(row["history"].split("_")[-1])
                continue
            loss, gain, _, _ = row["targets_pp"]
            i = int(row["history"].split("_")[-1])
            ax.scatter(loss, gain, s=45, color=plt.cm.tab10(i))
            ax.annotate(f"{i:02d}", (loss, gain), xytext=(5, 5 if i % 2 else -12), textcoords="offset points")
        ax.axvline(1, color="#888888", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
        ax.axhline(0, color="#bbbbbb", linewidth=0.7)
        ax.set_title(f"{horizon} updates" + (" (primary)" if horizon == 32 else ""))
        ax.set_xlabel("Loss holding would avoid (pp)")
        ax.text(0.02, 0.02, "Censored: " + (", ".join(censored) or "none"), transform=ax.transAxes, fontsize=8)
        ax.margins(x=0.12, y=0.2)
    axes[0].set_ylabel("Final balanced gain holding would forgo (pp)")
    fig.suptitle("Actual first alarms: prevented loss versus missed learning; labels are history IDs")
    save_figure(fig, figures, "first-alarm-outcomes")

    candidate_text = ", ".join(summary["primary_candidates"]) or "none"
    lines = [
        "# Do the D/N/D health signals warn of harmful continuation?",
        "",
        "**Exploratory cached analysis on the eight previously inspected histories. No new SSL "
        "updates, classifier fits or labels. No pause/resume controller is tested or qualified.**",
        "",
        "At each full-training checkpoint, compare holding its current capability for the "
        "declared horizon with continuing on the same incoming frames. The loss target is the "
        "largest observed day/night loss below capability already present at that checkpoint. A "
        "positive final gain is learning holding would forgo. This differs from drawdown "
        "measured from a future running peak.",
        "",
        "## Held-history forecasts",
        "",
        "The primary horizon is 32 updates. The age baseline averages the other seven histories "
        "at the same clock; each health model adds only current geometry and its previous "
        "eight-update change. Every test history is excluded from fitting, centering and "
        "scaling. Regularization is fixed and there is no horizon or threshold search.",
        "",
        "| Model | Loss RMSE, pp | Improvement over age, pp [95% history interval] | Histories "
        "improved | Balanced / day / night gain RMSE, pp | Exploratory screen |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for name in models:
        model = primary["models"][name]
        rmse = model["rmse_pp"]["mean"]
        change = model["risk_rmse_improvement_vs_age"]
        lines.append(
            f"| {LABELS[name]} | {rmse[0]:.3f} | {change['mean']:+.3f} "
            f"[{change['ci95'][0]:+.3f}, {change['ci95'][1]:+.3f}] | {model['histories_improved']} / 8 | "
            f"{rmse[1]:.3f} / {rmse[2]:.3f} / {rmse[3]:.3f} | {model['exploratory_screen']} |"
        )
    lines += [
        "",
        f"Primary exploratory candidates: **{candidate_text}**. The screen requires at least 10% lower "
        "mean loss RMSE than age, a positive paired-history improvement interval and improvement in "
        "at least six histories. All four health models are disclosed. Passing is a reason for "
        "further confirmation, not controller qualification.",
        "",
        "![Forecast errors](figures/forecast-errors.png)",
        "",
        "![History improvements](figures/history-improvements.png)",
        "",
        "## Fixed-alarm warning quality",
        "",
        "These are repeated hypothetical warnings on the full-training trajectory, including "
        "checkpoints after the original permanent stop. They are not decisions reached by a "
        "paused or stopped model. A useful hold here means at least 1 pp observed loss avoided "
        "while final balanced learning forgone is at most 0.5 pp and neither domain exceeds 1 pp.",
        "",
        "| Instrument | Warnings | Loss precision | Loss recall | Useful-hold precision | "
        "Material learning forgone | Positive balanced gain forgone, pp |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for name in instruments:
        stat = primary["alarms"][name]["actual"]
        lines.append(
            f"| {name} | {int(stat['warning_count'])} | {percent(stat['harm_precision'])} | "
            f"{percent(stat['harm_recall'])} | {percent(stat['useful_hold_precision'])} | "
            f"{percent(stat['material_learning_fraction'])} | "
            f"{number(stat['mean_positive_balanced_gain_when_warned_pp'])} |"
        )
    lines += [
        "",
        "![Warning costs](figures/warning-costs.png)",
        "",
        "Age controls select the exact same warning count within each history, using only "
        "held-out age forecasts; ties choose earlier dates. Uniform values are exact "
        "expectations for the same count. Matched comparisons average loss or forgone learning "
        "over eligible decisions, not over an accumulated policy. Overlapping windows cannot be "
        "summed into realized protection or training savings.",
        "",
        "| Alarm | Protected-loss advantage over same-count age, pp per decision [95% history "
        "interval] | Extra positive balanced learning forgone, pp per decision |",
        "| --- | --- | ---: |",
    ]
    for name in instruments:
        row = primary["alarms"][name]
        change = row["protected_loss_advantage_vs_age"]
        lines.append(
            f"| {name} | {change['mean']:+.3f} [{change['ci95'][0]:+.3f}, {change['ci95'][1]:+.3f}] | "
            f"{row['positive_learning_cost_vs_age']['mean']:+.3f} |"
        )
    lines += [
        "",
        "## Actual first alarms",
        "",
        "The table uses the primary 32-update horizon. All gains are continue-minus-hold; "
        "positive values are the learning stopping would sacrifice. A missing full horizon is "
        "explicitly censored, not treated as a safe continuation.",
        "",
        "| History | Stop | Phase | Worst loss, pp | Balanced / day / night final gain, pp |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in summary["actual_first_alarm_outcomes"]:
        if row["horizon"] != 32:
            continue
        values = row["targets_pp"]
        lines.append(
            f"| {row['history']} | {row['stop']} | {row['phase']} | "
            + (
                f"{values[0]:.3f} | {values[1]:+.3f} / {values[2]:+.3f} / {values[3]:+.3f} |"
                if values is not None
                else "censored | censored |"
            )
        )
    lines += [
        "",
        "![First-alarm outcomes](figures/first-alarm-outcomes.png)",
        "",
        "## Horizons, reader sensitivity and limits",
        "",
        "| Horizon | Eligible decisions per history | Observed-loss prevalence | Support-draw "
        "disagreement on loss >=1 pp |",
        "| --- | ---: | --- | --- |",
    ]
    for horizon in config.horizons:
        row = summary["horizons"][str(horizon)]
        lines.append(
            f"| {horizon} | {row['decisions_per_history']} | {percent(row['harm_prevalence'])} | "
            f"{percent(row['support_draw_harm_disagreement_fraction'])} |"
        )
    lines += [
        "",
        "The targets average three fixed few-label readouts before taking a maximum. Their noise "
        "and the number of observed checkpoints affect apparent loss; compare models within a "
        "horizon. Histories, not overlapping dates or shared support draws, supply the eight "
        "uncertainty units. One SSL seed, reused queries and already inspected sources limit "
        "these intervals. Domain and phase labels enter only offline outcomes and annotation, "
        "never forecast inputs.",
        "",
        "A held-history forecast can expose incremental signal value without testing a changed "
        "training trajectory. Pausing and resuming changes subsequent learner states and needs "
        "new physical branches; this cached analysis does not validate that actuator. The "
        "future-capability protection claim remains unqualified.",
        "",
        "## Integrity and artifacts",
        "",
        "All 96 multioutput ridge fits, raw readouts, bank geometry, interval targets, warning "
        "budgets and flags pass independent reconstruction. No learner runtime is used. All "
        "earlier scientific inputs remain unchanged.",
        "",
        "[Summary and history intervals](summary.json), [derived causal features and offline "
        "outcomes](derived_windows.npz), [held-history forecasts and matched "
        "controls](predictions.npz), [training-fold coefficients](folds.json), [frozen "
        "inputs](input_lock.json), [independent verification](independent-verification.json), "
        "[analysis protocol](../../../docs/project-plan/dnd-signal-diagnostic-frozen.md).",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        if not (out / target).resolve().is_file():
            raise ValueError(f"Broken report link: {target}")
    figure_names = ("forecast-errors", "history-improvements", "warning-costs", "first-alarm-outcomes")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# D/N/D health signals: harmful continuation or useful learning?\n",
                "Saved figures from a frozen, independently checked cached analysis. Display cells "
                "perform no training.\n",
            ],
        }
    ]
    for count, name in enumerate(figure_names, 1):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": count,
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    "root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'cafl4ds').is_dir() "
                    "and (p / 'outputs').is_dir())\n",
                    f"display(Image(filename=str(root / '{config.output}/figures/{name}.png')))",
                ],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": base64.b64encode((figures / f"{name}.png").read_bytes()).decode(),
                            "text/plain": ["<saved independently checked scientific figure>"],
                        },
                    }
                ],
            }
        )
    converted = [
        re.sub(
            r"\]\(([^)]+)\)",
            lambda match: "](../../" + (out / match[1]).resolve().relative_to(ROOT).as_posix() + ")",
            line,
        )
        for line in lines
        if not line.startswith("![")
    ]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in converted]})
    notebook.write_text(
        json.dumps(
            {
                "cells": cells,
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
        else:
            for target in re.findall(r"\]\(([^)]+)\)", "".join(cell["source"])):
                if not (notebook.parent / target).resolve().is_file():
                    raise ValueError(f"Broken notebook link: {target}")
    check(read(out / "input_lock.json")["hashes"])
    check(read(out / "complete.json")["hashes"])
    write(
        out / "presentation-verification.json",
        {
            "passed": True,
            "notebook_code_compiled": True,
            "links_resolved": True,
            "saved_figures": 4,
            "hashes": {str(path): digest(path) for path in [report, notebook, *figures.glob("*")]},
        },
    )
    print({"report": str(report), "notebook": str(notebook)}, flush=True)


if __name__ == "__main__":
    main()
