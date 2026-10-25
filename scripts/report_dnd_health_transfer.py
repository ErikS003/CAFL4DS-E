"""Present new-history timing transfer, all source paths and exact work controls."""

import base64
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from omegaconf import OmegaConf

from scripts.run_dnd_health_transfer import CONFIG
from scripts.run_preaction_stopping import ROOT, check, digest, read, write


def main():
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert read(out / "independent-verification.json")["passed"]
    check(read(out / "complete.json")["hashes"])
    result = read(out / "summary.json")
    collection = read(out / "collection.json")
    rules = result["rules"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    x = np.array(result["points"])
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey=True, layout="constrained")
    for i, (record, ax) in enumerate(zip(collection["records"], axes.ravel(), strict=True)):
        day, night = record["day_updates"], record["night_updates"]
        stop = record["stop"]
        ax.axvspan(0, day, color="#ead0ae", alpha=0.25)
        ax.axvspan(day, day + night, color="#c7d8e9", alpha=0.35)
        ax.axvspan(day + night, 448, color="#ead0ae", alpha=0.25)
        for d, label, color in ((1, "Day", "#2874a6"), (2, "Night", "#b96929")):
            ax.plot(
                x,
                np.array(rules["full"]["history_trajectories_pp"])[i, :, d],
                color=color,
                label=f"{label}: full",
                linewidth=1.2,
            )
            ax.plot(
                x,
                np.array(rules["health"]["history_trajectories_pp"])[i, :, d],
                color=color,
                label=f"{label}: stopped",
                linestyle="--",
                linewidth=1.5,
            )
        ax.axvline(stop, color="0.3", linestyle=":", linewidth=1)
        ax.set_title(
            f"{record['name']} · D/N/D={day}/{night}/{448 - day - night} · stop {stop}"
            + ("" if record["alarm_fired"] else " (end)"),
            fontsize=10,
        )
        ax.set_xlim(0, 448)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.1)
    for ax in axes[-1]:
        ax.set_xlabel("SSL update clock")
    for ax in axes[:, 0]:
        ax.set_ylabel("Few-label capability (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle("Frozen health stopping on eight new histories with varied D/N/D transition times", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"source-trajectories.{ext}", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = {"frozen": "#999999", "full": "#2874a6", "fixed148": "#ad8d35", "fixed264": "#548d65", "health": "#b96929"}
    labels = {
        "frozen": "Warm frozen",
        "full": "Continuous",
        "fixed148": "Fixed 148",
        "fixed264": "Fixed 264",
        "health": "Health",
    }
    offsets = {"frozen": (5, -12), "full": (5, 5), "fixed148": (5, 9), "fixed264": (-45, 9), "health": (5, -12)}
    for name, rule in rules.items():
        m = rule["metrics"]
        risk = m["worst_domain_drawdown"]["mean"]
        gain = rule["terminal_minus_warm"]["mean"][0]
        axes[0].scatter(risk, gain, color=colors[name], s=70, label=labels[name])
        axes[0].annotate(labels[name], (risk, gain), xytext=offsets[name], textcoords="offset points", fontsize=9)
        axes[1].scatter(rule["total_updates"] / 8, gain, color=colors[name], s=70)
        axes[1].annotate(
            labels[name],
            (rule["total_updates"] / 8, gain),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xlabel("Mean largest observed worst-domain drawdown (pp)")
    axes[0].set_ylabel("Final balanced gain over warm (pp)")
    axes[0].set_title("Less observed loss versus retained learning")
    axes[1].set_xlabel("Mean training updates per history")
    axes[1].set_ylabel("Final balanced gain over warm (pp)")
    axes[1].set_title("Training work versus retained learning")
    for ax in axes:
        ax.axhline(0, color=".6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.12)
        ax.margins(0.15)
    fig.suptitle("All declared controls; labels are used only for offline scoring", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"risk-learning-work.{ext}", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, r in enumerate(collection["records"]):
        d, n = r["day_updates"], r["night_updates"]
        ax.barh(i, d, left=0, color="#ead0ae", height=0.65)
        ax.barh(i, n, left=d, color="#c7d8e9", height=0.65)
        ax.barh(i, 448 - d - n, left=d + n, color="#ead0ae", height=0.65)
        ax.scatter(r["stop"], i, color="#b2462d", marker="X", s=65, zorder=4)
        ax.text(r["stop"] + 4, i, str(r["stop"]), va="center", fontsize=9)
    ax.axvline(132, color="0.3", linestyle="--", linewidth=1, label="Monitoring starts:132")
    ax.axvline(148, color="0.55", linestyle=":", linewidth=1, label="Fixed148")
    ax.axvline(264, color="0.55", linestyle="-.", linewidth=1, label="Fixed264")
    ax.set_yticks(range(8), [r["name"] for r in collection["records"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 465)
    ax.set_xlabel("SSL update clock; red X is the health stop or stream end")
    ax.set_title("Transition clocks vary; the guard receives none of these boundary labels")
    handles, legend = ax.get_legend_handles_labels()
    handles += [Patch(color="#ead0ae"), Patch(color="#c7d8e9")]
    legend += ["Day", "Night"]
    fig.legend(handles, legend, loc="outside lower center", ncol=5, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"stop-timing.{ext}", dpi=170)
    plt.close(fig)

    with np.load(out / "shuffle_assignments.npz") as a:
        means = a["assignment_mean_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for ax, j, label in (
        (axes[0], 12, "Mean worst-domain drawdown (pp)"),
        (axes[1], 0, "Mean final balanced capability (%)"),
    ):
        values = means[:, j]
        actual = rules["health"]["metrics"][result["metric_order"][j]]["mean"]
        if np.ptp(values) < 1e-10:
            ax.bar(values.mean(), len(values), width=0.05, color="#97b7ce")
            ax.text(
                0.5, 0.8, "All allocations have identical outcomes", transform=ax.transAxes, ha="center", fontsize=9
            )
        else:
            ax.hist(values, bins=40, color="#97b7ce")
        ax.axvline(actual, color="#b2462d", linewidth=1.7, label="Actual health allocation")
        ax.set_xlabel(label)
        ax.set_ylabel("Exact timing assignments")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("All40,320 allocations preserve the same total training work", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"matched-work-shuffle.{ext}", dpi=170)
    plt.close(fig)

    lines = [
        "# Frozen health stopping: transfer across new D/N/D histories",
        "",
        "**Eight new clip-disjoint training histories, varied transition clocks, one SSL seed and reused offline evaluation. Thresholds and monitoring start were fixed before new learner outcomes. No controller is qualified.**",
        "",
        "The guard starts at update132 and receives global rank, normalized variance and CKA on a fixed monitor bank. Two consecutive10% alarms stop training permanently. It is never told a domain label or transition time. Extra evaluation checkpoints do not increment alarm persistence. Task labels fit copied evaluation learners and score outcomes offline.",
        "",
        "![Source trajectories](figures/source-trajectories.png)",
        "",
        "## Risk, capability and work",
        "",
        "| Rule | Total updates | Final balanced gain over warm, pp | Day / night maximum drawdown, pp | Worst-domain drawdown, pp | Seconds >1pp below earlier observed best, day / night |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for name, r in rules.items():
        m = r["metrics"]
        lines.append(
            f"| {name} | {r['total_updates']} | {r['terminal_minus_warm']['mean'][0]:+.3f} | {m['drawdown_day']['mean']:.3f} / {m['drawdown_night']['mean']:.3f} | {m['worst_domain_drawdown']['mean']:.3f} | {m['seconds_below_day']['mean']:.1f} / {m['seconds_below_night']['mean']:.1f} |"
        )
    lines += [
        "",
        "Drawdown is each history/domain's largest observed drop from its own running observed maximum. The worst-domain value takes the larger of the two. These evaluation maxima never update the guard reference. Duration uses left-observation values after update132, weighted by interval length and the nominal16Hz construction; it is not an observed continuous crossing time.",
        "",
        "![Risk and work](figures/risk-learning-work.png)",
        "",
        "## Does health add timing information?",
        "",
    ]
    comparison = result["health_vs_shuffle"]
    random = result["matched_work_shuffle"]
    lines += [
        f"Added-timing-information flag: **{result['added_timing_information_flag']}**. Full-training final-quality tolerances: **{result['full_quality_tolerances_met']}**. Distinct stop counts: **{result['distinct_stop_counts']}**. Fixed controls dominating health on risk/final balanced capability/update count: **{', '.join(result['fixed_controls_dominating_health']) or 'none'}**.",
        "",
        f"Against exact work-matched shuffled timing, mean worst-domain drawdown reduction is {comparison['risk_reduction']['mean']:+.3f}pp, conditional source range {comparison['risk_reduction']['ci95']}; fractional reduction {comparison['fractional_risk_reduction']:+.1%}. Final balanced/day/night differences are {' / '.join(f'{v:+.3f}' for v in comparison['terminal_difference_pp']['mean'])}pp. Fraction of exact assignments with no larger mean drawdown than health: {random['fraction_assignments_with_no_larger_drawdown']:.3f}.",
        "",
        "The primary flag requires at least20% and0.5pp risk reduction with a positive paired-history range, balanced final cost <=0.5pp and neither domain cost >1pp against shuffled timing. It does not certify a deployed policy. The shuffled allocation uses measured stop times, preserves exact total work, and deliberately removes their pairing with histories.",
        "",
        "![Exact shuffled timing](figures/matched-work-shuffle.png)",
        "",
        "## Actual stopping clocks",
        "",
        "| History | D1 / N / D2 updates | Stop | Alarm fired | Stage at stopping | Final day / night minus full, pp |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for i, r in enumerate(collection["records"]):
        differences = [
            rules["health"]["metrics"][f"final_{d}"]["history_values"][i]
            - rules["full"]["metrics"][f"final_{d}"]["history_values"][i]
            for d in ("day", "night")
        ]
        lines.append(
            f"| {r['name']} | {r['day_updates']} / {r['night_updates']} / {448 - r['day_updates'] - r['night_updates']} | {r['stop']} | {r['alarm_fired']} | {result['stop_phase'][i] if r['alarm_fired'] else 'stream ended'} | {differences[0]:+.3f} / {differences[1]:+.3f} |"
        )
    lines += [
        "",
        "![Stop clocks](figures/stop-timing.png)",
        "",
        "## Measurement and integrity",
        "",
        "There are224 new clips and eight source histories. Capture sessions are unknown. All online optimization uses seed5003. The expanded reader uses three fixed fifty-label learners,100 fine-tuning steps and3,663 clear-weather queries. Query/support pools are reused and shared across histories; source ranges are conditional on this measurement design, not independent-session or simultaneous-path safety intervals. Clips are timestamped internally, but the D/N/D visits concatenate different drives.",
        "",
        "The fixed scientific budget executes3,584 SSL updates and1,155 canonical classifier fits, including one shared warm readout. Exact replay adds3,584 SSL updates and two classifier fits. Guard forecasts never use labelled readouts. All frozen inputs, source allocation, dense embeddings, complete state fingerprints, raw scores, timing allocations and statistical flags pass independent verification. Physical skipped arrivals preserve the stopped model, Adam, modes and RNG.",
        "",
        "Scientific work is counterfactual full-prefix collection so all controls can be evaluated. Training updates saved by the simulated guard do not directly establish wall-clock savings: monitor/readout/forensic costs are recorded separately. No threshold is tuned from these results, and no pause/resume or predictor experiment is launched.",
        "",
        "Artifacts: [summary and source intervals](summary.json), [causal events and work](collection.json), [exact shuffle assignments](shuffle_assignments.npz), [frozen inputs](input_lock.json), [training replay](training-replay-verification.json), [independent verification](independent-verification.json), [protocol](../../../docs/project-plan/dnd-health-transfer-frozen.md).",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        assert (out / target).resolve().is_file(), target
    notebook = ROOT / "notebooks/sums/17_dnd_health_transfer.ipynb"
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Frozen health stopping on new D/N/D histories\n",
                "The guard does not receive transition clocks or task labels. All figures and outcomes below are saved from the frozen, audited experiment. Rerunning display cells performs no training.\n",
            ],
        }
    ]
    for count, name in enumerate(
        ("source-trajectories", "risk-learning-work", "stop-timing", "matched-work-shuffle"), 1
    ):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": count,
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    "root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'cafl4ds').is_dir() and (p / 'outputs').is_dir())\n",
                    f"display(Image(filename=str(root / '{config.output}/figures/{name}.png')))",
                ],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": base64.b64encode((figures / f"{name}.png").read_bytes()).decode(),
                            "text/plain": ["<saved audited figure>"],
                        },
                        "metadata": {},
                    }
                ],
            }
        )

    def notebook_links(line):
        return re.sub(
            r"\]\(([^)]+)\)", lambda m: "](../../" + (out / m[1]).resolve().relative_to(ROOT).as_posix() + ")", line
        )

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [(notebook_links(line) + "\n") for line in lines if not line.startswith("![")],
        }
    )
    notebook.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
                "nbformat": 4,
                "nbformat_minor": 4,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    check(read(out / "complete.json")["hashes"])
    write(
        out / "presentation-verification.json",
        {
            "passed": True,
            "primary_hashes_unchanged": True,
            "links_resolved": True,
            "hashes": {str(p): digest(p) for p in [report, notebook, *figures.glob("*")]},
            "source_sha256": digest(__file__),
        },
    )
    print({"report": str(report), "notebook": str(notebook)}, flush=True)


if __name__ == "__main__":
    main()
