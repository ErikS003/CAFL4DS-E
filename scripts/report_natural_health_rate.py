"""Present the archive-order stream, health response and conditional tradeoffs."""

from __future__ import annotations

import base64
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.run_natural_health_rate import CONFIG
from scripts.run_preaction_stopping import ROOT, check, digest, read, write

LABELS = {
    "frozen": "Warm frozen",
    "full": "Full training",
    "permanent": "Permanent stop",
    "quarter": "Fixed quarter",
    "adaptive": "Quarter + recovery",
    "paced": "Same-work pacing",
}
COLORS = {
    "frozen": "#888888",
    "full": "#2878a3",
    "permanent": "#ad8d2c",
    "quarter": "#776099",
    "adaptive": "#b95332",
    "paced": "#458d69",
}
FIGURES = ("stream-and-capability", "admission-and-health", "risk-learning-work", "reader-domain-contrasts")


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def reader_difference(summary, control, draw_index):
    a = np.array(summary["rules"]["adaptive"]["reader_curves"])[draw_index, -1]
    full = np.array(summary["rules"]["full"]["reader_curves"])
    if control == "permanent":
        point = summary["points"].index(summary["first_alarm"]) if summary["first_alarm"] is not None else -1
        b = full[draw_index, point]
    else:
        b = np.array(summary["rules"][control]["reader_curves"])[draw_index, -1]
    return a - b


def main():
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert read(out / "independent-verification.json")["passed"]
    check(read(out / "input_lock.json")["hashes"])
    check(read(out / "complete.json")["hashes"])
    summary, collection = read(out / "summary.json"), read(out / "collection.json")
    intake = read(ROOT / config.frames / "intake.json")
    notebook = ROOT / config.notebook
    assert not notebook.exists() and not (out / "report.md").exists()
    figures = out / "figures"
    figures.mkdir()
    rules, records, points = summary["rules"], collection["records"], summary["points"]
    horizon, fork = intake["batches"], summary["first_alarm"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [0.55, 2, 2, 2]}, layout="constrained"
    )
    lights = {"daytime": "#ecd4aa", "night": "#a5bfd5", "dawn/dusk": "#c8b0cf"}
    offset, legend = 0, set()
    for clip in intake["clips"]:
        count = clip["shape"][0]
        attrs = clip["attributes"] if isinstance(clip["attributes"], dict) else {}
        light = attrs.get("timeofday", "unknown")
        axes[0].broken_barh(
            [(offset / 32, count / 32)],
            (0, 1),
            facecolors=lights.get(light, "#cccccc"),
            label=light if light not in legend else None,
        )
        legend.add(light)
        offset += count
    axes[0].set_yticks([])
    axes[0].set_ylabel("Keyframe\ntag")
    axes[0].set_title(
        "Offline keyframe tags: beige = day; blue = night; purple = dusk; gray = unknown", fontsize=8, loc="left"
    )
    for domain, ax in enumerate(axes[1:]):
        for arm in ("full", "permanent", "quarter", "paced", "adaptive"):
            ax.plot(
                points,
                np.array(rules[arm]["curve"])[:, domain],
                color=COLORS[arm],
                label=LABELS[arm],
                linewidth=2 if arm == "adaptive" else 1.2,
                linestyle="--" if arm == "permanent" else "-",
            )
        if fork is not None:
            ax.axvline(fork, color="#666666", linestyle=":", linewidth=1)
        ax.set_ylabel(("Balanced", "Day", "Night")[domain] + " capability (%)")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Arriving batches of 32 frames; separate clips remain in pinned archive order")
    axes[-1].set_xlim(0, horizon)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncols=5, fontsize=8)
    fig.suptitle("Natural within-video order, unarranged regimes and one continuous learner state")
    save(fig, figures, FIGURES[0])

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    adaptive = records["adaptive"]
    for arm in ("adaptive", "quarter", "paced"):
        axes[0].step(
            range(1, horizon + 1),
            np.array(records[arm]["rates"]) * 100,
            where="post",
            color=COLORS[arm],
            label=LABELS[arm],
            linestyle="-" if arm == "adaptive" else ":" if arm == "quarter" else "--",
        )
    axes[0].set_yticks([25, 50, 75, 100])
    axes[0].set_ylabel("Target admission (%)")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    for event in adaptive["events"]:
        if event["alarm"]:
            axes[0].axvline(event["step"] + 1, color=COLORS["adaptive"], alpha=0.12, linewidth=0.6)
    for feature, ax in enumerate(axes[1:]):
        for arm in ("full", "adaptive"):
            events = records[arm]["events"]
            ax.plot(
                [e["step"] for e in events],
                [e["values"][feature] for e in events],
                color=COLORS[arm],
                linewidth=1.2,
                label=LABELS[arm],
            )
        ax.axhline(0.1, color="#666666", linestyle="--", linewidth=1)
        ax.set_ylabel(("Rank decline", "Variance decline", "CKA distance")[feature])
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Arriving batches; fixed reference at 132; health observations continue every 8 arrivals")
    axes[-1].set_xlim(0, horizon)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Label-free alarms reduce admission; another alarm resets the recovery schedule")
    save(fig, figures, FIGURES[1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, layout="constrained")
    arms = list(LABELS)
    for feature, ax in enumerate(axes):
        values = [
            rules[a]["metrics"][-1]
            if feature == 0
            else rules[a]["gain_over_warm"][0]
            if feature == 1
            else rules[a]["updates"]
            for a in arms
        ]
        ax.barh(range(len(arms)), values, color=[COLORS[a] for a in arms], height=0.6)
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
        ax.set_xlabel(
            ("Worst observed domain drawdown (pp)", "Final balanced gain over warm (pp)", "Accepted training batches")[
                feature
            ]
        )
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle("One stream and one SSL seed: observed protection, final learning and training work")
    save(fig, figures, FIGURES[2])

    controls = ("full", "permanent", "quarter", "paced")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, layout="constrained")
    for domain, ax in enumerate(axes):
        for i, draw in enumerate(config.draws):
            values = [reader_difference(summary, arm, i)[domain] for arm in controls]
            ax.scatter(values, np.arange(4) + (i - 1) * 0.14, label=str(draw), s=35)
        ax.set_yticks(range(4), [LABELS[a] for a in controls])
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title(("Balanced", "Day", "Night")[domain])
        ax.set_xlabel("Quarter + recovery minus control (pp)")
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, title="Support-and-fitting draw"
    )
    fig.suptitle("Reader sensitivity; shared fitting draws are not independent driving streams")
    save(fig, figures, FIGURES[3])

    suffix = adaptive["rates"][fork:] if fork is not None else []
    occupancy = [suffix.count(v) for v in (0.25, 0.5, 0.75, 1)]
    alarm_steps = [e["step"] for e in adaptive["events"] if e["alarm"]]
    spans = sum(c["frames"][-1]["pts_seconds"] - c["frames"][0]["pts_seconds"] for c in intake["clips"])
    lines = [
        "# Long BDD100K archive-order quarter admission and recovery",
        "",
        "**A descriptive physical experiment on one ordered stream and one SSL seed. "
        "No controller qualification or independent-source confidence interval is claimed.**",
        "",
        f"The stream contains {intake['frames_used']:,} used frames in {horizon:,} batches from "
        f"{len(intake['clips'])} train videos. Sampled within-clip spans sum to {spans / 3600:.2f} hours; "
        "inter-clip gaps and capture-session chronology are unknown. Clips preserve the pinned "
        "video archive's member order after warm/monitor/evaluation exclusions. There is no "
        "day/night grouping, weather balancing or scene-based sampling. Rare and unknown "
        "attributes remain eligible. Keyframe tags describe the stream offline and do not label every frame.",
        "",
        "The fixed reference is captured at arrival 132. Rank/normalized-variance declines or CKA "
        "distance of at least .10 on two consecutive eight-arrival observations trigger quarter "
        "admission on subsequent arrivals. Admission increases to half, three-quarter and full "
        "in 32-arrival stages; another alarm resets it. Whole admitted batches contain 32 images. "
        "No regime boundary, label or future outcome enters the controller.",
        "",
        "## Capability and observed forgetting",
        "",
        "| Policy | Accepted batches | Final balanced / day / night gain over warm, pp | "
        "Worst-domain observed drawdown, pp |",
        "| --- | ---: | --- | ---: |",
    ]
    for arm in arms:
        value, row = rules[arm]["gain_over_warm"], rules[arm]
        lines.append(
            f"| {LABELS[arm]} | {row['updates']} | {value[0]:+.3f} / {value[1]:+.3f} / {value[2]:+.3f} | "
            f"{row['metrics'][-1]:.3f} |"
        )
    lines += [
        "",
        "![Stream and capability](figures/stream-and-capability.png)",
        "",
        "Capability is sampled at warm, 132, every 128 arrivals, the first alarm and the final "
        "arrival. This is sparser than the earlier short D/N/D experiment; drawdowns are observed "
        "lower bounds and can miss short losses. The downstream task is the unchanged city/highway "
        "few-label reader on reused validation queries, even though the training stream includes "
        "other scenes and weather. Labels are used only for offline fitting and scoring.",
        "",
        "## Alarm and recovery behavior",
        "",
        f"First confirmed alarm: **{fork}**. Adaptive confirmed alarms: **{len(alarm_steps)}**. "
        f"Post-alarm arrival counts at 25/50/75/100% admission: **{occupancy}**. "
        f"Adaptive accepts {adaptive['updates']:,}/{horizon:,} batches "
        f"({100 * adaptive['updates'] / horizon:.1f}% of incoming image exposures).",
        "",
        "A persistent shift relative to the fixed reference can keep resetting quarter admission. "
        "Training continues; the reference does not chase the current learner. An absence of "
        "alarms is also a legitimate result and does not justify tuning the threshold after this run.",
        "",
        "![Admission and health](figures/admission-and-health.png)",
        "",
        "## Learning and equal-work controls",
        "",
        "| Control | Final adaptive minus control: balanced / day / night, pp | Drawdown reduction, pp |",
        "| --- | --- | ---: |",
    ]
    for arm in controls:
        row = summary["contrasts"][arm]
        value = row["final_difference_pp"]
        lines.append(
            f"| {LABELS[arm]} | {value[0]:+.3f} / {value[1]:+.3f} / {value[2]:+.3f} | "
            f"{row['drawdown_reduction_pp']:+.3f} |"
        )
    lines += [
        "",
        "Recovery versus permanent stopping measures regained learning and additional "
        "forgetting. Recovery versus quarter measures the effect of increasing admission. "
        "Recovery versus paced admission tests allocation of exactly the same accepted work "
        "after the shared first alarm. Pacing's eventual count is obtained offline from adaptive "
        "recovery; it is a conditional benchmark, not an online advance-budget policy. Full and "
        "all controls retain source order. No ranking is promoted from a single natural stream.",
        "",
        "![Risk learning work](figures/risk-learning-work.png)",
        "",
        "## Reader sensitivity",
        "",
        "| Reader draw | Control | Final adaptive-minus-control balanced / day / night, pp | Drawdown reduction, pp |",
        "| --- | --- | --- | ---: |",
    ]
    for i, draw in enumerate(config.draws):
        for arm in controls:
            value = reader_difference(summary, arm, i)
            reduction = rules[arm]["reader_metrics"][i][-1] - rules["adaptive"]["reader_metrics"][i][-1]
            lines.append(
                f"| {draw} | {LABELS[arm]} | {value[0]:+.3f} / {value[1]:+.3f} / {value[2]:+.3f} | {reduction:+.3f} |"
            )
    audit = read(out / "independent-verification.json")
    lines += [
        "",
        "![Reader contrasts](figures/reader-domain-contrasts.png)",
        "",
        "The three support/fitting draws share one query pool and one trained trajectory per "
        "policy. They diagnose reader dependence; they do not supply independent stream "
        "replicates. There are no source-confidence intervals and no automatic qualification flag.",
        "The primary drawdown is the maximum of the pooled reader-mean capability curve. Reader-specific "
        "drawdowns are separate maxima; averaging those estimates a different quantity. Both are "
        "reported without substituting a favorable reader for the primary result.",
        "",
        "## Integrity and actual work",
        "",
        f"Collection uses {audit['physical_ssl_updates']:,} physical SSL updates and "
        f"{audit['new_classifier_fits']:,} new canonical classifier fits across "
        f"{audit['physical_branches']} physical branches. Exact replay adds the same SSL work "
        "and two classifier fits. All four logical arms are retained, including any explicit "
        "identical-mask aliases. Full-state cache hits reuse readouts without refitting.",
        "",
        "Exact replay checks masks, alarm events, losses, gradients, monitor banks and complete "
        "model/Adam/mode/RNG fingerprints. Two classifier fits replay exactly. Independent "
        "reconstruction verifies the archive prefix, warm/held-out exclusions, every source "
        "frame hash and timestamp, raw scores, metrics and work. Skipped arrivals and "
        "measurements preserve learner state. Intake failures and network attempts remain "
        "recorded separately. Accepted-image savings do not establish wall-clock savings.",
        "",
        "[Summary](summary.json), [physical records](collection.json), "
        "[training replay](training-replay-verification.json), "
        "[independent audit](independent-verification.json), [frozen inputs](input_lock.json), "
        "[intake audit](../20261016-intake/independent-verification.json), "
        "[first-six source frames](../20261016-intake/contact-first-six.png), "
        "[protocol](../../../docs/project-plan/natural-health-rate-frozen.md).",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Long BDD100K archive-order quarter admission and recovery\n",
                "Saved audited results. Display cells do not train or select data.\n",
            ],
        }
    ]
    for count, name in enumerate(FIGURES, 1):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": count,
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    "root = next(p for p in [Path.cwd(), *Path.cwd().parents] "
                    "if (p / 'cafl4ds').is_dir() and (p / 'outputs').is_dir())\n",
                    f"display(Image(filename=str(root / '{config.output}/figures/{name}.png')))",
                ],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": base64.b64encode((figures / f"{name}.png").read_bytes()).decode(),
                            "text/plain": ["<saved audited scientific figure>"],
                        },
                    }
                ],
            }
        )
    converted = [
        re.sub(r"\]\(([^)]+)\)", lambda m: "](../../" + (out / m[1]).resolve().relative_to(ROOT).as_posix() + ")", line)
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
        )
        + "\n",
        encoding="utf-8",
    )
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
        else:
            for target in re.findall(r"\]\(([^)]+)\)", "".join(cell["source"])):
                assert (notebook.parent / target).resolve().is_file()
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text(encoding="utf-8")):
        assert (out / target).resolve().is_file()
    check(read(out / "input_lock.json")["hashes"])
    check(read(out / "complete.json")["hashes"])
    write(
        out / "presentation-verification.json",
        {
            "passed": True,
            "figures": list(FIGURES),
            "hashes": {str(p): digest(p) for p in [report, notebook, *figures.glob("*")]},
        },
    )
    print({"report": str(report), "notebook": str(notebook)}, flush=True)


if __name__ == "__main__":
    main()
