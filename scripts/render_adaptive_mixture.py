"""Plot matched-dose mixture outcomes, context allocations and charged costs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    summary = json.loads((out / "summary.json").read_text())
    paths = json.loads((out / "collection.json").read_text())["records"]
    rows = [json.loads(Path(p).read_text()) for p in paths]
    arms = ["warm", "current", "fixed", "random", "adaptive"]
    colors = ["#bbbbbb", "#8e99a4", "#537b97", "#b6a180", "#267d66"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), layout="constrained")
    terminal = summary["horizons"]["terminal"]
    x = np.arange(5)
    for i, metric in enumerate(("balanced", "A", "B")):
        axes[0, 0].bar(
            x + (i - 1) * 0.24,
            [terminal["means"][a]["finetune"][metric] for a in arms],
            width=0.24,
            label={"balanced": "Balanced", "A": "Day", "B": "Night"}[metric],
        )
    axes[0, 0].set(
        xticks=x, xticklabels=arms, ylabel="Fine-tuning accuracy (%)", title="Terminal quality and domain retention"
    )
    axes[0, 0].set_ylim(0, 70)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    for i, arm in enumerate(("current", "fixed", "random", "warm")):
        value = terminal["contrasts"][arm]["finetune"]["balanced"]
        mean = value["ordinary"]["mean_pp"]
        for kind, width, color in (("adjusted", 1.3, "#a5c5ba"), ("ordinary", 4, "#267d66")):
            lo, hi = value[kind]["ci95_pp"]
            axes[0, 1].errorbar(
                i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, elinewidth=width, capsize=4
            )
    axes[0, 1].axhline(0, color="#777777", lw=1)
    axes[0, 1].set(
        xticks=range(4),
        xticklabels=["Current", "Fixed", "Random", "Warm"],
        ylabel="Adaptive minus comparator (balanced pp)",
        title="Six-source intervals: ordinary / family-12",
    )
    for arm, color in zip(arms[1:], colors[1:], strict=True):
        selected = [r for r in rows if r["arm"] == arm]
        curve = np.array([[v["historical_count"] for v in r["trace"]] for r in selected]).mean(0)
        axes[1, 0].plot(np.arange(192) + 1, curve, color=color, label=arm, alpha=0.85)
    axes[1, 0].set(
        xlabel="Online update",
        ylabel="Mean replay images per batch of 32",
        title="Replay timing at exactly matched total dose",
        ylim=(-0.5, 17),
    )
    axes[1, 0].legend(frameon=False, fontsize=9, ncol=2)
    costs = summary["costs_per_trajectory"]
    bottom = np.zeros(4)
    for key, label in (
        ("ssl_seconds", "SSL"),
        ("gather_seconds", "Gather"),
        ("reference_seconds", "Frozen features"),
        ("decision_seconds", "Decision"),
        ("reservoir_plan_seconds", "Memory plan"),
    ):
        values = np.array([costs[a][key] for a in arms[1:]])
        axes[1, 1].bar(range(4), values, bottom=bottom, label=label)
        bottom += values
    axes[1, 1].set(
        xticks=range(4),
        xticklabels=arms[1:],
        ylabel="Measured seconds per trajectory",
        title="Prepared-processing cost; readouts excluded",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Six reused stream histories, three seeds; 192 updates each; no deployed-efficiency qualification", fontsize=11
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"adaptive-mixture.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
