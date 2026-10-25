"""Primary quality, charged costs and source-level coverage admission contrasts."""

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
    args = parser.parse_args()
    out = args.output.resolve()
    summary = json.loads((out / "summary.json").read_text())
    arms = ("early_stop", "uniform", "random", "coverage", "full")
    labels = ("Early stop", "Uniform", "Random", "Coverage", "Continuous")
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    accuracy, cost, heat = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    means = [summary["heldout_means"][a]["finetune"]["balanced"] for a in arms]
    accuracy.scatter(range(5), means, s=90, color=["#777777", "#999999", "#999999", "#145da0", "#d45b16"])
    accuracy.axhline(
        summary["frozen_means"]["finetune"]["balanced"], linestyle="--", color="#333333", label="Frozen warm"
    )
    for k, value in enumerate(means):
        accuracy.annotate(f"{value:.2f}", (k, value), xytext=(0, 8), textcoords="offset points", ha="center")
    accuracy.set(
        xticks=range(5),
        xticklabels=labels,
        ylabel="Terminal balanced few-label accuracy (%)",
        title="Six fresh source-stream means",
        ylim=(min(min(means), 56) - 0.4, max(max(means), 56) + 0.5),
    )
    accuracy.tick_params(axis="x", rotation=25)
    accuracy.legend(frameon=False)
    seconds = [summary["processing_seconds"][a] for a in arms]
    cost.bar(range(5), seconds, color=["#777777", "#999999", "#999999", "#145da0", "#d45b16"])
    cost.set(
        xticks=range(5),
        xticklabels=labels,
        ylabel="Prepared-arrival policy seconds / stream",
        title="Selection + training (offline diagnostics excluded)",
    )
    cost.tick_params(axis="x", rotation=25)
    for k, value in enumerate(seconds):
        cost.text(k, value + 0.06, f"{value:.2f}", ha="center")
    names = sorted(summary["per_source_primary"])
    refs = ("uniform", "random", "early_stop", "full")
    matrix = np.array(
        [
            [
                summary["per_source_primary"][n]["coverage"]["balanced"]
                - summary["per_source_primary"][n][r]["balanced"]
                for r in refs
            ]
            for n in names
        ]
    )
    limit = max(float(np.abs(matrix).max()), 0.1)
    display = heat.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    heat.set(
        xticks=range(4),
        xticklabels=("vs uniform", "vs random", "vs early stop", "vs continuous"),
        yticks=range(6),
        yticklabels=[
            f"{n}: {summary['source_metadata'][n]['dwell']} ({summary['source_metadata'][n]['pattern'][0]} start)"
            for n in names
        ],
        title="Coverage minus reference: primary balanced capability (pp)",
    )
    for i in range(6):
        for j in range(4):
            heat.text(
                j,
                i,
                f"{matrix[i, j]:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(matrix[i, j]) > limit * 0.6 else "black",
            )
    fig.colorbar(display, ax=heat, label="Accuracy difference (pp)", shrink=0.85)
    fig.suptitle("Budgeted accepted-batch coverage admission", fontsize=15)
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory / f"coverage-admission.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
