"""Render accumulated-drift timing and held-history warning results."""

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
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), layout="constrained")
    landmarks = [0, 4, 8, 16, 32]
    states = ["day_end", "night_end", "return_early"]
    colors = ["#267d66", "#bd593b", "#a47524"]
    for state, color in zip(states, colors, strict=True):
        rows = [r for r in summary["controls"] if r["checkpoint"] == state]
        drift = np.array([[r["landmarks"][str(j)]["geometry"]["cka_drift"] for j in landmarks] for r in rows])
        effect = np.array(
            [
                [
                    r["landmarks"]["32"]["scores"]["finetune"]["balanced"]
                    - r["landmarks"][str(j)]["scores"]["finetune"]["balanced"]
                    for j in landmarks
                ]
                for r in rows
            ]
        )
        label = state.replace("_", " ")
        for ax, values in ((axes[0, 0], drift), (axes[0, 1], effect)):
            ax.plot(landmarks, values.mean(0), "o-", color=color, label=label)
            ax.fill_between(landmarks, values.min(0), values.max(0), color=color, alpha=0.12)
    axes[0, 0].set(title="Selected controls: accumulated CKA movement", ylabel="1 − CKA versus local starting state")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set(
        title="Selected controls: benefit or loss still ahead", ylabel="Continue to 32 minus stop here (balanced pp)"
    )
    axes[0, 1].axhline(0, color="#777777", lw=1)
    for ax in axes[0]:
        ax.set(xlabel="SSL updates completed", xticks=landmarks)
        ax.axvline(8, color="#306da2", ls=":", alpha=0.5)
    names, colors = ["mean", "count_loss", "cka"], ["#b5b5b5", "#8e99a4", "#306da2"]
    x = np.arange(3)
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        values = [summary["forecasts"][str(j)]["models"][name]["mse"] for j in (4, 8, 16)]
        axes[1, 0].bar(
            x + (i - 1) * 0.24,
            values,
            width=0.24,
            color=color,
            label={"mean": "Mean", "count_loss": "Count + loss", "cka": "Count + loss + CKA"}[name],
        )
    axes[1, 0].set(
        xticks=x,
        xticklabels=["4 (secondary)", "8 (primary)", "16 (secondary)"],
        xlabel="Warning landmark",
        ylabel="Remaining domain-damage forecast MSE (pp²)",
        title="Held-history prediction of future damage",
    )
    axes[1, 0].legend(frameon=False, fontsize=9)
    points = summary["forecasts"]["8"]["points"]
    for stream in sorted({p["stream"] for p in points}):
        values = [p for p in points if p["stream"] == stream]
        axes[1, 1].scatter([p["cka_drift"] for p in values], [p["damage"] for p in values], s=35, label=stream)
    axes[1, 1].set(
        title="Primary: drift at 8 versus damage still ahead",
        xlabel="Accumulated CKA drift at update 8",
        ylabel="Worst-domain damage, updates 8 → 32 (pp)",
    )
    axes[1, 1].axhline(0, color="#777777", lw=1)
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Development screen: six reused histories; controls share one historical stream\n"
        "Control shading shows the range across three optimizer seeds, not a confidence interval",
        fontsize=11,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"health-warning.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
