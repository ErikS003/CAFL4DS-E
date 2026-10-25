"""Render hindsight stopping headroom and retrospective query-half sensitivity."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    result = json.loads((out / "summary.json").read_text())["primary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), layout="constrained")
    curve = result["fixed_curve"]
    axes[0].plot(
        [r["dose"] for r in curve],
        [r["mean_effect_vs_hold"][0] for r in curve],
        "o-",
        color="#306da2",
        label="Fixed stop",
    )
    for name, color, label in (
        ("unconstrained", "#bc6534", "Unconstrained hindsight"),
        ("retention_aware", "#267d66", "Retention-aware hindsight"),
    ):
        values = result["hindsight"][name]["evaluation"]
        axes[0].scatter(
            values["mean_updates"], values["mean_effect_vs_hold"][0], marker="*", s=140, color=color, label=label
        )
    axes[0].set(
        title="Optimistic local quality–work headroom",
        xlabel="Mean additional SSL updates",
        ylabel="Balanced effect versus stop now (pp)",
    )
    axes[0].legend(frameon=False, fontsize=8)
    count = [r["domain_loss_gt_1_cases"] for r in curve] + [
        result["hindsight"][name]["evaluation"]["domain_loss_gt_1_cases"]
        for name in ("unconstrained", "retention_aware")
    ]
    axes[1].bar(range(7), count, color=["#306da2"] * 5 + ["#bc6534", "#267d66"])
    axes[1].set(
        xticks=range(7),
        xticklabels=["0", "4", "8", "16", "32", "Best\nper case", "Guarded\nper case"],
        title="Full-query domain losses",
        xlabel="Additional stopping dose",
        ylabel="Cases with either domain loss >1pp",
        ylim=(0, 18),
    )
    axes[1].tick_params(axis="x", labelsize=8)
    for i, name in enumerate(("A_to_B", "B_to_A")):
        value = result["query_half_folds"][name]["contrasts"]["vs_mixture_balanced"]
        mean = value["ordinary"]["mean_pp"]
        for kind, width, color in (("adjusted", 1.4, "#9ab7cf"), ("ordinary", 4, "#306da2")):
            lo, hi = value[kind]["ci95_pp"]
            axes[2].errorbar(
                i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=color, elinewidth=width, capsize=4
            )
    axes[2].axhline(0, color="#777777", lw=1)
    axes[2].axhline(0.5, color="#267d66", lw=1, ls=":", label="Materiality floor")
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Choose A, score B", "Choose B, score A"],
        title="Choice transfer versus a mean-budget baseline",
        ylabel="Guarded hindsight minus common mixture (pp)",
    )
    axes[2].tick_params(axis="x", labelsize=9)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Six reused histories, 18 checkpoint cases; no new training or fresh confirmation", fontsize=12)
    fig.supxlabel(
        "Left/center use the same outcomes for selection and evaluation. Right: ordinary (thick) / family-10 (thin) conditional 95% intervals.",
        fontsize=9,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"stopping-headroom.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
