"""Build measured health trajectories and notebook 71 from long-gate checkpoints."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_filter_factorial_analysis import clustered_interval
from cafl4ds.single_image_forecast import HEALTH
from cafl4ds.single_image_long_gate import GATES
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "gate_original": "#2166ac",
    "gate_half": "#d6604d",
    "gate_zero": "#762a83",
    "full": "#1b7837",
    "hold": "#333333",
    "independent_10": "#969696",
    "independent_25": "#cc8b27",
    "independent_50": "#587f8e",
}


def report(root, partial=False):
    lock = read(root / "input-lock.json")
    roster = read(root / "collection.json")["records"]
    arms = list(lock["specs"])
    keys = sorted({(r["history"], r["seed"]) for r in roster})
    # Use only complete paired history/seed sets; never compare unequal subsets.
    keys = [key for key in keys if {r["arm"] for r in roster if (r["history"], r["seed"]) == key} == set(arms)]
    if not keys:
        return
    records = {(r["history"], r["seed"], r["arm"]): read(root / r["path"] / "trajectory.json") for r in roster}
    settings = lock["settings"]
    clocks = np.arange(settings["start"], settings["start"] + settings["offered"] + 1, settings["checkpoint_stride"])
    first = records[(*keys[0], arms[0])]
    metrics = list(first["checkpoints"][str(clocks[0])]["metrics"])
    values = np.array(
        [
            [
                [
                    [records[(*key, arm)]["checkpoints"][str(clock)]["metrics"][metric] for metric in metrics]
                    for clock in clocks
                ]
                for arm in arms
            ]
            for key in keys
        ]
    )
    accepted = np.zeros((len(keys), len(arms), len(clocks)), int)
    panel_values = np.empty((len(keys), len(arms), len(clocks), 3, 12))
    panel_metrics = list(first["checkpoints"][str(clocks[0])]["panels"][0])
    assert len(panel_metrics) == 12
    for i, key in enumerate(keys):
        for j, arm in enumerate(arms):
            branch = root / "trajectories" / key[0] / f"seed_{key[1]}" / arm
            with np.load(branch / "trace.npz") as trace:
                accepted[i, j, 1:] = np.cumsum(trace["accepted"])[
                    settings["checkpoint_stride"] - 1 :: settings["checkpoint_stride"]
                ]
            for c, clock in enumerate(clocks):
                panels = records[(*key, arm)]["checkpoints"][str(clock)]["panels"]
                panel_values[i, j, c] = [[p[m] for m in panel_metrics] for p in panels]
                for metric in panel_metrics:
                    np.testing.assert_allclose(
                        values[i, j, c, metrics.index(metric)], np.mean([p[metric] for p in panels])
                    )
    assert np.isfinite(values).all()
    groups = np.array([key[0] for key in keys])
    unique = np.unique(groups)
    draws = np.random.default_rng(settings["bootstrap_seed"]).integers(
        0, len(unique), (settings["bootstrap_draws"], len(unique))
    )
    curves = {
        arm: {m: clustered_interval(values[:, j, :, k], groups, draws) for k, m in enumerate(metrics)}
        for j, arm in enumerate(arms)
    }
    comparisons = {}
    for gate in GATES:
        j = arms.index(gate)
        for control in [
            "full",
            "hold",
            "independent_10",
            "independent_25",
            "independent_50",
            f"global_{gate}",
            f"clip_{gate}",
        ]:
            r = arms.index(control)
            comparisons[gate + "__" + control] = {
                m: clustered_interval(values[:, j, -1, k] - values[:, r, -1, k], groups, draws)
                for k, m in enumerate(metrics)
            }
        comparisons[f"clip_{gate}__global_{gate}"] = {
            m: clustered_interval(
                values[:, arms.index(f"clip_{gate}"), -1, k] - values[:, arms.index(f"global_{gate}"), -1, k],
                groups,
                draws,
            )
            for k, m in enumerate(metrics)
        }
    summary = dict(
        partial=partial,
        paired_history_seed_sets=len(keys),
        source_histories=len(unique),
        arms={},
        curves=curves,
        comparisons=comparisons,
        clocks=clocks.tolist(),
    )
    for j, arm in enumerate(arms):
        subset = [records[(*key, arm)] for key in keys]
        summary["arms"][arm] = dict(
            mean_accepted=float(accepted[:, j, -1].mean()),
            admission=float(accepted[:, j, -1].mean() / settings["offered"]),
            policy_seconds=float(np.mean([r["policy_seconds"] for r in subset])),
            before={m: float(values[:, j, 0, k].mean()) for k, m in enumerate(metrics)},
            after={m: float(values[:, j, -1, k].mean()) for k, m in enumerate(metrics)},
        )
    atomic_json(root / "summary.json", summary)
    np.savez_compressed(
        root / "endpoints.npz",
        values=values,
        accepted=accepted,
        panels=panel_values,
        metrics=np.array(metrics),
        panel_metrics=np.array(panel_metrics),
        arms=np.array(arms),
        clocks=clocks,
        histories=groups,
        seeds=np.array([key[1] for key in keys]),
    )
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    subtitles = f"{len(unique)} histories, {len(keys)} paired history/seed sets; " + (
        "PARTIAL" if partial else "complete"
    )

    def line(ax, arm, metric, updates=False, band=False):
        result = curves[arm][metric]
        x = accepted[:, arms.index(arm)].mean(0) if updates else clocks - clocks[0]
        color = COLORS.get(arm, "#888888")
        style = "--" if arm.startswith(("clip_", "global_", "independent_")) else "-"
        ax.plot(x, result["mean"], style, color=color, label=arm.replace("_", " "), lw=1.5)
        if band and len(unique) > 1:
            lo, hi = result["ci95"]
            ax.fill_between(x, lo, hi, color=color, alpha=0.10)
        if not updates and settings["offered"] >= 384:
            ax.axvline(384, color="#aaaaaa", lw=0.6, ls=":")
        ax.grid(alpha=0.2)

    images = []
    for bank in ("selector", "evaluation"):
        for updates in (False, True):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
            for ax, name in zip(axes.flat, [*HEALTH, "objective"], strict=True):
                for arm in [*GATES, "full", "hold", "independent_10", "independent_25", "independent_50"]:
                    line(ax, arm, bank + "_" + name, updates=updates)
                ax.set(
                    title=name.replace("_", " "),
                    xlabel="Mean accepted updates" if updates else "Offered images after fork",
                )
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(f"{bank.title()} fixed-bank health and reconstruction | {subtitles}")
            name = f"{bank}-health-{'updates' if updates else 'arrivals'}.png"
            fig.savefig(folder / name, dpi=140)
            plt.close(fig)
            images.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric in zip(
        axes.flat,
        ["linear_accuracy_balanced", "linear_ce_balanced", "knn_accuracy_balanced", "knn_ce_balanced"],
        strict=True,
    ):
        for arm in [*GATES, "full", "hold", "independent_10", "independent_25", "independent_50"]:
            line(ax, arm, metric)
        ax.set(title=metric.replace("_", " "), xlabel="Offered images after fork")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Reader performance across the longer stream | " + subtitles)
    fig.savefig(folder / "reader-arrivals.png", dpi=140)
    plt.close(fig)
    images.append("reader-arrivals.png")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, gate in enumerate(GATES):
        for row, updates in enumerate((False, True)):
            ax = axes[row, col]
            for arm in [gate, f"clip_{gate}", f"global_{gate}", "full", "hold"]:
                line(ax, arm, "linear_accuracy_balanced", updates=updates, band=arm == gate)
            ax.set(
                title=gate.replace("_", " "),
                xlabel="Mean accepted updates" if updates else "Offered images after fork",
                ylabel="Linear accuracy (%)",
            )
            ax.legend(fontsize=7)
    fig.suptitle("Total budget, clip allocation and within-clip choices | " + subtitles)
    fig.savefig(folder / "allocation-comparison.png", dpi=140)
    plt.close(fig)
    images.append("allocation-comparison.png")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for row, gate in enumerate(GATES):
        for arm in [gate, f"clip_{gate}", f"global_{gate}"]:
            counts = np.diff(accepted[:, arms.index(arm)], axis=1)
            axes[row, 0].plot(np.arange(1, counts.shape[1] + 1), counts.mean(0), label=arm.replace("_", " "))
        axes[row, 0].set(title=gate + ": updates per clip", xlabel="Continuation clip", ylabel="Accepted images")
        axes[row, 0].legend(fontsize=7)
        for arm in [gate, "full", "independent_10", "independent_25", "independent_50"]:
            means = []
            for key in keys:
                with np.load(root / "trajectories" / key[0] / f"seed_{key[1]}" / arm / "trace.npz") as tr:
                    mask, losses = tr["accepted"].reshape(-1, 96), tr["training"][:, 0].reshape(-1, 96)
                    sums = (losses * mask).sum(1)
                    means.append(np.divide(sums, mask.sum(1), out=np.full(len(sums), np.nan), where=mask.sum(1) > 0))
            stacked = np.array(means)
            count = np.isfinite(stacked).sum(0)
            mean = np.divide(np.nansum(stacked, axis=0), count, out=np.full(stacked.shape[1], np.nan), where=count > 0)
            axes[row, 1].plot(np.arange(1, len(mean) + 1), mean, label=arm.replace("_", " "))
        axes[row, 1].set(title="Accepted-image augmented training loss (diet dependent)", xlabel="Continuation clip")
        axes[row, 1].legend(fontsize=7)
    fig.savefig(folder / "diet-and-training-loss.png", dpi=140)
    plt.close(fig)
    images.append("diet-and-training-loss.png")
    rows = ["| Policy | Updates | Admission | Final accuracy | Policy seconds |", "|---|---:|---:|---:|---:|"]
    for arm, r in summary["arms"].items():
        rows.append(
            f"| {arm} | {r['mean_accepted']:.1f} | {100 * r['admission']:.1f}% | "
            f"{r['after']['linear_accuracy_balanced']:.3f}% | {r['policy_seconds']:.1f} |"
        )
    introduction = (
        "# 71 — Longer gate trajectories, relaxed thresholds and allocation controls\n\n"
        + (
            "**PARTIAL: these results cover only completed paired sets.**\n\n"
            if partial
            else "**Collection complete. All predeclared policies are reported.**\n\n"
        )
        + f"{subtitles}. Each continuation contains {settings['offered']:,} offered images. "
        "The same already trained parent starts each policy. "
        "Stream clips and evaluation panels were used in prior studies; "
        "this is an exploratory follow-up using existing material.\n\n"
        "The gate predicts delta-J. Original, half and zero thresholds relax the required improvement; "
        "they are not classification confidence levels. Independent random schedules were frozen before any gate ran. "
        "Global and clip controls are retrospective: they inherit total counts or per-clip counts respectively.\n\n"
        "Health uses fixed selector and evaluation banks with fixed masks. "
        "Accepted-image training loss is shown separately because its image diet differs. "
        "All reader curves are measured, support-fitted linear/kNN results; there is no fine-tuning curve. "
        "The dotted vertical marker identifies the old 384-arrival horizon. "
        "Whole-history intervals are exploratory and unadjusted.\n\n" + "\n".join(rows) + "\n\n"
        "Paired endpoint comparisons and whole-history intervals are in summary.json. "
        "Raw arrays, per-panel metrics, decisions, and final model/optimizer states are retained. "
        "The matched controls do not create an additive causal decomposition.\n"
    )
    text = introduction + "\n\n" + "\n\n".join(f"![{name}](figures/{name})" for name in images)
    (root / "report.md").write_text(text + "\n", encoding="utf-8")
    cells = [dict(cell_type="markdown", metadata={}, source=introduction.splitlines(keepends=True))]
    for name in images:
        payload = base64.b64encode((folder / name).read_bytes()).decode()
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=[f"![{name}](attachment:{name})"],
                attachments={name: {"image/png": payload}},
            )
        )
    notebook = dict(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        nbformat=4,
        nbformat_minor=4,
    )
    destinations = [root / "71_single_image_long_gate.ipynb"]
    if root.name != "smoke":
        destinations.append(ROOT / "notebooks/sums/71_single_image_long_gate.ipynb")
    for path in destinations:
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    report(args.output.resolve(), partial=args.partial)
