"""Report all locked batch-one forecasts in native units with source uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_forecast import error_summary

PRIMARY = (
    "rank",
    "variance",
    "covariance",
    "cka",
    "cosine",
    "bank_reconstruction",
    "incoming_reconstruction",
    "linear_accuracy_balanced",
    "knn_accuracy_balanced",
    "finetune20_accuracy_balanced",
    "finetune20_ce_balanced",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(out: Path):
    settings = read(out / "input_lock.json")["settings"]
    paths = sorted((out / "trajectories").glob("*/seed_*"))
    groups = sorted({p.parent.name for p in paths})
    if len(paths) != 12 or len(groups) != 6:
        raise ValueError("Incomplete scientific run")
    trajectories, full = [], []
    for path in paths:
        with np.load(path / "series.npz", allow_pickle=False) as a:
            metrics = a["metrics"].tolist()
        with np.load(path / "forecasts.npz", allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
        trajectories.append((path.parent.name, path.name, data))
        obs = read(path / "observations.json")
        full.append((path.parent.name, path.name, obs["full_finetune"]))
    models = [k for k in trajectories[0][2] if k != "targets"]
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = rng.integers(0, len(groups), (settings["bootstrap_draws"], len(groups)))
    result, rows = {}, []
    for j, metric in enumerate(metrics):
        y = np.concatenate([d["targets"][:, j] for _, _, d in trajectories])
        baseline_source = np.array(
            [
                np.mean(np.concatenate([d["targets"][:, j] ** 2 for g, _, d in trajectories if g == group]))
                for group in groups
            ]
        )
        scores = {}
        for model in models:
            prediction = np.concatenate([d[model][:, j] for _, _, d in trajectories])
            score = error_summary(y, prediction)
            source = {}
            for group in groups:
                values = [(d["targets"][:, j], d[model][:, j]) for g, _, d in trajectories if g == group]
                source[group] = error_summary(
                    np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values])
                )
            mse = np.array([source[g]["mse"] for g in groups])
            mae = np.array([source[g]["mae"] for g in groups])
            score["rmse"] = float(np.sqrt(mse.mean()))
            score["mae"] = float(mae.mean())
            score["by_history"] = source
            b_rmse = np.sqrt(mse[draws].mean(1))
            b_mae = mae[draws].mean(1)
            score["rmse_cluster_ci95"] = np.quantile(b_rmse, [0.025, 0.975]).tolist()
            score["mae_cluster_ci95"] = np.quantile(b_mae, [0.025, 0.975]).tolist()
            baseline_rmse = float(np.sqrt(baseline_source.mean()))
            score["rmse_reduction_vs_persistence_percent"] = (
                100 * (1 - score["rmse"] / baseline_rmse) if baseline_rmse else None
            )
            boot_base = np.sqrt(baseline_source[draws].mean(1))
            score["paired_rmse_reduction_ci95_percent"] = (
                np.quantile(100 * (1 - b_rmse / boot_base), [0.025, 0.975]).tolist() if np.all(boot_base > 0) else None
            )
            score["rmse_reduction_by_history_percent"] = {
                g: 100 * (1 - np.sqrt(mse[i] / baseline_source[i])) if baseline_source[i] else None
                for i, g in enumerate(groups)
            }
            scores[model] = score
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    **{k: v for k, v in score.items() if not isinstance(v, (dict, list))},
                }
            )
        # Paired image ablation and online/offline comparisons are fixed in advance.
        comparisons = {}
        for candidate, baseline in (
            ("offline_history_image", "offline_history"),
            ("online_history_image", "online_history"),
            ("online_history_image", "offline_history_image"),
            ("online_label_free_image", "offline_label_free_image"),
        ):
            candidate_mse = np.array([scores[candidate]["by_history"][g]["mse"] for g in groups])
            base_mse = np.array([scores[baseline]["by_history"][g]["mse"] for g in groups])
            boot_base = np.sqrt(base_mse[draws].mean(1))
            comparisons[candidate + "_vs_" + baseline] = {
                "rmse_reduction_percent": float(100 * (1 - np.sqrt(candidate_mse.mean() / base_mse.mean())))
                if base_mse.mean()
                else None,
                "paired_ci95_percent": np.quantile(
                    100 * (1 - np.sqrt(candidate_mse[draws].mean(1)) / boot_base), [0.025, 0.975]
                ).tolist()
                if np.all(boot_base > 0)
                else None,
            }
        result[metric] = {"models": scores, "paired_comparisons": comparisons}
    with (out / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Boundary diagnostics use metadata only at scoring, never as forecast inputs.
    boundary = {}
    for subset, select in (
        ("first_four_after_clip_join", np.arange(64, 192) % 16 < 4),
        ("other_arrivals", np.arange(64, 192) % 16 >= 4),
    ):
        boundary[subset] = {}
        for j, metric in enumerate(metrics):
            y = np.concatenate([d["targets"][select, j] for _, _, d in trajectories])
            boundary[subset][metric] = {
                model: error_summary(y, np.concatenate([d[model][select, j] for _, _, d in trajectories]))
                for model in models
            }
    sparse = {}
    for metric in full[0][2][0]["metrics"]:
        truth, frozen, live, history_ids = [], [], [], []
        for group, _, states in full:
            levels = np.array([s["metrics"][metric] for s in states])
            delta = np.diff(levels)
            truth.extend(delta[2:])
            frozen.extend([delta[:2].mean()] * 4)
            live.extend([delta[:i].mean() for i in range(2, 6)])
            history_ids.extend([group] * 4)
        y = np.array(truth)
        sparse[metric] = {
            "horizon_images": 32,
            "n": len(y),
            "initial_training_transitions_per_trajectory": 2,
            "persistence": error_summary(y, np.zeros_like(y)),
            "offline_mean": error_summary(y, np.array(frozen)),
            "online_mean": error_summary(y, np.array(live)),
        }
    summary = {
        "primary_error_units": "native units; accuracy and gain in percentage points",
        "target": "post-image minus pre-image state; next-level prediction = pre-state + predicted delta",
        "prefix_images": 64,
        "test_images_per_trajectory": 128,
        "test_transitions": 1536,
        "histories": 6,
        "optimizer_seeds": 2,
        "scientific_ssl_updates": 2304,
        "metrics": result,
        "boundary_diagnostics": boundary,
        "sparse_finetune100": sparse,
        "multiple_testing_adjusted": False,
        "independent_confirmation": False,
        "online_predictors_receive_past_downstream_labels": True,
        "selection_policy_tested": False,
    }
    atomic_json(out / "summary.json", summary)
    lines = [
        "# Batch-one next-state prediction",
        "",
        "Twelve physical MAE trajectories: six reused clip-disjoint histories × two seeds.",
        "Each has 192 one-image updates; first64 train the predictors, next128 are held-out sequential forecasts.",
        "All 1536 test forecasts precede their outcomes. Online models then learn that outcome.",
        "",
        "RMSE / MAE below are errors on next-state changes, in each metric's native units.",
        "Adding the pre-state gives exactly the same next-level errors. Accuracy uses percentage points.",
        "Frozen/online columns use history plus current-image features; label-free uses only health/history/image inputs.",
        "",
        "| Metric | Persistence RMSE / MAE | Offline RMSE / MAE | Online RMSE / MAE | Label-free online RMSE / MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY:
        score = result[metric]["models"]
        cells = [
            f"{score[m]['rmse']:.5g} / {score[m]['mae']:.5g}"
            for m in ("persistence", "offline_history_image", "online_history_image", "online_label_free_image")
        ]
        lines.append("| " + metric + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Fixed comparisons",
        "",
        "Positive RMSE reduction means better than persistence; intervals resample the six entire histories.",
        "The seed replicates stay inside their history. Intervals are descriptive and unadjusted across metrics.",
        "",
        "| Metric | Offline reduction % | Online reduction % [95% interval] | Image contribution to online reduction % | Actual ties % |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY:
        entry = result[metric]
        score = entry["models"]
        online = score["online_history_image"]
        ci = online["paired_rmse_reduction_ci95_percent"]
        extra = entry["paired_comparisons"]["online_history_image_vs_online_history"]["rmse_reduction_percent"]
        lines.append(
            f"| {metric} | {score['offline_history_image']['rmse_reduction_vs_persistence_percent']:.2f} | {online['rmse_reduction_vs_persistence_percent']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {extra:.2f} | {100 * online['tie_fraction']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "State/history models observe previous labeled readouts. They test forecasting with paid diagnostic measurements.",
        "Label-free inputs omit those readouts; training their downstream targets still uses labels. This is not label-free online supervision.",
        "Twenty-step copied fine-tuning is the dense assay. The separate100-step assay is measured every32 images, with only two prefix transitions per trajectory; its coarse baselines are secondary.",
        "A forecast of geometry movement is not a forecast of benefit. A forecast of single-image update effect is not evidence that using it for selection improves training.",
        "These are same-stream chronological tests on previously inspected development sources, with artificial clip joins. Future histories, learning rates, long horizons and policy efficacy are untested.",
        "All twelve models and all metrics, direction scores, source intervals and fixed pair comparisons are in scores.csv and summary.json.",
        "Raw per-state reader probabilities, bank embeddings, image features, training losses and optimizer endpoints are retained for audit.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary, trajectories, metrics


def plot(out, summary, trajectories, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = ("rank", "bank_reconstruction", "linear_accuracy_balanced", "finetune20_accuracy_balanced")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for metric, ax in zip(display, axes.flat, strict=True):
        j = metrics.index(metric)
        d = trajectories[0][2]
        x = np.arange(65, 193)
        ax.plot(x, d["targets"][:, j], label="Observed change", lw=1)
        for model, label in (
            ("offline_history_image", "Frozen predictor"),
            ("online_history_image", "Online predictor"),
        ):
            ax.plot(x, d[model][:, j], label=label, lw=1, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=metric, xlabel="Arriving image", ylabel="One-image change")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Batch 1: every forecast precedes the next image's update\nFirst fixed history and seed; all-history errors are reported separately"
    )
    fig.savefig(out / "forecast-traces.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(PRIMARY))
    width = 0.25
    for offset, model, label in (
        (-1, "offline_history_image", "Frozen + image"),
        (0, "online_history_image", "Online + image"),
        (1, "online_label_free_image", "Online label-free inputs"),
    ):
        values = [
            summary["metrics"][m]["models"][model]["rmse"] / summary["metrics"][m]["models"]["persistence"]["rmse"]
            for m in PRIMARY
        ]
        ax.bar(x + offset * width, values, width, label=label)
    ax.axhline(1, color="black", ls="--", label="Persistence")
    ax.set(
        xticks=x,
        xticklabels=PRIMARY,
        ylabel="RMSE / persistence RMSE (lower is better)",
        title="All held-out next-image forecasts; same metric-specific denominator",
    )
    ax.tick_params(axis="x", rotation=65)
    ax.legend(fontsize=8)
    fig.savefig(out / "rmse-comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    summary, trajectories, metrics = summarize(out)
    plot(out, summary, trajectories, metrics)


if __name__ == "__main__":
    main()
