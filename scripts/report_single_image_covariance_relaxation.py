"""Live N76 task, health, reconstruction and learning-rate measurements."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint
from scripts.report_single_image_reconstruction import interval_means
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "76_single_image_covariance_relaxation.ipynb"
NAME = "N76: adaptive LR, 1.30 warm"
COLORS = {
    NAME: "#2166ac",
    "N75: adaptive LR, 1.25 warm": "#b85c22",
    "N74: fixed LR, 1.25 warm": "#9c75a5",
    "N74: corrected J": "#6c8a59",
}


def load_record(branch: Path) -> tuple[dict[str, Any] | None, Path | None, bool]:
    """Only published immutable generations are reportable."""
    if (branch / "trajectory.json").exists():
        return read(branch / "trajectory.json"), branch / "trace.npz", True
    saved = load_checkpoint(branch)
    if saved is None:
        return None, None, False
    generation, record = saved
    return record, generation / "trace.npz", False


def series(record: dict[str, Any], key: str, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a metric only within the new trajectory's measured horizon."""
    clocks = sorted(t for t in map(int, record["checkpoints"]) if t <= start + count)
    values = np.array([record["checkpoints"][str(t)]["metrics"][key] for t in clocks])
    return np.array(clocks) - start, values


def finish_axes(ax: Axes, title: str, ylabel: str | None = None) -> None:
    """Apply consistent arrival axes to the reporting panels."""
    ax.set(title=title, xlabel="Offered continuation images", ylabel=ylabel)
    ax.grid(alpha=0.2)


def report(root: Path) -> None:  # noqa: C901 - One report assembles five related notebook figures.
    """Publish saved diagnostics without evaluating or changing the learning model."""
    lock = read(root / "input-lock.json")
    settings, start = lock["settings"], lock["settings"]["start"]
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    branch = directory / settings["arm"]
    record, path, complete = load_record(branch)
    with np.load(ROOT / settings["source"] / "references.npz") as raw:
        warm_rank, warm_variance, warm_covariance = raw["selector"][:3]
    if record is None and (directory / "common/readout.json").exists():
        common = read(directory / "common/readout.json")
        oldparent = read(Path(lock["prior_learning_rate_generation"]) / "commit.json")["checkpoints"][str(start)]
        common = dict(common, metrics={**oldparent["metrics"], **common["metrics"]})
        record = {"checkpoints": {str(start): common}, "costs": {}}
    n = max(map(int, record["checkpoints"])) - start if record else 0
    selected, admitted, attempted = np.zeros(n), np.zeros(n, bool), np.zeros((n, 4), bool)
    training, nominal_training, trial_covariance = np.zeros(n), np.zeros(n), np.zeros((n, 4))
    if path is not None:
        with np.load(path) as trace:
            selected = trace["selected_factor"][:n].copy()
            admitted = trace["j_admitted"][:n].copy()
            attempted = trace["attempted"][:n].copy()
            training = trace["training"][:n, 0].copy()
            nominal_training = trace["trial_training"][:n, 0, 0].copy()
            trial_covariance = trace["trial_health"][:n, :, 2].copy()
    accepted = selected > 0
    summary = {
        "single_trajectory": True,
        "target_offered": settings["offered"],
        "offered": n,
        "complete": complete,
        "new_accept_all": 0,
        "covariance_cap": lock["covariance_cap"],
        "covariance_warm_multiplier": settings["covariance_warm_multiplier"],
        "previous_covariance_cap": lock["previous_covariance_cap"],
        "accepted": int(accepted.sum()),
        "j_admitted": int(admitted.sum()),
        "rescued_updates": int(((selected > 0) & (selected < 1)).sum()),
        "trial_steps": int(attempted.sum()),
        "rejected_trial_steps": int(attempted.sum() - accepted.sum()),
        "nominal_rate_equivalent": float(selected.sum()),
        "last_accepted_arrival": int(np.flatnonzero(accepted)[-1] + 1) if accepted.any() else None,
    }
    text = (
        "# 76 — Adjustable learning rate with a 1.30-warm covariance ceiling\n\n"
        "One fresh_00 / seed-5003 trajectory restarts from the identical complete N73 576-update parent. "
        "The corrected J predictor, threshold, stream, image/view seeds and actual AdamW rate ladder "
        "are unchanged from N75. Only the covariance ceiling changes from 1.25 to 1.30 times warm. "
        "No new accept-all or fixed-rate trajectory is run.\n\n"
        "The 1.30 choice is exploratory, informed by the stopped N75 trajectory. A wider fixed ceiling "
        "can postpone rather than resolve a stall. Final/development probes are measurements only. "
        "All failed trials restore complete learning state; each image starts its rate search at nominal.\n\n"
        f"Latest committed prefix: **{n:,} arrivals; {summary['accepted']:,} retained updates; "
        f"{summary['rescued_updates']:,} smaller-rate rescues.**\n\n"
        "Reference curves show only previously saved measurements up to this arrival horizon. "
        "N75 was stopped at user request; its last checkpoint remains a descriptive reference. "
        "Curves stop at each policy's last measured point; no flat continuation is invented.\n\n"
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    images = []

    def save(fig: Figure, name: str) -> None:
        picture = figures / f"{name}.png"
        fig.savefig(picture, dpi=125)
        plt.close(fig)
        images.append(picture)

    if record:
        prior = ROOT / settings["prior"]
        corrected = prior / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "corrected_J/trajectory.json"
        policies = {
            NAME: record,
            "N75: adaptive LR, 1.25 warm": read(Path(lock["prior_learning_rate_generation"]) / "commit.json"),
            "N74: fixed LR, 1.25 warm": read(Path(lock["prior_fixed_generation"]) / "commit.json"),
            "N74: corrected J": read(corrected),
        }
        ticks, accuracy = series(record, "linear_accuracy_balanced", start, n)
        summary.update(
            latest_accuracy=float(accuracy[-1]),
            max_drawdown_pp=float((np.maximum.accumulate(accuracy) - accuracy).max()),
            current_covariance_warm_ratio=float(
                series(record, "selector_covariance", start, n)[1][-1] / warm_covariance
            ),
            selected_counts={str(f): int((selected == f).sum()) for f in settings["learning_rate_factors"]},
        )
        text += (
            f"Linear accuracy: **{summary['latest_accuracy']:.3f}%**; maximum observed checkpoint drawdown: "
            f"**{summary['max_drawdown_pp']:.3f} pp**. Trials: {summary['trial_steps']:,}; discarded trials: "
            f"{summary['rejected_trial_steps']:,}; "
            f"sum of retained LR factors: {summary['nominal_rate_equivalent']:.3f}. "
            "The latter is exposure accounting, not equivalent optimization.\n\n"
        )

        def plot_metric(
            ax: Axes,
            key: str,
            title: str,
            transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
        ) -> None:
            for name, saved in policies.items():
                x, y = series(saved, key, start, n)
                if transform:
                    y = transform(y, saved)
                ax.plot(
                    x,
                    y,
                    label=name,
                    color=COLORS[name],
                    lw=1.7 if name == NAME else 1.2,
                    alpha=1 if name == NAME else 0.7,
                    marker=".",
                    markersize=3,
                )
            finish_axes(ax, title)

        fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
        plot_metric(axes[0, 0], "linear_accuracy_balanced", "Balanced linear-probe accuracy (%)")
        plot_metric(
            axes[0, 1],
            "linear_accuracy_balanced",
            "Running-peak drawdown (percentage points)",
            lambda y, _: np.maximum.accumulate(y) - y,
        )
        plot_metric(axes[1, 0], "linear_accuracy_day", "Day linear-probe accuracy (%)")
        plot_metric(axes[1, 1], "linear_accuracy_night", "Night linear-probe accuracy (%)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Task performance and checkpoint drawdown")
        save(fig, "performance")

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        health = [
            ("selector_covariance", "Covariance / warm", lambda y, _: y / warm_covariance),
            ("selector_variance", "Variance / warm", lambda y, _: y / warm_variance),
            ("selector_rank", "RankMe (raw uncentered features)", None),
            ("selector_objective", "Selector J (lower is better)", None),
            ("selector_cka", "CKA drift from warm", None),
            (
                "selector_covariance",
                "Spread-adjusted covariance / warm (diagnostic)",
                lambda y, r: (
                    (y / series(r, "selector_variance", start, n)[1] ** 2) / (warm_covariance / warm_variance**2)
                ),
            ),
        ]
        for ax, (key, title, transform) in zip(axes.flat, health, strict=True):
            plot_metric(ax, key, title, transform)
        axes[0, 0].axhline(1.25, color="#b85c22", ls=":", label="N75 ceiling: 1.25")
        axes[0, 0].axhline(1.30, color="black", ls="--", label="N76 ceiling: 1.30")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].axhline(1, color="gray", ls=":")
        axes[0, 2].axhline(warm_rank, color="gray", ls=":")
        axes[1, 2].axhline(1, color="gray", ls=":")
        fig.suptitle("Representation health; spread adjustment is exploratory and does not control training")
        save(fig, "health")

        verified = 0
        for clock in map(int, record["checkpoints"]):
            rawpath = directory / "common/raw.npz" if clock == start else branch / f"checkpoint_{clock}/raw.npz"
            with np.load(rawpath) as raw:
                for bank in ("selector", "evaluation"):
                    np.testing.assert_allclose(
                        raw[f"{bank}_reconstruction"].mean(),
                        record["checkpoints"][str(clock)]["metrics"][f"{bank}_bank_reconstruction"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
            verified += 1
        summary["reconstruction_checkpoint_assays_verified"] = verified
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
        plot_metric(axes[0], "selector_bank_reconstruction", "Fixed selector-bank reconstruction")
        plot_metric(axes[1], "evaluation_bank_reconstruction", "Held-out fixed-bank reconstruction")
        axes[0].legend(fontsize=8)
        ends, means, counts = interval_means(training, accepted, n)
        _, candidate_means, candidate_counts = interval_means(nominal_training, admitted, n)
        axes[2].scatter(np.flatnonzero(accepted) + 1, training[accepted], s=7, alpha=0.2, color=COLORS[NAME])
        axes[2].plot(ends, [np.nan if v is None else v for v in means], label="Retained-update mean / 384 arrivals")
        axes[2].plot(
            ends, [np.nan if v is None else v for v in candidate_means], ls="--", label="All J-admitted candidates"
        )
        finish_axes(axes[2], "Actual per-image reconstruction before its update; empty intervals are gaps")
        axes[2].legend(fontsize=8)
        fig.suptitle("Reconstruction loss: comparable fixed banks and selection-dependent training images")
        save(fig, "reconstruction")
        summary["training_intervals"] = {
            "offered_end": ends,
            "accepted_mean_loss": means,
            "accepted_counts": counts,
            "candidate_mean_loss": candidate_means,
            "candidate_counts": candidate_counts,
        }

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        axes[0, 0].plot(ticks, [admitted[:k].sum() for k in ticks], label="J-admitted")
        axes[0, 0].plot(ticks, [accepted[:k].sum() for k in ticks], label="Retained updates")
        for name, saved in list(policies.items())[1:]:
            if "done" in saved and "offered" not in saved and name.startswith("N75"):
                with np.load(Path(lock["prior_learning_rate_generation"]) / "trace.npz") as old:
                    oldselected = old["selected_factor"][: min(n, saved["done"])].copy()
                axes[0, 0].plot(
                    np.arange(len(oldselected)) + 1, np.cumsum(oldselected > 0), ls="--", label=name, color=COLORS[name]
                )
        axes[0, 0].legend(fontsize=7)
        finish_axes(axes[0, 0], "Admissions versus live updates")
        axes[0, 1].plot(
            ends, [100 * c / min(384, k) for k, c in zip(ends, counts, strict=True)], label="Retained / offered (%)"
        )
        axes[0, 1].plot(
            ends,
            [100 * c / min(384, k) for k, c in zip(ends, candidate_counts, strict=True)],
            label="J-admitted / offered (%)",
        )
        finish_axes(axes[0, 1], "Interval admission and retention")
        axes[0, 1].legend(fontsize=7)
        for factor in settings["learning_rate_factors"]:
            axes[0, 2].plot(ticks, [(selected[:k] == factor).sum() for k in ticks], label=f"LR × {factor:g}")
        finish_axes(axes[0, 2], "Retained updates by chosen learning rate")
        axes[0, 2].legend(fontsize=7)
        axes[1, 0].plot(ticks, [accepted[:k].sum() for k in ticks], label="Retained update count")
        axes[1, 0].plot(ticks, [selected[:k].sum() for k in ticks], label="Sum of LR factors")
        finish_axes(axes[1, 0], "Updates and rate-weighted exposure")
        axes[1, 0].legend(fontsize=7)
        for j, factor in enumerate(settings["learning_rate_factors"]):
            mask = attempted[:, j]
            axes[1, 1].scatter(
                np.flatnonzero(mask) + 1,
                trial_covariance[mask, j] / warm_covariance,
                s=4,
                alpha=0.2,
                label=f"LR × {factor:g}",
            )
        axes[1, 1].axhline(1.25, color="#b85c22", ls=":")
        axes[1, 1].axhline(1.30, color="black", ls="--")
        finish_axes(axes[1, 1], "Attempted covariance / warm; both ceilings")
        axes[1, 1].legend(fontsize=7)
        trial_counts = [int(attempted[max(0, k - 384) : k].sum()) for k in ends]
        axes[1, 2].plot(ends, trial_counts, label="Actual trial steps")
        axes[1, 2].plot(ends, [t - c for t, c in zip(trial_counts, counts, strict=True)], label="Discarded trials")
        finish_axes(axes[1, 2], "Trial burden per 384 offered images")
        axes[1, 2].legend(fontsize=7)
        fig.suptitle("Controller behavior and learning-rate choices")
        save(fig, "controller")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        for group in ("balanced", "day", "night"):
            x, y = series(record, f"development_accuracy_{group}", start, n)
            axes[0].plot(x, y, label=group)
            x, y = series(record, f"development_soft_error_{group}", start, n)
            axes[1].plot(x, y, label=group)
        finish_axes(axes[0], "Development linear-probe accuracy (%)")
        finish_axes(axes[1], "Development soft boundary error (lower is better)")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle("Development diagnostics; these measurements do not select updates")
        save(fig, "development")
    else:
        text += "Prepared; shared-parent assay and GPU preflight pending.\n"
    text += (
        "\nFixed reconstruction uses identical bank images, pixels and masks at every checkpoint. "
        "Training reconstruction uses the actual arriving view before its update; candidate means count "
        "each image once, without counting retries again. Both can move differently.\n\n"
        "Spread-adjusted covariance is C / mean-coordinate-variance², relative to warm. "
        "It is an exploratory diagnostic, not a validated redundancy or task-retention guarantee. "
        "RankMe uses raw uncentered features. Checkpoint drawdown can miss dips between assays.\n"
    )
    atomic_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        text + "\n" + "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in images), encoding="utf-8"
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}]
    for picture in images:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"![{picture.stem}](attachment:{picture.name})"],
                "attachments": {picture.name: {"image/png": base64.b64encode(picture.read_bytes()).decode()}},
            }
        )
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    atomic_json(root / NOTEBOOK, notebook)
    atomic_json(ROOT / "notebooks/sums" / NOTEBOOK, notebook)
