"""N75 checkpoint report: per-image LR search versus saved fixed-rate policies."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]


def report(root):
    lock = read(root / "input-lock.json")
    settings = lock["settings"]
    start = settings["start"]
    branch = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / settings["arm"]
    data = None
    if (branch / "trajectory.json").exists():
        record = read(branch / "trajectory.json")
        path = branch / "trace.npz"
        complete = True
    else:
        checkpoint = load_checkpoint(branch)
        if checkpoint is None:
            record = None
        else:
            generation, record = checkpoint
            path = generation / "trace.npz"
            complete = False
    if record is not None:
        clocks = sorted(map(int, record["checkpoints"]))
        n = clocks[-1] - start
        with np.load(path) as tr:
            selected = tr["selected_factor"][:n].copy()
            attempted = tr["attempted"][:n].copy()
        acc = np.array([record["checkpoints"][str(t)]["metrics"]["linear_accuracy_balanced"] for t in clocks])
        data = dict(
            record=record, clocks=clocks, n=n, selected=selected, attempted=attempted, accuracy=acc, complete=complete
        )
    summary = dict(
        single_trajectory=True,
        target_offered=settings["offered"],
        complete=False,
        new_accept_all=0,
        learning_rate_factors=settings["learning_rate_factors"],
        threshold=lock["threshold"],
        covariance_cap=lock["covariance_cap"],
    )
    text = (
        "# 75 — Adjustable learning rate under covariance protection\n\n"
        f"One history ({lock['history']}) and seed ({lock['seed']}); identical N73 576-update parent and "
        f"{settings['offered']:,} offered continuation images. "
        "Existing N74 policies supply references; no new accept-all run.\n\n"
        "The corrected J predictor, threshold and covariance ceiling are unchanged. For a J-admitted image, "
        "try nominal LR, then 1/4, 1/16 and 1/64 from the same complete parent if covariance fails. "
        "Retain the first passing candidate as one live update, or skip after all four fail. "
        "Every failed trial restores weights, optimizer moments/step, modes and RNG. "
        "Each new image starts its search at nominal LR. Final/development probes do not choose the rate.\n\n"
        "This isolates LR flexibility. J-induced stalls remain possible, and the fixed covariance cap may still bind. "
        "More tiny updates do not by themselves establish useful learning. The nominal-rate-equivalent sum is "
        "accounting, not equivalent optimization, because momentum evolves after each retained step.\n\n"
    )
    images = []
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    if data is None:
        text += "Calibration and source preparation complete; GPU preflight/long-run measurements pending.\n"
    else:
        n = data["n"]
        selected = data["selected"]
        clocks = data["clocks"]
        accepted = int(np.sum(selected > 0))
        summary.update(
            offered=n,
            accepted=accepted,
            rescued_updates=int(np.sum((selected > 0) & (selected < 1))),
            trial_steps=int(data["attempted"].sum()),
            rejected_trial_steps=int(data["attempted"].sum() - accepted),
            nominal_rate_equivalent=float(selected.sum()),
            latest_accuracy=float(data["accuracy"][-1]),
            max_drawdown_pp=float(np.max(np.maximum.accumulate(data["accuracy"]) - data["accuracy"])),
            complete=data["complete"],
            policy_seconds=sum(record["costs"].values()),
            selected_counts={str(f): int(np.sum(selected == f)) for f in settings["learning_rate_factors"]},
        )
        text += (
            f"Latest committed prefix: **{n:,} arrivals; {accepted:,} retained updates; "
            f"{summary['rescued_updates']:,} rescued at smaller rates; "
            f"{summary['latest_accuracy']:.3f}% linear accuracy.**\n\n"
        )
        text += "| LR factor | Actual retained updates |\n|---|---:|\n"
        for factor, count in summary["selected_counts"].items():
            text += f"| {factor} | {count:,} |\n"
        text += (
            f"\nTotal trial steps: {summary['trial_steps']:,}; "
            f"rejected trial steps: {summary['rejected_trial_steps']:,}; "
            f"sum of retained LR factors: {summary['nominal_rate_equivalent']:.3f}.\n"
        )
        prior = ROOT / settings["prior"]
        priorbase = prior / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
        fixed = read(Path(lock["prior_fixed_generation"]) / "commit.json")
        corrected = read(priorbase / "corrected_J/trajectory.json")
        references = {"fixed covariance": fixed, "corrected J": corrected}
        shared = min(start + n, max(map(int, fixed["checkpoints"])))
        summary["shared_offered"] = shared - start
        summary["shared_accuracy"] = {
            "adaptive LR": record["checkpoints"][str(shared)]["metrics"]["linear_accuracy_balanced"],
            **{
                name: r["checkpoints"][str(shared)]["metrics"]["linear_accuracy_balanced"]
                for name, r in references.items()
            },
        }
        text += (
            f"\nCompared at {shared - start:,} shared arrivals: "
            + ", ".join(f"{k} {v:.3f}%" for k, v in summary["shared_accuracy"].items())
            + ".\n"
        )
        colors = {"adaptive LR": "#2166ac", "fixed covariance": "#b85c22", "corrected J": "#6c8a59"}
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
        for ax, key in zip(
            axes.flat,
            ["linear_accuracy_balanced", "linear_ce_balanced", "selector_covariance", "selector_objective"],
            strict=True,
        ):
            for name, r in {"adaptive LR": record, **references}.items():
                ticks = sorted(map(int, r["checkpoints"]))
                ax.plot(
                    np.array(ticks) - start,
                    [r["checkpoints"][str(t)]["metrics"][key] for t in ticks],
                    label=name,
                    color=colors[name],
                    alpha=1 if name == "adaptive LR" else 0.65,
                )
            if key == "selector_covariance":
                ax.axhline(lock["covariance_cap"], color="gray", ls="--", label="Covariance cap")
            ax.set(title=key.replace("_", " "), xlabel="Offered continuation images")
            ax.grid(alpha=0.2)
        axes.flat[0].legend()
        fig.suptitle("N75: adaptive-rate prefix and preserved N74 reference trajectories")
        p = figures / "performance.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        images.append(p)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        offsets = np.array(clocks) - start
        for factor in settings["learning_rate_factors"]:
            axes[0].plot(offsets, [int(np.sum(selected[:k] == factor)) for k in offsets], label=f"LR x {factor:g}")
        axes[0].set(title="Retained updates by selected LR", xlabel="Offered images")
        axes[0].legend()
        axes[1].plot(offsets, [int(np.sum(selected[:k] > 0)) for k in offsets], label="Retained update count")
        axes[1].plot(offsets, [float(selected[:k].sum()) for k in offsets], label="Sum of LR factors")
        axes[1].set(title="Update count versus rate-weighted exposure", xlabel="Offered images")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.2)
        p = figures / "learning-rates.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        images.append(p)
    atomic_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        text + "\n" + "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in images), encoding="utf-8"
    )
    cells = [dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))]
    for p in images:
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=[f"![{p.stem}](attachment:{p.name})"],
                attachments={p.name: {"image/png": base64.b64encode(p.read_bytes()).decode()}},
            )
        )
    nb = dict(
        cells=cells,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
        nbformat=4,
        nbformat_minor=5,
    )
    atomic_json(root / "75_single_image_learning_rate.ipynb", nb)
    atomic_json(ROOT / "notebooks/sums/75_single_image_learning_rate.ipynb", nb)
