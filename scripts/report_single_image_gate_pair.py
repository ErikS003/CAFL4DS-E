"""Report a single pair at shared measured clocks without across-run intervals."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint
from cafl4ds.single_image_forecast import HEALTH
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gate_original", "full")
COLORS = dict(gate_original="#2166ac", full="#1b7837")


def report(root):
    lock = read(root / "input-lock.json")
    start, target = lock["settings"]["start"], lock["settings"]["offered"]
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    common = directory / "common/readout.json"
    data = {}
    for arm in ARMS:
        branch = directory / arm
        record_path = branch / "trajectory.json"
        if record_path.exists():
            record = read(record_path)
            with np.load(branch / "trace.npz") as trace:
                accepted = trace["accepted"].copy()
            complete = True
        else:
            checkpoint = load_checkpoint(branch)
            if checkpoint is None:
                if not common.exists():
                    continue
                record = dict(checkpoints={str(start): read(common)})
                accepted = np.zeros(target, dtype=bool)
            else:
                generation, record = checkpoint
                with np.load(generation / "trace.npz") as trace:
                    accepted = trace["accepted"].copy()
            complete = False
        clocks = sorted(map(int, record["checkpoints"]))
        data[arm] = dict(
            record=record,
            clocks=clocks,
            accepted=accepted,
            complete=complete,
            updates=[int(accepted[: clock - start].sum()) for clock in clocks],
        )
    shared = sorted(set.intersection(*(set(row["clocks"]) for row in data.values()))) if len(data) == 2 else []
    comparison_clock = shared[-1] if shared else None
    summary = dict(
        single_pair=True,
        history=lock["history"],
        seed=lock["seed"],
        offered_target_each=target,
        common_offered=None if comparison_clock is None else comparison_clock - start,
        collection_complete=len(data) == 2 and all(row["complete"] for row in data.values()),
        uncertainty_intervals=None,
        arms={},
    )
    text = (
        "# 73 — One longer original-gate/full pair\n\n"
        f"Each policy is offered {target:,} temporal frames from 600 distinct cached clips. "
        f"History {lock['history']}, seed {lock['seed']}; shared 576-update parent. "
        "The first 19,200 continuation frames repeat the prior stream for this pair; "
        "the remaining clips extend that stream without repeats within the trajectory.\n\n"
        "This is one exploratory pair. Cached data and evaluation panels were used previously. "
        "No across-run confidence interval is available. The frozen gate extrapolates its arrival clock.\n\n"
    )
    if comparison_clock is not None:
        text += f"Compared at the latest shared checkpoint: **{comparison_clock - start:,} offered images each**.\n\n"
        text += (
            "| Policy | Latest offered | Accepted at common checkpoint | Accuracy at common checkpoint | Finished |\n"
        )
        text += "|---|---:|---:|---:|---|\n"
        for arm, row in data.items():
            metrics = row["record"]["checkpoints"][str(comparison_clock)]["metrics"]
            count = int(row["accepted"][: comparison_clock - start].sum())
            summary["arms"][arm] = dict(
                latest_offered=row["clocks"][-1] - start,
                common_accepted=count,
                common_metrics=metrics,
                complete=row["complete"],
                policy_seconds=sum(row["record"].get("costs", {}).values()),
                research_seconds=row["record"].get("research_seconds", 0.0),
            )
            text += (
                f"| {arm} | {row['clocks'][-1] - start:,} | {count:,} | "
                f"{metrics['linear_accuracy_balanced']:.3f}% | {row['complete']} |\n"
            )
    else:
        text += "Preparing the shared initial measurement and bounded GPU verification.\n"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    images = []
    if data:
        groups = [
            ("readers", ["linear_accuracy_balanced", "linear_ce_balanced", "knn_accuracy_balanced", "knn_ce_balanced"]),
            ("evaluation-health", ["evaluation_" + metric for metric in [*HEALTH, "objective"]]),
            ("selector-health", ["selector_" + metric for metric in [*HEALTH, "objective"]]),
        ]
        for name, metrics in groups:
            for updates in (False, True):
                fig, axes = plt.subplots(2, len(metrics) // 2, figsize=(13, 7), constrained_layout=True)
                for ax, metric in zip(axes.flat, metrics, strict=True):
                    for arm, row in data.items():
                        x = row["updates"] if updates else np.array(row["clocks"]) - start
                        y = [row["record"]["checkpoints"][str(clock)]["metrics"][metric] for clock in row["clocks"]]
                        ax.plot(x, y, label=arm.replace("_", " "), color=COLORS[arm])
                    ax.set(title=metric.replace("_", " "), xlabel="Accepted updates" if updates else "Offered images")
                    ax.grid(alpha=0.2)
                axes.flat[0].legend(fontsize=8)
                fig.suptitle("One paired trajectory; no across-run uncertainty estimate")
                path = figures / f"{name}-{'updates' if updates else 'arrivals'}.png"
                fig.savefig(path, dpi=115)
                plt.close(fig)
                images.append(path)
    atomic_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        text + "\n" + "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in images), encoding="utf-8"
    )
    cells = [dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))]
    for path in images:
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=[f"![{path.stem}](attachment:{path.name})"],
                attachments={path.name: {"image/png": base64.b64encode(path.read_bytes()).decode()}},
            )
        )
    notebook = dict(cells=cells, metadata={}, nbformat=4, nbformat_minor=5)
    atomic_json(root / "73_single_image_gate_pair.ipynb", notebook)
    atomic_json(ROOT / "notebooks/sums/73_single_image_gate_pair.ipynb", notebook)
