"""Build Notebook 91 from only the six requested saved Notebook 90 metrics."""

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.single_image_extended_gate import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/single-image-heuristic/20261025-online-admission"
OFFLINE = ROOT / "outputs/single-image-heuristic/20261024-no-rollback"
BRANCH = Path("trajectories/fresh_00/seed_5003/heuristic")
DESTINATION = ROOT / "notebooks/selected/91_single_image_selected_metrics.ipynb"
METRICS = [
    ("linear_accuracy_balanced", "Balanced linear accuracy (%)"),
    ("selector_rank", "RankMe"),
    ("selector_variance", "Variance"),
    ("selector_covariance", "Covariance"),
    ("selector_bank_reconstruction", "Reconstruction (fixed bank)"),
]
COLORS = {
    "Offline heuristic (A)": "#2466b0",
    "Online heuristic (B)": "#d34e2c",
    "Full training": "#458b65",
}
METHODS = """## Methods

All three methods continue image-reconstruction training from the same warm parent on the same ordered image stream (seed 5003). A and B also share the augmentation and acceptance-RNG seeds. Each accepted image produces one model update.

### A: Offline heuristic

A reuses N84's saved run. "Offline" means that its admission predictor was fitted beforehand and remains frozen; the image model continues learning. The predictor estimates how an update would change J: fixed-bank reconstruction relative to the warm parent, plus penalties for reduced RankMe and variance (weight 0.25 each). A lower J is preferred. Predictions below the calibrated negative threshold receive a 50% update probability; other negative predictions receive 25%, and nonnegative predictions receive 10%. Unavailable or heavily clipped predictions fall back to 25%. This preserves some learning opportunities even when predictions are unfavorable.

Fixed-bank health is checked every 128 arrivals. The normal learning rate is one quarter of nominal; persistent loss of rank/variance, or jointly elevated covariance and reconstruction loss, switches it to one sixteenth. If fewer than eight updates occur in the preceding 256 arrivals, a short burst supplies 16 consecutive updates at the smaller rate. Development checks can refresh the backup and health reference, but task-based rollback is disabled. Numerical failures still trigger restoration and bounded recovery.

### B: Online heuristic

B starts from A's rules and adds a predictor that learns during its own trajectory. This ridge regression predicts the immediate change in balanced development soft error, a smooth task-error measure: negative change means predicted improvement. Its inputs are available before the decision and include image/state features, A's J prediction, the proposed learning rate, caution mode, and the preceding development error. After each accepted update, the observed error change supplies its training target. Rejected images supply no outcome labels.

The first 128 accepted outcome samples initialize feature scaling and the model; the first arrival is excluded from these samples. Thereafter, ordinary admission probability is multiplied by 1.2 for predicted improvement or 0.8 for predicted harm, then clipped to [0.05, 0.95]. Zero or nonfinite predictions leave it unchanged. Recent outcomes receive more weight, with a half-life of 256 accepted samples. B retains A's health/rate rules, bursts, numerical recovery, and disabled task rollback. Development labels inform this adjustment; final evaluation labels are used only for reporting.

### Full training

The saved full-exposure reference updates on every offered image and therefore uses a larger training budget. The plotted comparison is one paired seed and is not matched for update count.
"""


def image_cell(fig, name, title):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "cell_type": "markdown",
        "id": name,
        "metadata": {},
        "source": [f"## {title}\n", f"![{title}](attachment:{name}.png)"],
        "attachments": {f"{name}.png": {"image/png": encoded}},
    }


def main():
    lock = json.loads((SOURCE / "input-lock.json").read_text())
    common = json.loads((SOURCE / BRANCH.parent / "common/readout.json").read_text())
    records, traces, sources = {}, {}, {}
    for name, root in [("Offline heuristic (A)", OFFLINE), ("Online heuristic (B)", SOURCE)]:
        saved = load_checkpoint(root / BRANCH)
        if saved is None:
            raise FileNotFoundError(f"No committed checkpoint for {name}")
        generation, records[name] = saved
        sources[name] = str((generation / "commit.json").relative_to(ROOT))
        with np.load(generation / "trace.npz") as trace:
            traces[name] = {key: trace[key].copy() for key in ("accepted", "training")}
    full_path = Path(lock["inherited_full_reference"])
    records["Full training"] = json.loads(full_path.read_text())
    sources["Full training"] = str(full_path.relative_to(ROOT))
    with np.load(full_path.parent / "trace.npz") as trace:
        traces["Full training"] = {key: trace[key].copy() for key in ("accepted", "training")}

    start = lock["settings"]["start"]
    horizon = max(384, records["Online heuristic (B)"].get("done", 0))
    introduction = (
        "# Notebook 91: selected metrics from Notebook 90\n\n"
        "Saved offline heuristic (A), online heuristic (B), and full-training comparison "
        f"from N90, through {horizon:,} offered continuation images. "
        "Dotted lines mark the warm parent. Full training uses more updates; "
        "this is a one-seed comparison. Reconstruction is fixed-bank loss; "
        "trained-image reconstruction is the loss on accepted training images.\n\n"
        + METHODS
    )
    cells = [{"cell_type": "markdown", "id": "introduction", "metadata": {},
              "source": introduction.splitlines(True)}]
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
        for name, record in records.items():
            pairs = sorted(
                (int(step) - start, checkpoint["metrics"][metric])
                for step, checkpoint in record["checkpoints"].items()
                if 0 <= int(step) - start <= horizon and metric in checkpoint["metrics"]
            )
            if not pairs:
                raise ValueError(f"Missing requested metric {metric} for {name}")
            ax.plot(*np.array(pairs).T, label=name, color=COLORS[name], linewidth=1.5)
        if metric in common["metrics"]:
            ax.axhline(common["metrics"][metric], color="gray", ls=":", lw=0.8)
        ax.set(xlabel="Offered continuation images", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        cells.append(image_cell(fig, metric, title))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for ax, (name, trace) in zip(axes, traces.items()):
        record = records[name]
        n = min(horizon, record.get("done", record.get("offered", 0)))
        accepted = trace["accepted"][:n]
        x = np.arange(1, n + 1)[accepted]
        ax.scatter(x, trace["training"][:n, 0][accepted], s=2, alpha=0.35, color=COLORS[name])
        ax.set(title=name, xlabel="Offered continuation images",
               ylabel="Trained-image reconstruction loss")
        ax.grid(alpha=0.2)
    cells.append(image_cell(fig, "trained_image_reconstruction", "Trained image reconstruction"))
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"source_notebook": "90_single_image_online_admission.ipynb",
                     "source_records": sources, "offered_horizon": horizon},
        "cells": cells,
    }
    DESTINATION.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
