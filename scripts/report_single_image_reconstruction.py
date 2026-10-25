"""Add reconstruction plots to live N75 outputs without editing frozen training code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TAG = "n75_reconstruction_reporting"
MARKER = "<!-- n75-reconstruction-reporting -->"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def interval_means(values, mask, count, stride=384):
    """Empty learning intervals have no observed training loss, never zero loss."""
    positions, means, counts = [], [], []
    for first in range(0, count, stride):
        last = min(count, first + stride)
        take = mask[first:last]
        positions.append(last)
        counts.append(int(take.sum()))
        means.append(float(values[first:last][take].mean()) if take.any() else None)
    return positions, means, counts


def decorate(root):
    lock = read(root / "input-lock.json")
    notebook = root / "75_single_image_learning_rate.ipynb"
    if not notebook.exists():
        return False
    source_mtime = notebook.stat().st_mtime_ns
    nb = read(notebook)
    first = "".join(nb["cells"][0]["source"])
    matched = re.search(r"Latest committed prefix:\s*\*\*([\d,]+) arrivals", first)
    if matched is None:
        return False
    count = int(matched[1].replace(",", ""))
    start = lock["settings"]["start"]
    branch = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / lock["settings"]["arm"]
    if (branch / "trajectory.json").exists():
        record, trace_path = read(branch / "trajectory.json"), branch / "trace.npz"
    else:
        checkpoint = load_checkpoint(branch)
        if checkpoint is None:
            return False
        generation, record = checkpoint
        trace_path = generation / "trace.npz"
    clocks = sorted(t for t in map(int, record["checkpoints"]) if t <= start + count)
    if not clocks or clocks[-1] != start + count:
        return False
    with np.load(trace_path) as trace:
        accepted = trace["accepted"][:count].copy()
        training = trace["training"][:count, 0].copy()
        tried = trace["attempted"][:count, 0].copy()
        nominal_losses = trace["trial_training"][:count, 0, 0].copy()
    reference = read(Path(lock["prior_fixed_generation"]) / "commit.json")
    prior = ROOT / lock["settings"]["prior"]
    old_J = read(prior / "trajectories" / lock["history"] / f"seed_{lock['seed']}" / "corrected_J/trajectory.json")
    policies = {"adaptive LR": record, "fixed covariance": reference, "corrected J": old_J}
    colors = {"adaptive LR": "#2166ac", "fixed covariance": "#b85c22", "corrected J": "#6c8a59"}
    fixed_values = {}
    # Audit plotted adaptive checkpoint means against the saved per-image fixed-mask assays.
    for t in clocks:
        raw_path = branch.parent / "common/raw.npz" if t == start else branch / f"checkpoint_{t}/raw.npz"
        with np.load(raw_path) as raw:
            for bank in ("selector", "evaluation"):
                np.testing.assert_allclose(
                    raw[f"{bank}_reconstruction"].mean(),
                    record["checkpoints"][str(t)]["metrics"][f"{bank}_bank_reconstruction"],
                    rtol=1e-12,
                    atol=1e-12,
                )
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for ax, bank, title in zip(
        axes[:2],
        ("selector", "evaluation"),
        ("Fixed selector-bank reconstruction", "Held-out fixed-bank reconstruction"),
        strict=True,
    ):
        key = f"{bank}_bank_reconstruction"
        for name, r in policies.items():
            ticks = sorted(t for t in map(int, r["checkpoints"]) if t <= start + count)
            values = [float(r["checkpoints"][str(t)]["metrics"][key]) for t in ticks]
            ax.plot(
                np.array(ticks) - start,
                values,
                label=name,
                color=colors[name],
                marker=".",
                markersize=3,
                alpha=1 if name == "adaptive LR" else 0.65,
            )
            fixed_values.setdefault(name, {})[bank] = dict(offered=(np.array(ticks) - start).tolist(), loss=values)
        parent = record["checkpoints"][str(start)]["metrics"][key]
        ax.axhline(parent, color="gray", ls="--", lw=1, label="Shared parent")
        ax.set(title=title, ylabel="Masked reconstruction loss", xlabel="Offered continuation images")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
    positions, means, counts = interval_means(training, accepted, count)
    _, trial_means, trial_counts = interval_means(nominal_losses, tried, count)
    axes[2].scatter(
        np.flatnonzero(accepted) + 1,
        training[accepted],
        s=9,
        color=colors["adaptive LR"],
        alpha=0.2,
        label="Individual retained-update losses",
    )
    axes[2].plot(
        positions,
        [np.nan if v is None else v for v in means],
        marker="o",
        markersize=3,
        color=colors["adaptive LR"],
        label="Retained updates: mean per 384 arrivals",
    )
    axes[2].plot(
        positions,
        [np.nan if v is None else v for v in trial_means],
        ls="--",
        color="#a05b97",
        label="All J-admitted candidates: mean per 384 arrivals",
    )
    axes[2].set(
        title="Training-image reconstruction loss before the update (selected-image dependent)",
        ylabel="Masked training loss",
        xlabel="Offered continuation images",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"N75 reconstruction over the first {count:,} offered images; lower is better")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    picture = figures / "reconstruction.png"
    fig.savefig(picture, dpi=135)
    plt.close(fig)
    explanation = (
        "### Reconstruction loss over time\n\n"
        "The first two panels use the same fixed selector and held-out image banks, pixels and masks at "
        "each checkpoint. They measure reconstruction on comparable images as the model changes. "
        "Reference policies are clipped to this notebook's current arrival horizon.\n\n"
        "The bottom panel shows the actual masked reconstruction loss on each retained training image "
        "before its update, plus means within 384 offered arrivals. The dashed line includes all "
        "J-admitted candidate images, including those ultimately rejected by covariance protection. "
        "Each image contributes once; smaller-rate retries are not counted again. These training "
        "curves depend on which images arrive and are selected. Empty intervals are gaps, not zero loss.\n\n"
        "This plot is reconstructed from existing saved measurements; it adds no training or probe evaluations.\n"
    )
    additions = [
        dict(cell_type="markdown", metadata={TAG: True}, source=explanation.splitlines(keepends=True)),
        dict(
            cell_type="markdown",
            metadata={TAG: True},
            source=["![Reconstruction loss](attachment:reconstruction.png)"],
            attachments={"reconstruction.png": {"image/png": base64.b64encode(picture.read_bytes()).decode()}},
        ),
    ]
    nb["cells"] = [cell for cell in nb["cells"] if not cell.get("metadata", {}).get(TAG)] + additions
    nb.setdefault("metadata", {})[TAG] = dict(
        offered=count, checkpoint_assays_verified=len(clocks), reporting_only=True
    )
    # Discard this rendering if the training reporter published a newer prefix.
    if notebook.stat().st_mtime_ns != source_mtime:
        return False
    atomic_json(notebook, nb)
    atomic_json(ROOT / "notebooks/sums/75_single_image_learning_rate.ipynb", nb)
    original_text = (root / "report.md").read_text(encoding="utf-8").split(MARKER)[0].rstrip()
    (root / "report.md").write_text(
        original_text + "\n\n" + MARKER + "\n" + explanation + "\n![Reconstruction loss](figures/reconstruction.png)\n",
        encoding="utf-8",
    )
    atomic_json(
        root / "reconstruction-summary.json",
        dict(
            offered=count,
            fixed_banks=fixed_values,
            training_intervals=dict(
                offered_end=positions,
                accepted_mean_loss=means,
                accepted_counts=counts,
                candidate_mean_loss=trial_means,
                candidate_counts=trial_counts,
            ),
            checkpoint_assays_verified=len(clocks),
            frozen_training_reporter_unchanged=hashlib.sha256(
                (ROOT / "scripts/report_single_image_learning_rate.py").read_bytes()
            ).hexdigest()
            == lock["hashes"][str((ROOT / "scripts/report_single_image_learning_rate.py").resolve())],
        ),
    )
    atomic_json(root / "reconstruction-watch-status.json", dict(stage="updated", offered=count, reporting_only=True))
    print(json.dumps(dict(stage="reconstruction_updated", offered=count)), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    last_mtime, quiet_since = None, None
    while True:
        notebook = root / "75_single_image_learning_rate.ipynb"
        try:
            current_mtime = notebook.stat().st_mtime_ns if notebook.exists() else None
            if current_mtime != last_mtime:
                if decorate(root):
                    last_mtime = notebook.stat().st_mtime_ns
                quiet_since = None
            if not args.watch:
                return
            status = read(root / "status.json")
            if not status.get("running", False) and status.get("stage") in ("complete", "stopped_by_user", "failed"):
                quiet_since = time.monotonic() if quiet_since is None else quiet_since
                if time.monotonic() - quiet_since >= 6:
                    decorate(root)
                    atomic_json(root / "reconstruction-watch-status.json", dict(stage="finished", reporting_only=True))
                    return
            else:
                quiet_since = None
        except (FileNotFoundError, json.JSONDecodeError):
            # A committed generation can be retired as the worker publishes a newer checkpoint.
            if not args.watch:
                raise
        time.sleep(2)


if __name__ == "__main__":
    main()
