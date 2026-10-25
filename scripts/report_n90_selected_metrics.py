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

B has two distinct learners. The **image model** is the encoder and reconstruction decoder: an accepted image produces one self-supervised AdamW update that minimizes masked-image reconstruction error. The **online task predictor** is a separate, small linear regression model. It learns which proposed image updates are likely to help the development classification task, and uses that prediction to adjust the chance of accepting an image. Its regression objective does not replace the image model's reconstruction loss or backpropagate into the encoder.

#### What is it trained on?

One training example is a pair of **pre-update features** and the **task-error change observed after an accepted image update**. It is not a regression trained directly on raw images or their reconstruction targets. The runner measures the current encoder using linear classifiers fitted on labeled development support images and evaluated on separate development query images. The same support/query identities are used across measurements, and the classifiers are refitted after accepted image updates.

The task score S is a smooth classification-error measure averaged over the balanced development query panel and reader support draws. For each query image, take the linear classifier's probability of its true class, p_true, and compute `sigmoid((0.5 - p_true) / 0.05)`. Confident correct predictions contribute near zero; incorrect predictions contribute near one. S changes smoothly near the decision boundary, even when ordinary accuracy has not changed. It is neither cross-entropy nor reconstruction loss.

The regression label is `y = S_after - S_before`, measured across that one accepted update. A negative label means immediate development-task improvement; a positive label means deterioration. The first arrival is excluded from predictor training. Rejected images and numerically failed updates provide no observed training label: their hypothetical effects are not treated as zero. Thus the predictor learns from the updates B actually accepts, rather than from outcomes for every offered image.

This makes the controller **task-informed**: development class labels supply both its outcome measurements and a task-gradient feature. The image update itself remains self-supervised. The separate final evaluation labels are used only for reporting, and never train this predictor or choose admission decisions.

#### What does it see before making a prediction?

The predictor receives 19 numeric inputs, all available before executing the candidate update:

| Input group | Count | Contents |
| --- | ---: | --- |
| Current representation health | 4 | RankMe, variance, covariance, fixed-bank reconstruction |
| Recent health changes | 4 | Preceding-arrival changes in those same four measurements |
| Candidate update information | 5 | Candidate training-view reconstruction loss, gradient norm, gradient/momentum cosine, proposed AdamW displacement norm, and development cross-entropy-gradient/displacement cosine |
| Image novelty | 2 | Distance from the warm feature centroid and nearest-recent-offered-image cosine distance |
| Controller and task context | 4 | Frozen J forecast, candidate learning-rate factor, caution-mode indicator, and preceding development soft error |

The gradient/displacement features describe a proposed update using the current model and optimizer state. Computing these features does not apply the image update. The task-gradient cosine measures the relationship between the proposed displacement and the development classification-loss gradient; it is a feature, not the predictor's training label.

#### What objective does the predictor optimize?

It uses **exponentially discounted ridge regression**: choose linear coefficients that minimize squared error between predicted and observed task-error changes, with a penalty on coefficient size. In compact form, the objective is:

```text
sum over past labeled updates i:
    gamma^(t-i) * (beta dot phi(x_i) - y_i / target_scale)^2
plus 10 * sum(feature_coefficients^2) + 0.1 * intercept^2
```

Here phi contains the standardized/clipped inputs plus an intercept, and `gamma = 2^(-1/256)`. An observation therefore has half its original weight after 256 later labeled accepted updates. This discount lets the predictor adapt as the image model changes; it is not retrained from scratch after each image.

The first 128 labeled accepted updates initialize input means, input standard deviations, target scale, and the initial regression. Input scaling then stays fixed, standardized coordinates are clipped to [-5, 5], and coefficients continue updating from subsequent accepted outcomes. The model has 19 feature coefficients and one intercept. Ridge regularization limits unstable coefficients when features are correlated and individual task changes are small.

#### What does it predict, and how does that affect training?

Before deciding whether to train on an arriving image, it predicts a **signed scalar change in development soft error if that image receives the proposed update**. Predictions are converted back to the original soft-error units. This is an estimate of immediate update benefit, not a probability of improvement, a prediction of final accuracy, or a prediction of J. The trace also stores zero-change, recent-mean, and initialization-frozen regression predictions as diagnostic baselines; only the continually updated regression controls B.

Once initialized, B takes A's ordinary admission probability p and applies:

- Predicted change below zero: `p_new = clip(1.2 * p, 0.05, 0.95)`.
- Predicted change above zero: `p_new = clip(0.8 * p, 0.05, 0.95)`.
- Zero/nonfinite prediction, or predictor not initialized: leave p unchanged.

For example, A's 25% probability becomes 30% for predicted benefit or 20% for predicted harm. The multiplier uses the prediction's sign, not its magnitude. The seeded acceptance draw then determines whether the update happens. This adjustment applies only to ordinary decisions in normal/caution mode; forced bursts and numerical recovery keep A's rules. N91's predictor changes admission probability, while the existing health controller still determines learning rate.

The order is **compute features and predict, make the admission decision, execute an accepted image update, measure its actual development effect, then update the regression**. That outcome can influence future decisions, but cannot influence the prediction already made for the same image. B retains the fixed bank, J weights, health/rate rules, bursts, numerical recovery, and disabled task-based rollback.

#### What is the limitation of this objective?

The target covers one immediate update and one repeatedly observed development panel. Predicting it well does not establish that the chosen update improves later evaluation accuracy. Rejected-image outcomes remain unknown, and the model sees the selection bias created by its own policy. N91 therefore reports a bounded, task-informed admission experiment. The separate N92 experiment extends feedback to complete windows and explores rates; those mechanisms are not part of N91's B results.

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
