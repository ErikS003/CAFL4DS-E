"""Publish audited prediction pilots as reports, figures and an executed reading notebook."""

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
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def figure_cell(path, number):
    relative = path.relative_to(ROOT).as_posix()
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": number,
        "source": [
            "from pathlib import Path\n",
            "from IPython.display import Image, display\n",
            "root = Path.cwd()\n",
            "while not (root / 'cafl4ds').is_dir() and root != root.parent:\n    root = root.parent\n",
            f"display(Image(filename=str(root / '{relative}')))",
        ],
        "outputs": [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "text/plain": ["Saved audited pilot figure"],
                },
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--image-health", type=Path, required=True)
    args = parser.parse_args()
    aout, bout = args.capability.resolve(), args.image_health.resolve()
    for directory in (aout, bout):
        if (
            not read(directory / "verification.json")["passed"]
            or not read(directory / "independent-verification.json")["passed"]
        ):
            raise ValueError("Both pilot audits must pass before publication")
    a, b = read(aout / "summary.json"), read(bout / "summary.json")
    out = ROOT / "outputs/research-summary/20261014-prediction-pilots"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    models = a["outcomes"]["immediate_balanced"]["models"]
    keys = list(models)
    labels = ["Zero", "Train mean", "Count/loss", "+ Parent health", "+ Paid preview", "+ Paid anchor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(labels, [models[k]["rmse"] for k in keys], color=["#a7adb7"] * 3 + ["#337eae", "#c89451", "#c89451"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Held-history RMSE (accuracy pp)", title="Capability: 16-update train minus hold")
    target = a["outcomes"]["immediate_balanced"]["targets"]
    for i, group in enumerate(sorted(set(a["groups"]))):
        mask = np.array(a["groups"]) == group
        axes[1].scatter(np.array(models["parent_health"]["predictions"])[mask], np.array(target)[mask], label=group)
    lim = max(abs(np.array(target)).max(), abs(np.array(models["parent_health"]["predictions"])).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], "--", color="gray")
    axes[1].set(
        xlabel="Forecast capability effect (pp)",
        ylabel="Measured capability effect (pp)",
        title="18 means; six reused history groups",
    )
    axes[1].legend(fontsize=7)
    save(fig, out / "capability")
    primary = b["interventions"]["marginal"]["metrics"]["variance"]
    single = b["interventions"]["single"]["metrics"]["variance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ("image_state", "adamw_derivative"), strict=True):
        predicted = np.array(primary["models"][name]["predictions"])
        measured = np.array(primary["targets"])
        for group in sorted(set(b["interventions"]["marginal"]["groups"])):
            mask = np.array(b["interventions"]["marginal"]["groups"]) == group
            ax.scatter(predicted[mask] * 100, measured[mask] * 100, s=14, alpha=0.7, label=group)
        lower, upper = min(predicted.min(), measured.min()) * 100, max(predicted.max(), measured.max()) * 100
        ax.plot([lower, upper], [lower, upper], "--", color="gray")
        ax.axhline(0, lw=0.5, color="gray")
        ax.axvline(0, lw=0.5, color="gray")
        ax.set(
            xlabel="Predicted relative variance effect (%)",
            ylabel="Measured relative variance effect (%)",
            title="Image + state ridge" if name == "image_state" else "Paid AdamW-aware health derivative",
        )
    axes[1].legend(fontsize=7)
    fig.suptitle("Marginal inclusion of one image: other31 losses and Adam state matched", y=1.03)
    save(fig, out / "image-health")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in (
        (axes[0], primary, "Primary: marginal inclusion, weight1/32"),
        (axes[1], single, "Secondary: train one image alone, weight1"),
    ):
        names = list(data["models"])
        ax.barh(names, [data["models"][n]["skill_vs_zero_percent"] or 0 for n in names], color="#337eae")
        ax.axvline(0, color="gray", lw=0.7)
        ax.invert_yaxis()
        ax.set(xlabel="MSE reduction versus zero effect (%)", title=title)
    save(fig, out / "forecast-skill")
    ac = a["outcomes"]["immediate_balanced"]["primary_candidate_contrasts"]
    capability_text = (
        "# Capability response prediction pilot\n\n"
        f"Primary development screen passed: **{a['development_screen_passed']}**. "
        "Health did not add qualified prediction of balanced few-label fine-tuning capability.\n\n"
        "| Inputs | Held-history MSE (pp squared) | RMSE (pp) |\n|---|---:|---:|\n"
        + "".join(f"| {name} | {m['mse']:.6f} | {m['rmse']:.4f} |\n" for name, m in models.items())
        + f"\nParent-health error change versus count/loss: {-100 * ac['count_loss']['fractional_reduction']:+.2f}%; "
        f"versus training mean: {-100 * ac['mean']['fractional_reduction']:+.2f}%. "
        "Both adjusted improvement intervals cross zero. These are development failures, not proof that health can never work.\n\n"
        "18 seed/panel-averaged points from six reused clip-disjoint histories. Entire histories excluded from fitting and scaling. "
        "Capture sessions remain unknown. No new SSL updates or downstream fits. Targets reconstructed from raw cached predictions; "
        "SVD ridge implementation independently reproduces every fitted forecast. Secondary domain/horizon outcomes remain in summary.json.\n\n"
        "Primary target: spend minus hold after16 updates, accuracy after50-label/100-step copied-encoder fine-tuning, "
        "three support panels and400 balanced day/night city/highway queries. This is not fixed-classifier accuracy or CE.\n\n"
        "![Capability predictions](../../research-summary/20261014-prediction-pilots/capability.png)\n"
    )
    (aout / "report.md").write_text(capability_text, encoding="utf-8")
    lines = [
        "# Individual image impact on health\n",
        "Primary: full batch32 minus the same update omitting one image's loss at fixed normalization.\n",
        "| Predictor | MSE | Skill vs zero | Within-parent rank correlation | Development screen |",
        "|---|---:|---:|---:|---| ",
    ]
    for name, m in primary["models"].items():
        rank = primary["rankings"][name]["mean_within_parent_draw_spearman"]
        passed = b["interventions"]["marginal"]["candidate_screen_passed"].get(name)
        lines.append(
            f"| {name} | {m['mse']:.6g} | {m['skill_vs_zero_percent']:.2f}% | {rank if rank is not None else 'undefined'} | {passed if passed is not None else 'baseline'} |"
        )
    lines += [
        "\n144 marginal contrasts:18 parents, four slots, two augmentation/mask draws, six whole-history validation groups. "
        "Another144 single-image-versus-hold contrasts are separate secondary evidence. Optimization seed5003 only; "
        "adjacent frames share clips. The new interventions use previously inspected histories.\n",
        "The primary outcome is relative change in normalized feature variance on the fixed256-image bank. "
        "It does not identify beneficial capability changes. Rank, covariance and warm-relative drift are secondary.\n",
        "The AdamW derivative pays for exact proposed optimizer displacements and a reference-bank backward pass. "
        "Its success would establish a local health approximation, not a cheap selector. The image/state ridge uses pre-update "
        "losses, image statistics and parent health with held-history fitting.\n",
        f"Cross-draw marginal target correlation: {primary['draw_repeat_correlation']}. "
        "Correlation and MSE skill are distinct; within-parent ranking is reported separately.\n",
        "324 scientific updates,324 exact replay updates,72 all-included/subtraction controls. "
        "Eight updates in the preserved device-failure attempt bring total work to728. "
        "No new downloads, downstream fits or model/pixel caches. Endpoint geometry and all fitted forecasts are independently reconstructed. "
        "Derivative evidence comprises finite-difference tests and exact collector replay; no independent new GPU derivative reconstruction.\n",
        "![Image-to-health forecasts](../../research-summary/20261014-prediction-pilots/image-health.png)\n",
    ]
    (bout / "report.md").write_text("\n".join(lines), encoding="utf-8")
    cells = [
        markdown(
            "# What can CAFL-E predict?\n\nTwo separate development pilots,14 October2026. "
            "Capability prediction and image-to-health prediction have different targets and qualification tests. "
            "All plots use saved audited results; opening this notebook launches no training.\n"
        ),
        markdown(
            "## 1. Predict downstream capability changes\n\n" + capability_text.split("\n\n", 1)[1].split("![")[0]
        ),
        figure_cell(out / "capability.png", 1),
        markdown("## 2. Predict an individual image's effect on health\n\n" + "\n".join(lines[1:-1])),
        figure_cell(out / "image-health.png", 2),
        figure_cell(out / "forecast-skill.png", 3),
        markdown(
            "## Interpretation and boundaries\n\n"
            "A predictable health change need not predict task improvement. A paid local derivative need not yield a useful cheap selector. "
            "The six histories have disjoint clip IDs, but capture-session independence is unknown. "
            "These are bounded development results; independent-source confirmation and actual policy benefit remain separate requirements.\n\n"
            "Protocols: `docs/project-plan/prediction-capability-pilot-frozen.md` and `prediction-image-health-pilot-frozen.md`. "
            "Reports, raw contrasts, folds, cost records and both audits remain in the two output directories. "
            "Rebuild with `python -m scripts.report_prediction_pilots --capability outputs/prediction-capability/20261014-pilot "
            "--image-health outputs/prediction-image-health/20261014-pilot-device-repair`.\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = ROOT / "notebooks/sums/03_prediction_pilots.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    atomic_json(
        out / "provenance.json",
        {
            "sources": {
                str(p): digest(p)
                for d in (aout, bout)
                for p in (d / "summary.json", d / "verification.json", d / "independent-verification.json")
            },
            "figures": {str(p): digest(p) for p in out.glob("*.png")},
            "notebook": str(path),
        },
    )
    print(json.dumps({"notebook": str(path), "figures": str(out)}))


if __name__ == "__main__":
    main()
