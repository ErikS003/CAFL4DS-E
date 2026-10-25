"""Present the fresh no-memory rerun using the established scientific plot layouts."""

from __future__ import annotations

import base64
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.natural_health_rate import path_metrics
from scripts.natural_health_view import ARMS, COLORS, LABELS, SIGNALS, decorate, save, trailing_updates, write
from scripts.run_preaction_stopping import ROOT, digest, read

FIGURES = (
    "capability-readers",
    "health-signals",
    "alarm-inputs",
    "training-and-admission",
    "audit-health",
    "heldout-loss",
)


def diagnostic_figures(out, collected):
    """Keep audit geometry and matched held-out losses outside the live alarm."""
    figures = out / "figures"
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, layout="constrained")
    for i, name in enumerate(SIGNALS):
        for arm in ARMS:
            h = collected["audit_health"][arm]
            axes[i].plot(h["points"], h["values"][name], color=COLORS[arm], label=LABELS[arm])
        axes[i].set_ylabel(name)
        axes[i].axvline(132, color="gray", linestyle=":")
    decorate(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("Audit-bank geometry: diagnostic only, never controls admission")
    save(fig, figures, "audit-health")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, layout="constrained")
    for i, bank in enumerate(("monitor", "audit")):
        for arm in ARMS:
            x = collected["points"]
            y = np.array([collected["losses"][arm][str(p)][bank] for p in x])
            for seed in range(2):
                axes[i].plot(x, y[:, seed], color=COLORS[arm], alpha=0.22)
            axes[i].plot(x, y.mean(1), color=COLORS[arm], label=LABELS[arm])
        axes[i].set_ylabel(bank + " SSL loss")
    decorate(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("Held-out SSL loss: fixed banks, two private mask seeds, no learning")
    save(fig, figures, "heldout-loss")


def present(config, out, collected, verification):
    """Show only full training and quarter recovery in every scientific figure."""
    source = out
    summary = read(source / "readout-summary.json")
    records = read(source / "collection.json")["records"]
    points = summary["points"]
    figures = out / "figures"
    figures.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curves = {
        arm: {
            kind: np.array([collected["cache"][collected["bindings"][arm][str(p)]]["scores"][kind] for p in points])
            for kind in ("knn", "linear")
        }
        for arm in ARMS
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey="row", layout="constrained")
    for column, kind in enumerate(("knn", "linear", "finetune")):
        for arm in ARMS:
            values = (
                curves[arm][kind]
                if kind != "finetune"
                else np.array(summary["rules"][arm]["reader_curves"]).transpose(1, 0, 2)
            )
            for domain in range(3):
                ax = axes[domain, column]
                for draw in range(3):
                    ax.plot(points, values[:, draw, domain], color=COLORS[arm], alpha=0.22, linewidth=0.8)
                ax.plot(points, values.mean(1)[:, domain], color=COLORS[arm], linewidth=2, label=LABELS[arm])
                if column == 0:
                    ax.set_ylabel(("Balanced", "Day", "Night")[domain] + " accuracy (%)")
        axes[0, column].set_title(
            {
                "knn": "kNN · frozen encoder",
                "linear": "Linear probe · frozen encoder",
                "finetune": "Fine-tuning · freshly refitted readout",
            }[kind]
        )
    decorate(axes)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("Two policies, matched 50-label supports and queries · thin = three readers; thick = their mean")
    save(fig, figures, FIGURES[0])

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, layout="constrained")
    names = ("RankMe", "Normalized variance", "Normalized covariance", "CKA distance to 132", "Cosine drift from 132")
    for i, name in enumerate(SIGNALS):
        for arm in ARMS:
            health = collected["health"][arm]
            axes[i].plot(health["points"], health["values"][name], color=COLORS[arm], label=LABELS[arm], linewidth=1.5)
        axes[i].set_ylabel(names[i])
        axes[i].axvline(132, color="#888888", linestyle=":", linewidth=1)
    decorate(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("All five fixed-bank health signals · same 256 images; CKA/cosine begin at the fixed reference")
    save(fig, figures, FIGURES[1])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, layout="constrained")
    for i in range(3):
        for arm in ARMS:
            events = records[arm]["events"]
            axes[i].plot(
                [e["step"] for e in events], [e["values"][i] for e in events], color=COLORS[arm], label=LABELS[arm]
            )
        axes[i].axhline(0.1, color="#666666", linestyle="--", linewidth=1, label="Alarm threshold")
        axes[i].set_ylabel(("Relative rank decline", "Relative variance decline", "CKA distance")[i])
    decorate(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    fig.suptitle("Actual alarm inputs · any input ≥ 0.10 on two consecutive scheduled observations")
    save(fig, figures, FIGURES[2])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, layout="constrained")
    for arm in ARMS:
        record = records[arm]
        x = np.arange(1, len(record["rates"]) + 1)
        axes[0].step(x, np.array(record["rates"]) * 100, where="post", color=COLORS[arm], label=LABELS[arm])
        axes[1].plot(x, np.cumsum(record["accepted"]), color=COLORS[arm])
        for i in range(2):
            smoothed = trailing_updates(record["trace"], i)
            selected = np.isfinite(smoothed)
            axes[i + 2].scatter(x[selected], smoothed[selected], color=COLORS[arm], s=2, alpha=0.65)
    alarms = [e["step"] + 1 for e in records["adaptive"]["events"] if e["alarm"]]
    for step in alarms:
        axes[0].axvline(step, color=COLORS["adaptive"], alpha=0.08, linewidth=0.6)
    for ax, name in zip(
        axes, ("Target admission (%)", "Accepted batches", "SSL loss", "Pre-clip gradient norm"), strict=True
    ):
        ax.set_ylabel(name)
    axes[0].set_yticks([25, 50, 75, 100])
    decorate(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle(
        "Admission and work · loss/gradient = trailing 32 accepted updates; rejected arrivals have no measurement"
    )
    save(fig, figures, FIGURES[3])

    diagnostic_figures(out, collected)

    result = {
        "points": points,
        "arms": ARMS,
        "draws": list(config.draws),
        "probe_reader_curves": curves,
        "source_finetune_curves": {arm: summary["rules"][arm]["reader_curves"] for arm in ARMS},
        "post_hoc": False,
        "controller_qualified": False,
        "independent_source_replications": 1,
    }
    result["reader_path_metrics"] = {}
    for arm in ARMS:
        result["reader_path_metrics"][arm] = {}
        for kind in ("knn", "linear", "finetune"):
            panel = (
                curves[arm][kind]
                if kind != "finetune"
                else np.asarray(summary["rules"][arm]["reader_curves"]).transpose(1, 0, 2)
            )
            result["reader_path_metrics"][arm][kind] = {
                "pooled": path_metrics(panel.mean(1), points),
                "per_draw": [path_metrics(panel[:, i], points) for i in range(3)],
            }
    write(out / "summary.json", result)
    lines = [
        "# Full training versus quarter + recovery",
        "",
        "Only these two freshly executed natural-order policies are displayed. All plots use arriving batches; "
        "separate videos are in archive order, not known capture-session chronology.",
        "",
        "## Capability: three different readouts",
        "",
        "kNN and linear probes keep the SSL encoder frozen. Fine-tuning copies and adapts it using "
        "labels; all three readers are fitted anew in this run. All three use the same "
        "three support draws (50 scene labels each) and 3,663 primary held-out queries. Balanced "
        "accuracy averages classes within day/night, then domains. Thin curves are individual "
        "readers; thick curves are their mean, not independent-source uncertainty.",
        "",
        "kNN uses 20 cosine neighbours weighted by inverse distance. The linear probe is logistic "
        "regression with C=1 and support-only coordinate standardization. No hyperparameter "
        "selection uses query results. All readers were declared before collection and remain offline measurements.",
        "",
        "| Readout | Policy | Start balanced / day / night (%) | Final balanced / day / night (%) |",
        "| --- | --- | --- | --- |",
    ]
    for kind in ("knn", "linear", "finetune"):
        for arm in ARMS:
            curve = curves[arm][kind].mean(1) if kind != "finetune" else np.array(summary["rules"][arm]["curve"])
            left, right = (" / ".join(f"{v:.3f}" for v in curve[i]) for i in (0, -1))
            lines.append(f"| {kind} | {LABELS[arm]} | {left} | {right} |")
    lines += [
        "",
        "## Observed worst-domain drawdowns",
        "",
        "Pooled curves, sparse checkpoint grid; these are observed lower bounds, not continuous guarantees.",
        "",
        "| Reader | Full training | Quarter + recovery |",
        "| --- | ---: | ---: |",
    ]
    for kind in ("knn", "linear", "finetune"):
        a, b = (result["reader_path_metrics"][arm][kind]["pooled"][-1] for arm in ARMS)
        lines.append(f"| {kind} | {a:.3f} pp | {b:.3f} pp |")
    sections = [
        ("capability-readers", "Frozen-feature performance and the freshly refitted fine-tuning readout"),
        ("health-signals", "All five label-free health signals"),
        ("alarm-inputs", "The three signals actually used by the alarm"),
        ("training-and-admission", "Training rate, work and optimization traces"),
    ]
    captions = [
        "Checkpoints are sparse; connecting lines do not establish behavior between measurements.",
        "RankMe measures effective dimension; normalized variance measures spread; covariance "
        "measures redundancy. CKA measures geometric change, while cosine also responds to "
        "coordinate rotation. Movement alone does not prove capability damage. Covariance and "
        "cosine were observed offline and do not trigger this controller.",
        "The rank and variance inputs are fractional declines relative to arrival 132. The "
        "threshold is unchanged. Each policy has its own evolving geometry and warnings; labels "
        "and day/night tags never enter the admission rule.",
        "Each alarm drops subsequent admission to 25%; stages advance to 50/75/100% every 32 "
        "arrivals unless another alarm resets them. Loss and gradient norm are from admitted "
        "training images, not a matched held-out population; different admitted diets limit "
        "their direct quality interpretation. Markers do not fill rejected arrivals.",
    ]
    sections.extend(
        [("audit-health", "Independent audit-bank geometry"), ("heldout-loss", "Held-out SSL loss on both fixed banks")]
    )
    captions.extend(
        [
            "Audit-bank diagnostics at capability checkpoints; never supplied to the alarm. Its reference is fixed at arrival132.",
            "Two private mask seeds on each fixed bank. These losses never train the learner or control admission; lower SSL loss alone does not prove downstream protection.",
        ]
    )
    for (name, title), caption in zip(sections, captions, strict=True):
        lines += ["", f"## {title}", "", f"![{title}](figures/{name}.png)", "", caption]
    lines += [
        "",
        "## Provenance and limits",
        "",
        f"Fresh live collection and its exact second execution each make {collected['collection_work']['ssl_completed']:,} "
        f"SSL updates. There are {verification['unique_feature_states']} unique frozen-feature "
        "states. Every new probe prediction is reproduced by a verification refit. Original "
        "scientific artifacts remain unchanged. The original policy, seed and source stream "
        "were reproduced with fresh fits, live alarms and no training memory. One stream and one SSL seed "
        "provide descriptive evidence, not controller qualification.",
        "",
        "[Original experiment](../../natural-health-rate/20261016-development/report.md) · "
        "[Readout plan](../../../docs/project-plan/natural-health-no-replay-rerun-frozen.md) · "
        "[Verification](verification.json) · [Summary](summary.json)",
    ]
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    notebook = ROOT / config.notebook
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Full training versus quarter + recovery\n\n",
                "Five health signals, frozen-encoder kNN and linear probes, and the freshly refitted fine-tuning reader.\n\n",
                "[Full methods, scores and provenance]"
                "(../../outputs/natural-health-no-replay-rerun/20261018/report.md). "
                "This physical rerun uses one ordered BDD100K stream and one SSL seed. "
                "Thin capability curves show the three support draws; thick curves show their mean.\n",
            ],
        }
    ]
    for number, ((name, title), caption) in enumerate(zip(sections, captions, strict=True), 1):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n\n{caption}\n"]})
        relative = f"../../outputs/natural-health-no-replay-rerun/20261018/figures/{name}.png"
        cells.append(
            {
                "cell_type": "code",
                "execution_count": number,
                "metadata": {},
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    "root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'cafl4ds/configs').is_dir())\n",
                    f"display(Image(filename=str(root / {relative[6:]!r})))\n",
                ],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": base64.b64encode((figures / f"{name}.png").read_bytes()).decode(),
                            "text/plain": [f"{title}\n"],
                        },
                    }
                ],
            }
        )
    payload = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }
    assert not notebook.exists()
    notebook.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
    hashes = {
        str(p): digest(p)
        for p in [*out.glob("*.json"), *out.glob("*.md"), *figures.glob("*"), *out.glob("features/*.npz"), notebook]
    }
    write(out / "complete.json", {"passed": True, "hashes": hashes})
