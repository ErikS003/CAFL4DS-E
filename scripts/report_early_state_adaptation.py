"""Scientific figures and an executed saved-result notebook for study39."""

from __future__ import annotations

import base64
import contextlib
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from scripts.prepare_early_state_adaptation import CONFIG, ROOT, digest, read, write

COLORS = {"current": "#2878b5", "balanced": "#dc6b27"}


def figures(study, summary):
    out = ROOT / study.output
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    result = []

    def save(fig, name):
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        result.append(path)

    ft = summary["readers"]["finetune"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for component, ax in enumerate(axes):
        for i, arm in enumerate(("current", "balanced")):
            r = ft["earlier_screens"][arm]
            values = np.asarray(r["case_means_pp"])[..., component].ravel()
            ax.scatter(np.full(12, i) + np.linspace(-0.12, 0.12, 12), values, s=22, alpha=0.6, color=COLORS[arm])
            lo, hi = r["family_two_conditional_bounds_pp"][component]
            mean = r["mean_pp"][component]
            ax.plot([i + 0.22] * 2, [lo, hi], color="black", lw=2)
            ax.scatter([i + 0.22], [mean], marker="D", color="black", s=35, zorder=4)
        ax.axhline(0, color="gray", lw=1)
        ax.axhline(-1, color="gray", lw=0.7, ls=":")
        ax.set(xticks=[0, 1], xticklabels=["Current", "Half replay"], title=("Balanced", "Day", "Night")[component])
    axes[0].set_ylabel("Earlier learner: change versus freeze (pp)")
    fig.suptitle("12 source/future cases; diamonds and bands condition on these sources and futures")
    fig.tight_layout()
    save(fig, "01-earlier-effects")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    matrices = []
    for age in (0, 1):
        for arm in ("current", "balanced"):
            v = (
                np.asarray(ft["earlier_screens"][arm]["case_means_pp"])
                if age == 0
                else np.asarray(ft["late_effects_pp"][arm]).mean(2)
            )
            matrices.append(v[..., 0])
    limit = max(1, max(np.abs(v).max() for v in matrices))
    for ax, matrix, label in zip(
        axes.ravel(),
        matrices,
        ("512 updates: current", "512 updates: half replay", "4,096 updates: current", "4,096 updates: half replay"),
    ):
        im = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
        for b in range(3):
            for f in range(4):
                ax.text(
                    f,
                    b,
                    f"{matrix[b, f]:+.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1},
                )
        ax.set(title=label, xticks=range(4), xticklabels=[1, 2, 3, 4], yticks=range(3), yticklabels=[1, 2, 3])
    for ax in axes[-1]:
        ax.set_xlabel("Fresh continuation block")
    for ax in axes[:, 0]:
        ax.set_ylabel("Historical source")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Balanced effect versus freeze (pp)", shrink=0.8)
    fig.suptitle("Every source/future mean, with 12 matched support draws")
    save(fig, "02-source-future-effects")

    inter = np.asarray(summary["intermediate_effects_pp"])[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for c, ax in enumerate(axes):
        for age in (0, 1):
            for ai, arm in enumerate(("current", "balanced")):
                terminal = (
                    np.asarray(ft["earlier_screens"][arm]["mean_pp"])
                    if age == 0
                    else np.asarray(ft["late_effects_pp"][arm]).mean((0, 1, 2))
                )
                middle = inter[age, ai].mean((0, 1, 2))
                ax.plot(
                    [0, 128, 512],
                    [0, middle[c], terminal[c]],
                    marker="o",
                    color=COLORS[arm],
                    ls="-" if age == 0 else "--",
                    label=f"{('Early', 'Late')[age]} {arm}",
                )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=("Balanced", "Day", "Night")[c], xlabel="Continuation updates", ylabel="Change versus parent (pp)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Same 12 support draws at both landmarks; lines connect measured checkpoints")
    fig.tight_layout()
    save(fig, "03-matched-trajectories")

    rows = [r for r in summary["health_descriptive"] if r["bank"] == "monitor" and r["step"] == 512]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels = [
        "Rank drop",
        "Variance drop",
        "Covariance rise",
        "CKA drift",
        "Cosine drift",
        "Mean Adam encoder displacement",
    ]
    for i, ax in enumerate(axes.ravel()):
        for age in (512, 4096):
            for arm in ("current", "balanced"):
                selected = [r for r in rows if r["age"] == age and r["arm"] == arm]
                x = [r["health"][i] if i < 5 else r["mean_encoder_step_l2"] for r in selected]
                ax.scatter(
                    x,
                    [r["capability_change_pp"][0] for r in selected],
                    s=28,
                    color=COLORS[arm],
                    marker="o" if age == 512 else "x",
                    alpha=0.75,
                    label=f"{age}: {arm}",
                )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(xlabel=labels[i], ylabel="Balanced change versus parent (pp)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Terminal monitor geometry and update size: descriptive associations, shared learners")
    fig.tight_layout()
    save(fig, "04-health-and-capability")

    plan = read(out / "plan.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for age in (512, 4096):
        for arm in ("current", "balanced"):
            traces = [
                read(out / "branches" / p["id"] / arm / "record.json")["trace"]
                for p in plan["parents"]
                if p["age"] == age
            ]
            displacement = np.mean([[v["encoder_step_l2"] for v in t] for t in traces], axis=0)
            loss = np.mean([[v["response"][0] for v in t] for t in traces], axis=0)
            for ax, values in zip(axes, (displacement, loss)):
                ax.plot(
                    np.arange(1, 513), values, color=COLORS[arm], ls="-" if age == 512 else "--", label=f"{age}: {arm}"
                )
    axes[0].set(ylabel="Actual encoder parameter step L2", yscale="log")
    axes[1].set_ylabel("SSL loss")
    for ax in axes:
        ax.set_xlabel("Continuation update")
    axes[1].legend(fontsize=8)
    fig.suptitle("Mean across the 12 fixed source/future cases")
    fig.tight_layout()
    save(fig, "05-update-dynamics")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for c, ax in enumerate(axes):
        for arm in ("current", "balanced"):
            values = np.asarray(ft["earlier_screens"][arm]["support_means_pp"])
            ax.plot(range(1, 13), values[:, c], marker="o", color=COLORS[arm], label=arm)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(
            title=("Balanced", "Day", "Night")[c],
            xlabel="Matched support draw",
            ylabel="Earlier-state gain versus freeze (pp)",
        )
    axes[-1].legend()
    fig.suptitle("Support sensitivity on fixed sources, futures and queries")
    fig.tight_layout()
    save(fig, "06-support-sensitivity")
    return result


def report():
    study = OmegaConf.load(CONFIG)
    out = ROOT / study.output
    verification = read(out / "verification.json")
    assert verification["passed"] and verification["summary_sha256"] == digest(out / "summary.json")
    summary = read(out / "summary.json")
    paths = figures(study, summary)
    screens = summary["readers"]["finetune"]["earlier_screens"]
    text = "# Earlier learners on fresh continuation clips\n\n"
    text += (
        "The predeclared development screen **"
        + ("passes" if summary["prioritizes_health_mechanism_study"] else "fails")
        + "**.\n\n"
    )
    text += "| Earlier-state algorithm | Balanced gain (pp) | Day | Night | Conditional family-two balanced band | Pass |\n|---|---:|---:|---:|---|---|\n"
    for arm, r in screens.items():
        a, b, c = r["mean_pp"]
        lo, hi = r["family_two_conditional_bounds_pp"][0]
        text += f"| {arm} | {a:+.3f} | {b:+.3f} | {c:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['passes']} |\n"
    text += "\nThree source histories, one SSL draw and shared warm initialization, crossed with four disjoint fresh continuation groups. Parent ages 512 and 4,096 jointly vary weights, Adam and observed history; replay memory content also varies. Freeze is the unchanged parent. Every checkpoint uses the same 12 canonical 50-label support draws.\n\n"
    text += "The 2,048 new query identities come from unused local training images under the saved inventory. Clip disjointness does not authenticate capture-session independence. Labels/mirror provenance remains provisional. Bootstrap bands condition on the chosen source histories, futures, support pool and query population; they are not population confirmation or training-RNG replication.\n\n"
    text += "Health is observational instrumentation. Geometry movement has no predetermined beneficial sign. The charts compare parent-relative capability, geometry and measured parameter displacement; they do not establish causal mediation or qualify a health controller. A screen pass prioritizes a separately designed mechanism experiment.\n\n"
    text += "Work: 52,224 SSL updates including exact reproduction; 1,224 new fits per reader kind, independent all-panel kNN/linear checks and 54 fine-tuning refits. All original studies and failed attempts are preserved.\n"
    write(
        out / "report-metadata.json",
        {"figures": [str(p) for p in paths], "summary_sha256": digest(out / "summary.json")},
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    code = (
        "import json\nfrom pathlib import Path\n"
        f"root = Path({str(out)!r})\n"
        "summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))\n"
        "verification = json.loads((root / 'verification.json').read_text(encoding='utf-8'))\n"
        "assert verification['passed']\n"
        "for arm, result in summary['readers']['finetune']['earlier_screens'].items():\n"
        "    assert result['passes'] == all(result['requirements'].values())\n"
        "    print(arm, result['mean_pp'], 'passes:', result['passes'])\n"
    )
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, "saved-study39-reconstruction", "exec"), {})
    cells = [
        {"cell_type": "markdown", "id": "study39-report", "metadata": {}, "source": text.splitlines(True)},
        {
            "cell_type": "code",
            "id": "study39-reconstruction",
            "metadata": {},
            "source": code.splitlines(True),
            "execution_count": 1,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": capture.getvalue().splitlines(True)}],
        },
    ]
    for i, p in enumerate(paths):
        assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        cells.append(
            {
                "cell_type": "code",
                "id": f"figure-{i + 1}",
                "metadata": {},
                "source": [f"from IPython.display import Image, display\ndisplay(Image(filename={str(p)!r}))"],
                "execution_count": i + 2,
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": base64.b64encode(p.read_bytes()).decode(), "text/plain": [p.stem]},
                    }
                ],
            }
        )
    notebook = ROOT / study.notebook
    notebook.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    saved_notebook = read(notebook)
    assert len(saved_notebook["cells"]) == 8 and saved_notebook["cells"][1]["execution_count"] == 1
    hashes = {str(p): digest(p) for p in [notebook, out / "report.md", *paths]}
    write(
        out / "delivery-verification.json",
        {
            "passed": True,
            "hashes": hashes,
            "numerical_verification_passed": True,
            "saved_result_reconstruction_executed": True,
            "visual_QA_pending": True,
            "nbformat": "4.5 standard JSON; nbformat package unavailable",
        },
    )
    print({"notebook": str(notebook), "figures": len(paths), "visual_QA": "pending"}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        report()
