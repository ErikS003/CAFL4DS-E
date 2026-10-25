"""Six scientific figures and saved-result notebook for cached night diagnostics."""

from __future__ import annotations

import base64
import contextlib
import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.early_state_night_diagnostic import ASSAYS, FIELDS, SIGNALS
from scripts.run_early_state_night_diagnostic import ROOT, digest, read, reserve, write

COLORS = ("#2878b5", "#3a945c", "#9566b1", "#db4b34")
SHORT = (
    "M rank",
    "M variance",
    "M covariance",
    "M CKA",
    "M cosine",
    "A rank",
    "A variance",
    "A covariance",
    "A CKA",
    "A cosine",
    "Loss cumulative",
    "Loss trailing16",
    "Gradient cumulative",
    "Gradient trailing16",
    "Step L2 cumulative",
    "Step L2 trailing16",
)


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matching(rows, **kwargs):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kwargs.items())]


def coefficient(row, field="spearman"):
    return np.nan if row[field] in (None, "") else float(row[field])


def render_figures(study, gains, correlations, signals):
    out = ROOT / study["output"]
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    paths = []

    def save(fig, name):
        path = directory / (name + ".png")
        fig.savefig(path, dpi=145, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    def heat(ax, values, limit):
        im = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for s in range(3):
            for f in range(4):
                ax.text(
                    f,
                    s,
                    f"{values[s, f]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
                )
        ax.set(xticks=range(4), xticklabels=[1, 2, 3, 4], yticks=range(3), yticklabels=[1, 2, 3])
        return im

    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True, sharey=True)
    limit = max(1, np.abs(gains[0, ..., 2].mean(-1)).max())
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            row = ai * 2 + armi
            for ti, step in enumerate(study["checkpoints"]):
                ax = axes[row, ti]
                im = heat(ax, gains[0, ai, armi, :, :, ti, :, 2].mean(-1), limit)
                ax.set_title(f"Age {age}, {arm}, continuation {step}")
            axes[row, 0].set_ylabel("Source history")
    for ax in axes[-1]:
        ax.set_xlabel("Continuation context")
    fig.suptitle("Night accuracy change versus freeze; all histories, contexts, ages and diets")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Night gain (pp)", shrink=0.6)
    save(fig, "01-all-night-cases")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    limit = max(1, np.abs(gains[0, 0, 1, ..., 5:].mean(-2)).max())
    for ti, step in enumerate(study["checkpoints"]):
        for ci, name in enumerate(("city", "highway"), 5):
            ax = axes[ti, ci - 5]
            im = heat(ax, gains[0, 0, 1, :, :, ti, :, ci].mean(-1), limit)
            ax.set_title(f"Night {name}, continuation {step}")
            if ci == 5:
                ax.set_ylabel("Source history")
    for ax in axes[-1]:
        ax.set_xlabel("Continuation context")
    fig.suptitle("Earlier half replay: class-specific changes; twelve matched supports per cell")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Class gain (pp)", shrink=0.7)
    save(fig, "02-night-class-effects")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            ax = axes[ai, armi]
            values = gains[0, ai, armi].mean(0)
            for fi in range(4):
                for ti, step in enumerate(study["checkpoints"]):
                    ax.plot(
                        range(1, 13),
                        values[fi, ti, :, 2],
                        color=COLORS[fi],
                        ls=("--", "-")[ti],
                        marker="o" if ti == 1 else None,
                        ms=3,
                        label=f"F{fi + 1}, {step}",
                    )
            ax.axhline(0, color="gray", lw=0.8)
            ax.axhline(-1, color="gray", lw=0.7, ls=":")
            ax.set(title=f"Age {age}: {arm}", xlabel="Paired support draw", ylabel="Night gain versus freeze (pp)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("All contexts and both checkpoints; source histories averaged, draws are not independent runs")
    fig.tight_layout()
    save(fig, "03-support-sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    values = gains[:, :, :, :, :, 1, :, 2].mean((3, 5))
    limit = max(1, np.abs(values).max())
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            ax = axes[ai, armi]
            matrix = values[:, ai, armi]
            im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for r in range(3):
                for f in range(4):
                    ax.text(
                        f,
                        r,
                        f"{matrix[r, f]:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
                    )
            ax.set(
                title=f"Age {age}: {arm}",
                xticks=range(4),
                xticklabels=[1, 2, 3, 4],
                yticks=range(3),
                yticklabels=["Fine-tuning", "kNN", "Frozen linear"],
                xlabel="Context",
            )
    fig.suptitle("Terminal night effects depend on the downstream procedure; fine-tuning stays primary")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Night gain versus freeze (pp)", shrink=0.75)
    save(fig, "04-reader-comparison")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    for ai, age in enumerate(study["ages"]):
        for armi, arm in enumerate(study["arms"]):
            matrix = np.zeros((len(SIGNALS), len(ASSAYS)))
            for i, signal in enumerate(SIGNALS):
                for j, (origin, endpoint, assay) in enumerate(ASSAYS):
                    row = matching(
                        correlations,
                        age=age,
                        arm=arm,
                        origin=origin,
                        endpoint=endpoint,
                        assay=assay,
                        component="night",
                        signal=signal,
                        deletion="all",
                    )[0]
                    matrix[i, j] = coefficient(row)
            ax = axes[ai, armi]
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set(
                title=f"Age {age}: {arm}",
                xticks=range(5),
                xticklabels=["16 → 512", "128 now", "128 → 512", "128 remain", "512 now"],
                yticks=range(16),
                yticklabels=SHORT,
            )
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Within-age/diet Spearman association with night LOSS; twelve shared-history cases, no significance claim"
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Spearman rho; positive = more loss", shrink=0.65)
    save(fig, "05-stratified-associations")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    rows = matching(signals, age=512, arm="balanced", origin=128)
    names = [*FIELDS, "cumulative_loss", "cumulative_gradient", "cumulative_step_l2"]
    for i, (ax, name) in enumerate(zip(axes.ravel(), names, strict=True)):
        for row in rows:
            si, fi = int(row["source"]) - 1, int(row["future"]) - 1
            risk = -(gains[0, 0, 1, si, fi, 1, :, 2] - gains[0, 0, 1, si, fi, 0, :, 2]).mean()
            for bank, marker in (("monitor", "o"), ("audit", "x")) if i < 5 else ((None, "o"),):
                key = f"{bank}_{name}" if bank else name
                ax.scatter(float(row[key]), risk, color=COLORS[fi], marker=marker, s=35, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.7)
        xlabel = {
            "cumulative_loss": "Mean loss (updates 1–128)",
            "cumulative_gradient": "Mean preclip gradient norm (1–128)",
            "cumulative_step_l2": "Mean encoder step L2 (1–128)",
        }.get(name, name.replace("_", " "))
        ax.set(xlabel=xlabel, ylabel="Remaining night loss 128 → 512 (pp)")
    legend = [
        plt.Line2D([], [], marker="o", color=c, linestyle="", label=f"Context {i + 1}") for i, c in enumerate(COLORS)
    ]
    legend += [
        plt.Line2D([], [], marker=m, color="black", linestyle="", label=b)
        for b, m in (("Monitor bank", "o"), ("Audit bank", "x"))
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("Earlier half replay, origin128: both banks and cheap dynamics versus later loss; all twelve cases")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, "06-health-dynamics-and-remaining-loss")
    return paths


def report(study):
    out = ROOT / study["output"]
    verification = read(out / "verification.json")
    assert verification["passed"] and verification["summary_sha256"] == digest(out / "summary.json")
    summary = read(out / "summary.json")
    with np.load(out / "decomposition.npz") as z:
        gains = z["gains"]
    correlations, signals = read_csv(out / "correlations.csv"), read_csv(out / "signals.csv")
    paths = render_figures(study, gains, correlations, signals)
    focus = summary["future4_earlier_half_replay"]
    text = "# Cached night-loss diagnostic after notebook39\n\n"
    text += "Exploratory decomposition of saved results. No new training, model inference, readout fits or downloads. Notebook39's failed gate is unchanged.\n\n"
    text += "Future4, earlier half replay: gains versus the unchanged parent, in percentage points.\n\n"
    text += "| Checkpoint | Balanced | Day | Night | Night city | Night highway | Negative night support means |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for ti, step in enumerate(study["checkpoints"]):
        v = focus["means_pp"][ti]
        text += f"| {step} | {v[0]:+.3f} | {v[1]:+.3f} | {v[2]:+.3f} | {v[5]:+.3f} | {v[6]:+.3f} | {focus['negative_night_support_draws'][ti]}/12 |\n"
    text += "\n| Source | Night at128 | Night at512 | City at512 | Highway at512 | Night at512 leaving this source out |\n|---|---:|---:|---:|---:|---:|\n"
    for si in range(3):
        v = focus["source_means_pp"][si]
        leave = focus["leave_source_out_means_pp"][si][1][2]
        text += f"| {si + 1} | {v[0][2]:+.3f} | {v[1][2]:+.3f} | {v[1][5]:+.3f} | {v[1][6]:+.3f} | {leave:+.3f} |\n"
    text += "\nLeave-source-out values diagnose concentration. Removing a source does not rescue or requalify the original screen. Figures and full tables retain every age, diet, source, context and reader. Parent/post absolute scores and newly-wrong/newly-correct accounting are in the case/support tables.\n\n"
    text += "Signals within earlier half replay, measured through128 updates, versus subsequent night loss from128 to512. Each coefficient uses twelve source/context means after averaging twelve supports. Deletion ranges cover all three source and four context omissions; they are sensitivity ranges, not confidence intervals or held-out prediction.\n\n"
    text += "| Signal | Pearson | Spearman | Spearman group-deletion range |\n|---|---:|---:|---|\n"
    for field in SIGNALS:
        rows = matching(
            correlations,
            age=512,
            arm="balanced",
            origin=128,
            endpoint=512,
            assay="remaining",
            component="night",
            signal=field,
        )
        row = matching(rows, deletion="all")[0]
        values = [coefficient(r) for r in rows if r["deletion"] != "all"]
        valid = [v for v in values if np.isfinite(v)]
        bounds = "undefined" if not valid else f"[{min(valid):+.2f}, {max(valid):+.2f}]"
        a, b = coefficient(row, "pearson"), coefficient(row)
        text += f"| {field} | {a:+.3f} | {b:+.3f} | {bounds} |\n"
    text += "\nAll sixteen signals, five time assays and night class outcomes are reported without selecting a best sensor. Contemporary associations are separated from temporally earlier observations. Earlier observations on this reused collection do not establish forecasting transfer.\n\n"
    text += "Health geometry has no assigned beneficial sign. Diet, weights, Adam state, history, content and update direction can still confound interpretation. Actual step L2 is measured after Adam; its cumulative sum is path length, not net parameter displacement. Neither correlations nor group deletions establish causality, mediation, a threshold, or an alarm. The two fixed banks are measurements of shared learners, not independent replications.\n\n"
    text += "The table labels city/highway describe query classes. This analysis does not establish the time-of-day composition or causal content of the incoming continuation videos. There are three histories, four futures, one original SSL draw and shared warm weights; capture-session independence is unknown. All results are exploratory development evidence.\n\n"
    text += "Artifacts: `support-effects.csv`, `case-effects.csv`, `query-transition-summary.csv`, `signals.csv`, `correlations.csv`, `bank-agreement.csv`, `decomposition.npz`. Query transition arrays store per-query counts of losses/gains across12 paired supports; persistent means>=9 draws. Largest-decile concentration is descriptive selection over query identities, never an evaluation subset.\n\n"
    text += "Original1,224 saved panels reconstructed; all8,160 Pearson/Spearman and group-deletion rows independently checked with SciPy. No new model work or downloads. No subsequent intervention is launched.\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    code = (
        "import json\nfrom pathlib import Path\n"
        + f"root = Path({str(out)!r})\n"
        + "summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))\n"
        "verification = json.loads((root / 'verification.json').read_text(encoding='utf-8'))\n"
        "assert verification['passed'] and summary['exploratory']\n"
        "assert summary['new_SSL_updates'] == summary['new_readout_fits'] == summary['download_bytes'] == 0\n"
        "print('Future4 earlier half replay:', summary['future4_earlier_half_replay']['means_pp'])\n"
        "print('Original screen remains unchanged:', summary['original_gate_unchanged'])\n"
    )
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, "cached-night-diagnostic-reconstruction", "exec"), {})
    cells = [
        {"cell_type": "markdown", "id": "diagnostic40-report", "metadata": {}, "source": text.splitlines(True)},
        {
            "cell_type": "code",
            "id": "diagnostic40-reconstruction",
            "metadata": {},
            "execution_count": 1,
            "source": code.splitlines(True),
            "outputs": [{"output_type": "stream", "name": "stdout", "text": capture.getvalue().splitlines(True)}],
        },
    ]
    for i, path in enumerate(paths):
        cells.append(
            {
                "cell_type": "code",
                "id": f"diagnostic40-figure-{i + 1}",
                "metadata": {},
                "execution_count": i + 2,
                "source": [f"from IPython.display import Image, display\ndisplay(Image(filename={str(path)!r}))"],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": base64.b64encode(path.read_bytes()).decode(), "text/plain": [path.stem]},
                    }
                ],
            }
        )
    notebook = ROOT / study["notebook"]
    notebook.parent.mkdir(parents=True, exist_ok=True)
    write(
        notebook,
        {
            "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        },
    )
    assert len(read(notebook)["cells"]) == 8
    write(
        out / "delivery-verification.json",
        {
            "passed": True,
            "visual_QA_pending": True,
            "saved_result_reconstruction_executed": True,
            "hashes": {str(p): digest(p) for p in (notebook, out / "report.md", *paths)},
        },
    )
    reserve(study)
    print({"notebook": str(notebook), "figures": len(paths), "visual_QA_pending": True}, flush=True)
