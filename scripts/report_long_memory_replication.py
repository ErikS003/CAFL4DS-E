"""Present source-level memory utility, fixed-anchor retention and diagnostic health."""

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

from cafl4ds.long_memory_replication import conditional_interval, endpoint_screen, path_summary
from scripts.natural_health_view import write
from scripts.prepare_long_memory_replication import CONFIG
from scripts.run_preaction_stopping import ROOT, check, digest, read

KINDS = ("finetune", "knn", "linear")
DOMAINS = ("Balanced", "Day", "Night")
ARMS = ("current", "fifo", "reservoir")
COLORS = {"current": "#297ea8", "fifo": "#bd7138", "reservoir": "#3b8c66"}
LABELS = {"current": "Current only", "fifo": "Recent FIFO", "reservoir": "Historical reservoir"}


def load_scores(study):
    out = ROOT / study.study_root
    points = read(out / "input_lock.json")["points"]
    warm = read(out / "readouts/warm/complete.json")["scores"]
    curves = np.empty((3, 3, 3, 3, 33, 3, 3))
    endpoints = np.empty((3, 3, 3, 3, 256, 3))
    records = {}
    for bi, block in enumerate(study.blocks):
        for si, seed in enumerate(study.ssl_seeds):
            for ai, arm in enumerate(ARMS):
                directory = out / f"block{block}-seed{seed}-{arm}"
                records[bi, si, ai] = read(directory / "record.json")
                for ki, kind in enumerate(KINDS):
                    curves[ki, bi, si, ai, 0] = warm[kind][:3]
                for pi, point in enumerate(points[1:], 1):
                    completed = read(out / "readouts" / f"b{block}-s{seed}-{arm}-{point}" / "complete.json")
                    for ki, kind in enumerate(KINDS):
                        values = np.asarray(completed["scores"][kind])
                        curves[ki, bi, si, ai, pi] = values[:3]
                        if point == 4096:
                            endpoints[ki, bi, si, ai] = values
    return points, warm, curves, endpoints, records


def paired_correctness(out, study, arm, comparator):
    with np.load(ROOT / study.evaluation / "data.npz") as saved:
        labels, domains = saved["labels"][saved["query"]], saved["domains"][saved["query"]]
    draws = read(ROOT / study.evaluation / "data-role-manifest.json")["support_seeds"]
    result = np.zeros((256, len(labels)), dtype=np.float64)
    for block in study.blocks:
        for seed in study.ssl_seeds:
            a = out / "readouts" / f"b{block}-s{seed}-{arm}-4096"
            b = out / "readouts" / ("warm" if comparator == "warm" else f"b{block}-s{seed}-{comparator}-4096")
            for di, draw in enumerate(draws):
                with np.load(a / f"panel-{draw}.npz") as current, np.load(b / f"panel-{draw}.npz") as base:
                    result[di] += (
                        (current["probabilities"][:3663].argmax(-1) == labels).astype(float)
                        - (base["probabilities"][:3663].argmax(-1) == labels)
                    ) / 9
    return result, labels, domains


def collect(study):
    out = ROOT / study.study_root
    assert read(out / "independent-verification.json")["passed"]
    check(read(out / "input_lock.json")["hashes"])
    points, warm, curves, endpoints, records = load_scores(study)
    primary = endpoint_screen(endpoints[0, :, :, 0], endpoints[0, :, :, 1], endpoints[0, :, :, 2], warm["finetune"])
    # Reconstruct the point decision with explicit scalar loops, independently of the helper.
    source_effects, source_fifo, positives = [], [], []
    for b in range(3):
        run_values = [
            sum(float(endpoints[0, b, s, 2, p, 0] - endpoints[0, b, s, 0, p, 0]) for p in range(256)) / 256
            for s in range(3)
        ]
        source_effects.append(sum(run_values) / 3)
        source_fifo.append(
            sum(float(endpoints[0, b, s, 2, p, 0] - endpoints[0, b, s, 1, p, 0]) for s in range(3) for p in range(256))
            / 768
        )
        positives.append(sum(x > 0 for x in run_values))
    np.testing.assert_allclose(source_effects, primary["per_source_vs_current"][:, 0], atol=1e-12, rtol=0)
    expected = (
        sum(source_effects) / 3 >= 1
        and min(source_effects) > 0
        and min(positives) >= 2
        and min(source_fifo) > 0
        and (primary["per_source_vs_current"][:, 1:] >= -1).all()
        and primary["balanced_warm_gain"] > 0
    )
    assert bool(expected) == primary["point_screen_passed"]
    intervals = {}
    with np.load(ROOT / study.evaluation / "resamples_256.npz") as saved:
        panel, query = saved["panel"], saved["query"]
    with threadpool_limits(limits=4):
        for comparator in ("current", "fifo", "warm"):
            difference, labels, domains = paired_correctness(out, study, "reservoir", comparator)
            intervals[comparator] = conditional_interval(difference, labels, domains, panel, query)
    precision = all(np.max(intervals[c]["half_width"]) <= 1 for c in ("current", "fifo"))
    paths = {}
    for ki, kind in enumerate(KINDS):
        for ai, arm in enumerate(ARMS):
            # First average readers to form each run's pooled path; keep per-draw paths too.
            path = path_summary(curves[ki, :, :, ai].mean(-2), points)
            path["per_draw"] = path_summary(np.swapaxes(curves[ki, :, :, ai], -3, -2), points)
            paths[f"{kind}/{arm}"] = path
    counters = [read(p) for p in out.glob("block*-seed*-*/*-work.json")]
    assert len(counters) == 54 and sum(c["ssl_started"] for c in counters) == 221184
    fits = [read(out / f"reader-worker-{i}.json") for i in range(study.reader_workers)]
    for kind in KINDS:
        assert sum(c[f"{kind}_started"] for c in fits) == sum(c[f"{kind}_completed"] for c in fits) == 9679
    summary = {
        "primary": primary,
        "conditional_intervals": intervals,
        "one_pp_precision_passed": precision,
        "development_screen_passed": primary["point_screen_passed"] and precision,
        "paths": paths,
        "endpoint_means": endpoints.mean(axis=(1, 2, 4)),
        "endpoint_source_means": endpoints.mean(axis=(2, 4)),
        "warm_means": {k: np.mean(v, 0) for k, v in warm.items()},
        "work": {
            "scientific_SSL": 221184,
            "collection_SSL": 110592,
            "reader_fits_per_kind": 9679,
            "independent_FT_refits": 28,
        },
        "controller_qualified": False,
        "fresh_evaluation_population": False,
        "protection_claim_qualified": False,
    }
    write(out / "summary.json", summary)
    np.savez_compressed(out / "capability-arrays.npz", curves=curves, endpoints=endpoints, points=points)
    write(
        out / "statistics-verification.json",
        {
            "passed": True,
            "explicit_point_gate_reconstruction": True,
            "summary_sha256": digest(out / "summary.json"),
            "source_effects": source_effects,
        },
    )
    return summary, curves, endpoints, records, points


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figures(study, summary, curves, endpoints, records, points):
    directory = ROOT / study.study_root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for gi, ax in enumerate(axes):
        for bi in range(3):
            effects = (endpoints[0, bi, :, 2] - endpoints[0, bi, :, 0]).mean(1)
            ax.scatter(
                np.full(3, bi + 1) + np.array([-0.08, 0, 0.08]), effects[:, gi], color=COLORS["reservoir"], alpha=0.6
            )
            ax.scatter(bi + 1, effects[:, gi].mean(), marker="D", color="black", s=40)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(1 if gi == 0 else -1, color="gray", ls=":", lw=0.8)
        ax.set(title=DOMAINS[gi], xlabel="Source block", xticks=[1, 2, 3])
    axes[0].set_ylabel("Reservoir − current-only accuracy (pp)")
    fig.suptitle("Primary endpoint: 256 matched reader draws; dots are training draws")
    save(fig, directory, "01_endpoint_effects")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for bi in range(3):
        for gi, ax in enumerate(axes[bi], 1):
            for ai, arm in enumerate(ARMS):
                values = curves[0, bi, :, ai, :, :, gi].mean(-1)
                for value in values:
                    ax.plot(points, value, color=COLORS[arm], alpha=0.2, lw=0.7)
                ax.plot(points, values.mean(0), color=COLORS[arm], label=LABELS[arm])
            ax.axhline(curves[0, bi, 0, 0, 0, :, gi].mean(), color="gray", ls=":")
            ax.set(title=f"Source {bi + 1}: {DOMAINS[gi]}", ylabel="Fine-tuning accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Incoming batches")
    fig.suptitle("Descriptive trajectories use three reader draws throughout, including the endpoint")
    save(fig, directory, "02_domain_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ki, ax in enumerate(axes):
        for ai, arm in enumerate(ARMS):
            ax.plot(
                range(3), endpoints[ki, :, :, ai].mean(axis=(0, 1, 2)), marker="o", color=COLORS[arm], label=LABELS[arm]
            )
        ax.plot(range(3), summary["warm_means"][KINDS[ki]], color="gray", ls=":", label="Warm frozen")
        ax.set(title=KINDS[ki], xticks=range(3), xticklabels=DOMAINS)
    axes[0].set_ylabel("Final accuracy (%)")
    axes[0].legend(fontsize=8)
    save(fig, directory, "03_readers")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes,
        ("mean_warm_shortfall", "max_warm_shortfall", "max_drawdown"),
        ("Mean shortfall below warm", "Maximum shortfall below warm", "Peak-relative drawdown"),
        strict=True,
    ):
        for arm in ARMS:
            values = np.asarray(summary["paths"][f"finetune/{arm}"][metric])[..., 1:].max(-1)
            ax.plot([1, 2, 3], values.mean(1), marker="o", color=COLORS[arm], label=LABELS[arm])
        ax.set(title=title, xlabel="Source block", ylabel="Worst-domain loss (pp)", xticks=[1, 2, 3])
    axes[0].legend(fontsize=8)
    save(fig, directory, "04_fixed_anchor_retention")

    signals = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")
    fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex=True)
    for bi in range(3):
        for ki, signal in enumerate(signals):
            ax = axes[ki, bi]
            for ai, arm in enumerate(ARMS):
                all_values = []
                for si in range(3):
                    rows = records[bi, si, ai]["readings"]
                    selected = sorted(int(t) for t, r in rows.items() if r["metrics"]["monitor"]["change"] is not None)
                    values = [rows[str(t)]["metrics"]["monitor"]["change"][signal] for t in selected]
                    all_values.append(values)
                ax.plot(selected, np.mean(all_values, 0), color=COLORS[arm], label=LABELS[arm])
            ax.set_title(f"Source {bi + 1}: {signal}", fontsize=9)
            ax.axhline(0, color="gray", lw=0.5)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Incoming batches")
    fig.suptitle("Health is diagnostic; geometric change has no assigned capability direction")
    save(fig, directory, "05_health")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ai, arm in enumerate(ARMS[1:], 1):
        trace = records[0, 0, ai]["trace"]
        ages = [np.mean(r["diet"]["replay_arrival_ages"]) if r["diet"]["replay_ids"] else 0 for r in trace]
        axes[0].plot(range(1, 4097), ages, color=COLORS[arm], alpha=0.75, label=LABELS[arm])
        ids = set()
        counts = []
        for row in trace:
            ids.update(row["diet"]["replay_ids"])
            counts.append(len(ids))
        axes[1].plot(range(1, 4097), counts, color=COLORS[arm], label=LABELS[arm])
    axes[0].set(title="Historical exposure age", ylabel="Mean age (incoming batches)", xlabel="Incoming batches")
    axes[1].set(title="Distinct historical identities replayed", ylabel="Frames", xlabel="Incoming batches")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed sampler draws produce the same index/age histories across blocks and training draws")
    save(fig, directory, "06_actual_diets")


def deliver(study, summary):
    out = ROOT / study.study_root
    primary = summary["primary"]
    verdict = (
        "passes the bounded development screen"
        if summary["development_screen_passed"]
        else "does not qualify a positive memory lead"
    )
    lines = [
        "# Long-horizon historical memory: source replication",
        "",
        f"The fixed reservoir {verdict}.",
        "",
        "| Reader | Reservoir − current: balanced/day/night (pp) | Reservoir − FIFO: balanced/day/night (pp) |",
        "|---|---|---|",
    ]
    means = summary["endpoint_means"]
    for ki, kind in enumerate(KINDS):
        a, b = np.asarray(means[ki][2]) - means[ki][0], np.asarray(means[ki][2]) - means[ki][1]
        lines.append(
            f"| {kind} | " + " / ".join(f"{x:+.3f}" for x in a) + " | " + " / ".join(f"{x:+.3f}" for x in b) + " |"
        )
    lines += [
        "",
        "| Source | Reservoir − current: balanced/day/night (pp) | Reservoir − FIFO: balanced/day/night (pp) |",
        "|---|---|---|",
    ]
    for bi in range(3):
        a, b = primary["per_source_vs_current"][bi], primary["per_source_vs_fifo"][bi]
        lines.append(
            f"| {bi + 1} | " + " / ".join(f"{x:+.3f}" for x in a) + " | " + " / ".join(f"{x:+.3f}" for x in b) + " |"
        )
    lines += ["", "| Comparator | Conditional balanced / day / night 95% ranges (pp) |", "|---|---|"]
    for comparator, interval in summary["conditional_intervals"].items():
        values = [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(interval["lower"], interval["upper"], strict=True)]
        lines.append(f"| {comparator} | " + " / ".join(values) + " |")
    lines += [
        "",
        "These ranges resample the paired reader draws and query images. They describe conditional evaluation sensitivity, not confidence over new capture sessions.",
    ]
    lines += [
        "",
        "| Development requirement | Decision |",
        "|---|---|",
        *[
            f"| {name.replace('_', ' ')} | {'Pass' if passed else 'Fail'} |"
            for name, passed in primary["gates"].items()
        ],
        "",
        f"Conditional one-point precision: {summary['one_pp_precision_passed']}.",
        "",
        "Three new clip-disjoint 131,072-frame blocks, three training draws each, and three fixed policies.",
        "Primary terminal scores use 256 paired reader draws; intermediate curves use three. The warm reference is shared.",
        "Source blocks and the evaluation population do not establish independent capture-session/population confirmation.",
        "Fine-tuning measures adaptability after a fixed labeled budget. Frozen readers answer a separate question.",
        "Replay can include observed frames not previously selected for training. Gains may reflect added coverage as well as retention; age, source coverage and redundancy change together.",
        "Health never chooses an action. No health controller is qualified. Path metrics are descriptive on a sparse fixed grid.",
        "",
        "Scientific SSL: 110,592 collection plus 110,592 exact reproduction updates. Independent source/diet/reader audits pass.",
        "All 9,679 kNN and linear fits are independently refitted; 28 fixed fine-tuning fits reproduce exactly.",
        "Engineering checks add 64 discarded SSL updates and six fine-tuning fits, recorded separately from the scientific comparison.",
        "",
    ]
    report = "\n".join(lines)
    (out / "report.md").write_text(report, encoding="utf-8")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [report]}]
    code = (
        "import json\nfrom pathlib import Path\n"
        "root = Path.cwd()\n"
        "while not (root / 'cafl4ds').is_dir():\n    root = root.parent\n"
        f"summary = json.loads((root / '{study.study_root}/summary.json').read_text())\n"
        "print('Reservoir minus current, pp:', summary['primary']['overall_vs_current'])\n"
        "print('Development screen:', summary['development_screen_passed'])\n"
    )
    # Execute the exact notebook reconstruction and store its actual stdout.
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(code, {})  # noqa: S102 - execute only the literal reconstruction authored just above
    printed = capture.getvalue()
    cells.append(
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "source": [code],
            "outputs": [{"output_type": "stream", "name": "stdout", "text": [printed]}],
        }
    )
    for fig in sorted((out / "figures").glob("*.png")):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {fig.stem.replace('_', ' ')}\n"]})
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": base64.b64encode(fig.read_bytes()).decode()},
                    }
                ],
            }
        )
    notebook = ROOT / study.notebook
    for index, cell in enumerate(cells):
        cell["id"] = f"long-memory-{index:03d}"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": cells,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    for folder in ("phase1", "latest-results"):
        target = ROOT / "docs/experiments" / folder / "long-memory-replication.md"
        text = (
            report
            if folder == "phase1"
            else (
                "# Long-horizon historical-memory replication\n\n"
                f"The fixed historical reservoir {verdict}. Full-trajectory policies were compared on fresh source clips "
                "under paired stochastic training draws. Health remained observational.\n\n"
                "Endpoint utility and fixed-anchor retention are reported separately. Frozen-feature readers remain "
                "separate outcomes from few-label fine-tuning adaptability. Reused evaluation data and unknown capture "
                "sessions limit population confirmation. No health controller is qualified.\n\n"
                "Exact trajectory reproduction and independent source, memory and reader audits pass.\n"
            )
        )
        text += "\n[Frozen methods](../../project-plan/long-memory-replication-frozen.md) · [Full report](../../../outputs/long-memory-replication/20261018/report.md) · [Notebook](../../../notebooks/sums/35_long_memory_replication.ipynb)\n"
        target.write_text(text, encoding="utf-8")
    hashes = {
        str(p): digest(p)
        for p in [out / "summary.json", out / "report.md", notebook, *sorted((out / "figures").glob("*.png"))]
    }
    write(
        out / "presentation-verification.json",
        {
            "passed": True,
            "hashes": hashes,
            "executed_notebook_reconstruction": True,
            "figure_count": 6,
            "visual_inspection_pending": True,
        },
    )


def main():
    study = OmegaConf.load(CONFIG)
    summary, curves, endpoints, records, points = collect(study)
    figures(study, summary, curves, endpoints, records, points)
    deliver(study, summary)
    print(
        {
            "development_screen_passed": summary["development_screen_passed"],
            "primary_effect": summary["primary"]["overall_vs_current"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
