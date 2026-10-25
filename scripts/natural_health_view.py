"""Replay two recorded trajectories and present health plus frozen-feature probes."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import base64
import hashlib
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from omegaconf import OmegaConf
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_calibration import embeddings, geometry
from cafl4ds.measurements import cosine_drift, linear_cka
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import capture, restore
from cafl4ds.target_alignment_measurement import metric_means
from cafl4ds.update_filtering import optimize_current
from scripts.evaluate_checkpoint_robustness import probe_predictions
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, plain, read
from scripts.run_retention_anchor_signal import setup

CONFIG = ROOT / "cafl4ds/configs/natural_health_view.yaml"
ARMS = ("full", "adaptive")
LABELS = {"full": "Full training", "adaptive": "Quarter + recovery"}
COLORS = {"full": "#2878a3", "adaptive": "#b95332"}
SIGNALS = ("rank", "variance", "covariance", "cka", "cosine")
FIGURES = ("capability-readers", "health-signals", "alarm-inputs", "training-and-admission")


def write(path, value):
    """Write JSON atomically with finite numpy values converted to builtins."""
    atomic_json(path, plain(value))


def trailing_updates(trace, index, window=32):
    """Smooth over accepted updates and preserve rejected arrivals as missing values."""
    result, recent = [], []
    for row in trace:
        if row["accepted"]:
            recent.append(row["response"][index])
            recent = recent[-window:]
            result.append(float(np.mean(recent)))
        else:
            result.append(np.nan)
    return np.array(result)


def accuracy_panel(predictions, arrays):
    """Keep class and domain weights identical across all three readouts."""
    query = arrays["query"]
    return metric_means(
        100.0 * (np.asarray(predictions) == arrays["labels"][query]),
        arrays["labels"][query],
        arrays["domains"][query],
    )


def frozen_predictions(features, arrays, draws, counts, journal):
    """Fit established readers using support-only normalization and no query tuning."""
    result = {}
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        for kind in ("knn", "linear"):
            panel = []
            for draw in draws:
                selected = arrays[f"support_{draw}"]
                counts[f"{kind}_started"] += 1
                write(journal, counts)
                panel.append(
                    probe_predictions(
                        kind,
                        torch.from_numpy(features[selected]),
                        torch.from_numpy(arrays["labels"][selected]),
                        torch.from_numpy(features[arrays["query"]]),
                    ).numpy()
                )
                counts[f"{kind}_completed"] += 1
                write(journal, counts)
            result[kind] = np.array(panel, dtype=np.int8)
    return result


def freeze(config, out):
    """Lock the extension before new features, probe fits or their outcomes."""
    source = ROOT / config.source
    hashes = {}
    for name in ("input_lock.json", "complete.json", "presentation-verification.json", "delivery-verification.json"):
        artifact = source / name
        record = read(artifact)
        hashes.update(record["hashes"])
        hashes[str(artifact)] = digest(artifact)
    check(hashes)
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "docs/project-plan/natural-health-view.md",
        ROOT / "tests/unit/test_natural_health_view.py",
        ROOT / "scripts/evaluate_checkpoint_robustness.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "cafl4ds/target_alignment_measurement.py",
        ROOT / config.evaluation / "data.npz",
        ROOT / config.evaluation / "images.pt",
    ]
    hashes.update({str(p): digest(p) for p in paths})
    out.mkdir(parents=True, exist_ok=False)
    (out / "features").mkdir()
    (out / "attempts").mkdir()
    lock = {
        "hashes": hashes,
        "config": OmegaConf.to_container(config, resolve=True),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "prepared_unix": time.time(),
        "new_ssl_optimization": False,
        "max_replay_ssl_updates": 6025,
        "max_fits_per_kind_per_stage": 210,
        "post_hoc_readout_extension": True,
    }
    write(out / "input_lock.json", lock)
    return lock


def collect(config, out, lock, counts, journal):
    """Replay the existing input masks and collect state-preserving frozen features."""
    source = ROOT / config.source
    prior = read(source / "input_lock.json")
    records = read(source / "collection.json")["records"]
    images = torch.load(ROOT / config.evaluation / "images.pt", weights_only=True)
    with np.load(ROOT / config.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    assert len(arrays["query"]) == 3663
    for draw in config.draws:
        selected = arrays[f"support_{draw}"]
        assert len(selected) == 50 and not np.intersect1d(selected, arrays["query"]).size
        assert [(arrays["labels"][selected] == c).sum() for c in (0, 1)] == [25, 25]
    assert config.knn_neighbors == 20 and config.linear_c == 1 and config.linear_max_iter == 1000
    bank = reference_images(prior["monitor_reference"])
    stream = OrderedFrames(read(ROOT / config.frames / "intake.json")["clips"])
    method = OmegaConf.create(prior["model_config"])
    torch.manual_seed(11)
    initial = _method(method).cuda()
    start = torch.load(source / "initial_state.pt", map_location="cpu", weights_only=True)
    cache, bindings, geometry_records = {}, {}, {}
    for arm in ARMS:
        record = records[arm]
        model, optimizer = restore(initial, method.optim, start)
        bindings[arm] = {}
        stored = np.load(record["embedding_path"])
        monitor_points = {int(k) for k in stored.files}
        health_points = sorted(monitor_points)
        reference = stored[str(config.reference_step)]
        geometry_records[arm] = {
            "points": health_points,
            "values": {name: [] for name in SIGNALS},
        }
        with threadpool_limits(limits=4):
            for step in health_points:
                values = health_values(stored[str(step)], reference)
                for name in SIGNALS:
                    geometry_records[arm]["values"][name].append(
                        values[name] if step >= config.reference_step or name not in ("cka", "cosine") else None
                    )

        def observe(step, model=model, optimizer=optimizer, stored=stored, record=record, arm=arm):
            marker = fingerprint(capture(model, optimizer))
            z = embeddings(model, bank)
            np.testing.assert_array_equal(z, stored[str(step)])
            if str(step) in record["monitors"]:
                assert marker == record["monitors"][str(step)]["learner_fingerprint"]
            if str(step) in record["observations"]:
                expected = record["observations"][str(step)]
                assert marker == expected["learner_fingerprint"]
                encoder_marker = fingerprint(model.encoder.state_dict())
                assert encoder_marker == expected["encoder_fingerprint"]
                if encoder_marker not in cache:
                    features = embeddings(model, images)
                    assert features.shape == (len(images), 96)
                    predictions = frozen_predictions(features, arrays, config.draws, counts, journal)
                    path = out / "features" / f"{encoder_marker}.npz"
                    np.savez_compressed(path, features=features, **predictions)
                    cache[encoder_marker] = {
                        "path": str(path),
                        "sha256": digest(path),
                        "scores": {k: accuracy_panel(v, arrays) for k, v in predictions.items()},
                    }
                bindings[arm][str(step)] = encoder_marker
                write(out / "bindings.json", bindings)
                write(out / "feature_cache.json", cache)
            assert fingerprint(capture(model, optimizer)) == marker, "Evaluation changed learner state"

        observe(0)
        for row in record["trace"]:
            step = row["step"]
            if row["accepted"]:
                counts["ssl_started"] += 1
                write(journal, counts)
                pixels = stream.batch(step - 1)
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["input_sha256"]
                batch = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().cuda() / 255
                measured = list(optimize_current(model, optimizer, batch, True))
                assert measured == row["response"], (arm, step, "update differs")
                counts["ssl_completed"] += 1
                write(journal, counts)
            else:
                assert fingerprint(capture(model, optimizer)) == row["skipped_state_fingerprint"]
            if step in monitor_points:
                observe(step)
            if step % 128 == 0 or step == len(record["trace"]):
                status = {
                    "stage": "feature replay",
                    "arm": arm,
                    "step": step,
                    "horizon": len(record["trace"]),
                    "ssl_updates": counts["ssl_completed"],
                    "feature_states": len(cache),
                }
                write(out / "status.json", status)
                print(status, flush=True)
        assert fingerprint(capture(model, optimizer)) == record["final_fingerprint"]
        stored.close()
    assert counts["ssl_started"] == counts["ssl_completed"] == lock["max_replay_ssl_updates"]
    assert all(counts[f"{kind}_completed"] <= 210 for kind in ("linear", "knn"))
    result = {
        "bindings": bindings,
        "cache": cache,
        "health": geometry_records,
        "collection_work": dict(counts),
    }
    write(out / "collection.json", result)
    return result, arrays


def verify(config, out, collected, arrays):
    """Refit saved features, independently aggregate predictions and cross-check health."""
    counts = {f"{kind}_{state}": 0 for kind in ("knn", "linear") for state in ("started", "completed")}
    journal = out / "attempts" / "verification.json"
    records = read(ROOT / config.source / "collection.json")["records"]
    for marker, row in collected["cache"].items():
        assert digest(row["path"]) == row["sha256"]
        with np.load(row["path"]) as saved:
            repeated = frozen_predictions(saved["features"], arrays, config.draws, counts, journal)
            for kind, prediction in repeated.items():
                np.testing.assert_array_equal(prediction, saved[kind])
                independent = []
                query, labels, domains = arrays["query"], arrays["labels"], arrays["domains"]
                for panel in prediction:
                    cells = [
                        100 * np.mean(panel[(domains[query] == d) & (labels[query] == c)] == c)
                        for d in (0, 1)
                        for c in (0, 1)
                    ]
                    independent.append([np.mean(cells), np.mean(cells[:2]), np.mean(cells[2:])])
                np.testing.assert_allclose(independent, row["scores"][kind], atol=1e-12, rtol=0)
        print(
            {"stage": "verify probes", "feature_state": marker[:12], "linear_fits": counts["linear_completed"]},
            flush=True,
        )
    for arm in ARMS:
        with np.load(records[arm]["embedding_path"]) as stored:
            reference = stored[str(config.reference_step)]
            for i, step in enumerate(collected["health"][arm]["points"]):
                z = stored[str(step)]
                expected = geometry(z)
                expected.update(cka=max(0.0, 1 - linear_cka(reference, z)), cosine=cosine_drift(reference, z))
                for name in SIGNALS:
                    value = collected["health"][arm]["values"][name][i]
                    if value is not None:
                        np.testing.assert_allclose(value, expected[name], rtol=2e-5, atol=2e-6)
    result = {
        "passed": True,
        "all_checkpoint_features_state_preserving": True,
        "recorded_update_losses_gradients_inputs_exact": True,
        "all_monitor_embeddings_and_fingerprints_exact": True,
        "all_probe_predictions_refit_exact": True,
        "independent_class_domain_aggregation": True,
        "health_float32_instrument_crosscheck": True,
        "unique_feature_states": len(collected["cache"]),
        "verification_work": counts,
    }
    write(out / "verification.json", result)
    return result


def decorate(axes, horizon=4096):
    """Use one arriving-batch clock throughout the focused comparison."""
    for ax in np.asarray(axes).flatten():
        ax.grid(alpha=0.16)
        ax.set_xlim(0, horizon)
    for ax in np.atleast_1d(axes)[-1:].flatten():
        ax.set_xlabel("Arriving batches of 32 images (archive order)")


def save(fig, figures, name):
    """Keep standalone scientific PNG and SVG artifacts."""
    fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def present(config, out, collected, verification):
    """Show only full training and quarter recovery in every scientific figure."""
    source = ROOT / config.source
    summary = read(source / "summary.json")
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
                "finetune": "Fine-tuning · existing readout",
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

    result = {
        "points": points,
        "arms": ARMS,
        "draws": list(config.draws),
        "probe_reader_curves": curves,
        "source_finetune_curves": {arm: summary["rules"][arm]["reader_curves"] for arm in ARMS},
        "post_hoc": True,
        "controller_qualified": False,
        "independent_source_replications": 1,
    }
    write(out / "summary.json", result)
    lines = [
        "# Full training versus quarter + recovery",
        "",
        "Only these two recorded natural-order policies are displayed. All plots use arriving batches; "
        "separate videos are in archive order, not known capture-session chronology.",
        "",
        "## Capability: three different readouts",
        "",
        "kNN and linear probes keep the SSL encoder frozen. Fine-tuning copies and adapts it using "
        "labels; those curves are reused from the original experiment. All three use the same "
        "three support draws (50 scene labels each) and 3,663 primary held-out queries. Balanced "
        "accuracy averages classes within day/night, then domains. Thin curves are individual "
        "readers; thick curves are their mean, not independent-source uncertainty.",
        "",
        "kNN uses 20 cosine neighbours weighted by inverse distance. The linear probe is logistic "
        "regression with C=1 and support-only coordinate standardization. No hyperparameter "
        "selection uses query results. New probes are post-hoc measurements, not controller inputs.",
        "",
        "| Readout | Policy | Start balanced / day / night (%) | Final balanced / day / night (%) |",
        "| --- | --- | --- | --- |",
    ]
    for kind in ("knn", "linear", "finetune"):
        for arm in ARMS:
            curve = curves[arm][kind].mean(1) if kind != "finetune" else np.array(summary["rules"][arm]["curve"])
            left, right = (" / ".join(f"{v:.3f}" for v in curve[i]) for i in (0, -1))
            lines.append(f"| {kind} | {LABELS[arm]} | {left} | {right} |")
    sections = [
        ("capability-readers", "Frozen-feature performance and the existing fine-tuning readout"),
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
    for (name, title), caption in zip(sections, captions, strict=True):
        lines += ["", f"## {title}", "", f"![{title}](figures/{name}.png)", "", caption]
    lines += [
        "",
        "## Provenance and limits",
        "",
        f"The two trajectories replay exactly for {collected['collection_work']['ssl_completed']:,} "
        f"SSL updates. There are {verification['unique_feature_states']} unique frozen-feature "
        "states. Every new probe prediction is reproduced by a verification refit. Original "
        "scientific artifacts remain unchanged. No new SSL policy, seed, source stream or labels "
        "were introduced; replay reconstructs existing states. One stream and one SSL seed "
        "provide descriptive evidence, not controller qualification.",
        "",
        "[Original experiment](../../natural-health-rate/20261016-development/report.md) · "
        "[Readout plan](../../../docs/project-plan/natural-health-view.md) · "
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
                "Five health signals, frozen-encoder kNN and linear probes, and the existing fine-tuning reader.\n\n",
                "[Full methods, scores and provenance]"
                "(../../outputs/natural-health-view/20261016-full-recovery/report.md). "
                "This post-hoc readout uses one ordered BDD100K stream and one SSL seed. "
                "Thin capability curves show the three support draws; thick curves show their mean.\n",
            ],
        }
    ]
    for number, ((name, title), caption) in enumerate(zip(sections, captions, strict=True), 1):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n\n{caption}\n"]})
        relative = f"../../outputs/natural-health-view/20261016-full-recovery/figures/{name}.png"
        cells.append(
            {
                "cell_type": "code",
                "execution_count": number,
                "metadata": {},
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    f"display(Image(filename=str(Path({relative!r}))))\n",
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


def main():
    """Run the bounded replay/readout extension with all earlier science immutable."""
    setup()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert list(config.arms) == list(ARMS) and not (ROOT / config.notebook).exists()
    lock = freeze(config, out)
    journal = out / "attempts" / f"collection_{os.getpid()}.json"
    counts = {f"{kind}_{state}": 0 for kind in ("ssl", "knn", "linear") for state in ("started", "completed")}
    counts["started_unix"] = time.time()
    try:
        collected, arrays = collect(config, out, lock, counts, journal)
        verification = verify(config, out, collected, arrays)
        check(lock["hashes"])
        present(config, out, collected, verification)
        write(out / "status.json", {"stage": "complete", "completed_unix": time.time()})
    except Exception as error:
        counts["error"] = repr(error)
        write(journal, counts)
        raise
    print({"focused_view": "complete", "output": str(out)}, flush=True)


if __name__ == "__main__":
    main()
