"""Summarize an immediately stopped run using identical completed comparisons."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_forecast import HEALTH
from cafl4ds.single_image_signal import performance64
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_extended_gate import assert_inputs
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gate_original", "gate_half", "gate_zero", "independent_25", "full")
COLORS = dict(zip(ARMS, ["#2166ac", "#d6604d", "#762a83", "#cc8b27", "#1b7837"], strict=True))


def cluster_interval(values, groups, draws):
    """Weight each complete pair equally; bootstrap intact, unequal-size histories."""
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    sums = np.array([values[groups == group].sum(0) for group in unique])
    counts = np.array([(groups == group).sum() for group in unique])
    denominator = counts[draws].sum(1).reshape((-1,) + (1,) * (values.ndim - 1))
    sampled = sums[draws].sum(1) / denominator
    return dict(mean=values.mean(0).tolist(), ci95=np.quantile(sampled, [0.025, 0.975], axis=0).tolist())


def finalize(root):
    assert read(root / "user-stop.json")["automatic_restart"] is False
    lock = assert_inputs(root)
    original = ROOT / lock["settings"]["source"]
    roster = [r for r in read(root / "collection.json")["records"] if r["arm"] in ARMS]
    records = {(r["history"], r["seed"], r["arm"]): read(root / r["path"] / "trajectory.json") for r in roster}
    keys = sorted({key[:2] for key in records})
    common = [key for key in keys if all((*key, arm) in records for arm in ARMS)]
    assert len(roster) == 27 and len(common) == 5
    settings = lock["settings"]
    clocks = np.arange(settings["start"], settings["start"] + settings["offered"] + 1, settings["checkpoint_stride"])
    metric_names = list(records[(*common[0], ARMS[0])]["checkpoints"][str(clocks[0])]["metrics"])
    panel = torch.load(original / "intake/evaluation-panel.pt", map_location="cpu", weights_only=True)
    models = joblib.load(original / "forecasters.joblib")
    schedules = dict(np.load(root / "independent-schedules.npz"))
    trace_cache, manifests = {}, {}
    audit_count = 0
    for row in roster:
        key = (row["history"], row["seed"], row["arm"])
        record, branch = records[key], root / row["path"]
        with np.load(branch / "trace.npz") as trace:
            accepted = trace["accepted"].copy()
            assert len(accepted) == settings["offered"] and accepted.sum() == record["accepted"]
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            if row["arm"].startswith("gate_"):
                forecaster = models["J_rich"][row["seed"]]
                np.testing.assert_allclose(
                    forecaster.predict(trace["features"]), trace["predictions"], rtol=1e-9, atol=1e-9
                )
                np.testing.assert_array_equal(accepted, trace["predictions"] < record["spec"]["threshold"])
                origin = trace["features"][:, 20]
                range_fraction = float(np.mean((origin < forecaster.minimum[20]) | (origin > forecaster.maximum[20])))
            elif row["arm"] == "full":
                assert accepted.all()
                range_fraction = None
            else:
                np.testing.assert_array_equal(accepted, schedules[f"{row['history']}__{row['seed']}__independent_25"])
                range_fraction = None
            trace_cache[key] = dict(accepted=accepted, training=trace["training"].copy(), origin_outside=range_fraction)
        state = torch.load(branch / "final_state.pt", map_location="cpu", weights_only=True)
        assert fingerprint(state) == record["final_fingerprint"]
        for clock in clocks:
            checkpoint = record["checkpoints"][str(clock)]
            raw_path = (
                branch.parent / "common/raw.npz" if clock == clocks[0] else branch / f"checkpoint_{clock}/raw.npz"
            )
            with np.load(raw_path) as raw:
                for p, probabilities in enumerate(raw["probabilities"]):
                    rebuilt = performance64(
                        dict(linear=probabilities[0], knn=probabilities[1]),
                        panel["query_labels"].numpy(),
                        panel["query_domains"].numpy(),
                    )
                    for metric, value in rebuilt.items():
                        np.testing.assert_allclose(value, checkpoint["panels"][p][metric], rtol=1e-12, atol=1e-12)
                for bank in ("selector", "evaluation"):
                    values = raw[bank + "_values"]
                    for metric, value in zip(HEALTH, values, strict=True):
                        np.testing.assert_allclose(value, checkpoint["metrics"][bank + "_" + metric])
                    np.testing.assert_allclose(
                        raw[bank + "_reconstruction"].mean(), checkpoint["metrics"][bank + "_bank_reconstruction"]
                    )
            for metric in checkpoint["panels"][0]:
                np.testing.assert_allclose(
                    checkpoint["metrics"][metric], np.mean([p[metric] for p in checkpoint["panels"]])
                )
            audit_count += 1
        manifests[str(branch / "trajectory.json")] = digest(branch / "trajectory.json")
        manifests[str(branch / "trace.npz")] = digest(branch / "trace.npz")
        manifests[str(branch / "final_state.pt")] = digest(branch / "final_state.pt")
    values = np.array(
        [
            [
                [
                    [records[(*key, arm)]["checkpoints"][str(clock)]["metrics"][m] for m in metric_names]
                    for clock in clocks
                ]
                for arm in ARMS
            ]
            for key in common
        ]
    )
    accepted = np.array(
        [
            [
                np.r_[
                    0,
                    np.cumsum(trace_cache[(*key, arm)]["accepted"])[
                        settings["checkpoint_stride"] - 1 :: settings["checkpoint_stride"]
                    ],
                ]
                for arm in ARMS
            ]
            for key in common
        ]
    )
    groups = np.array([key[0] for key in common])
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = rng.integers(0, len(np.unique(groups)), (settings["bootstrap_draws"], len(np.unique(groups))))
    curves = {
        arm: {m: cluster_interval(values[:, j, :, k], groups, draws) for k, m in enumerate(metric_names)}
        for j, arm in enumerate(ARMS)
    }
    result = dict(
        stopped_by_user=True,
        collection_complete=False,
        completed_trajectories=27,
        primary_trajectories=25,
        primary_complete_paired_sets=5,
        source_histories=3,
        common_pairs=[dict(history=h, seed=int(s)) for h, s in common],
        weighting="Each of five complete history/seed pairs has equal weight; bootstrap resamples intact histories",
        collection_seed_coverage={str(h): int((groups == h).sum()) for h in np.unique(groups)},
        arms={},
        comparisons={},
        curves=curves,
        exploratory=True,
        clocks=clocks.tolist(),
    )
    for j, arm in enumerate(ARMS):
        rows = [records[(*key, arm)] for key in common]
        result["arms"][arm] = dict(
            n=5,
            mean_accepted=float(accepted[:, j, -1].mean()),
            admission=float(accepted[:, j, -1].mean() / settings["offered"]),
            accepted_range=[int(accepted[:, j, -1].min()), int(accepted[:, j, -1].max())],
            mean_policy_seconds=float(np.mean([r["policy_seconds"] for r in rows])),
            before={m: float(values[:, j, 0, k].mean()) for k, m in enumerate(metric_names)},
            after={m: float(values[:, j, -1, k].mean()) for k, m in enumerate(metric_names)},
            change={m: float((values[:, j, -1, k] - values[:, j, 0, k]).mean()) for k, m in enumerate(metric_names)},
            origin_outside_calibration=float(np.mean([trace_cache[(*key, arm)]["origin_outside"] for key in common]))
            if arm.startswith("gate_")
            else None,
        )
    for gate in ARMS[:3]:
        for control in ARMS:
            if gate == control:
                continue
            result["comparisons"][gate + "__" + control] = {
                metric: cluster_interval(
                    values[:, ARMS.index(gate), -1, k] - values[:, ARMS.index(control), -1, k], groups, draws
                )
                for k, metric in enumerate(metric_names)
            }
    extra_keys = sorted({key[:2] for key in records if key[2] in ARMS[:2]})
    assert len(extra_keys) == 6
    original_gate = np.array(
        [
            records[(*key, "gate_original")]["checkpoints"][str(clocks[-1])]["metrics"]["linear_accuracy_balanced"]
            for key in extra_keys
        ]
    )
    half_gate = np.array(
        [
            records[(*key, "gate_half")]["checkpoints"][str(clocks[-1])]["metrics"]["linear_accuracy_balanced"]
            for key in extra_keys
        ]
    )
    result["additional_gate_only_comparison"] = dict(
        n=6,
        original_accuracy=float(original_gate.mean()),
        half_accuracy=float(half_gate.mean()),
        original_minus_half=cluster_interval(
            original_gate - half_gate, np.array([key[0] for key in extra_keys]), draws
        ),
    )
    result["reader_convergence_warnings"] = {
        path.name: path.read_text(errors="replace").count("ConvergenceWarning")
        for path in root.glob("*.stderr.log")
        if "ConvergenceWarning" in path.read_text(errors="replace")
    }
    result["best_mean_gate"] = max(ARMS[:3], key=lambda arm: result["arms"][arm]["after"]["linear_accuracy_balanced"])
    atomic_json(root / "stopped-results.json", result)
    np.savez_compressed(
        root / "stopped-results-arrays.npz",
        values=values,
        accepted=accepted,
        metrics=np.array(metric_names),
        arms=np.array(ARMS),
        groups=groups,
        seeds=np.array([key[1] for key in common]),
        clocks=clocks,
    )
    folder = root / "figures"
    images = []

    def line(ax, arm, metric):
        record = curves[arm][metric]
        ax.plot(clocks - clocks[0], record["mean"], label=arm.replace("_", " "), color=COLORS[arm], lw=1.6)
        if arm == "gate_original":
            ax.fill_between(clocks - clocks[0], *record["ci95"], color=COLORS[arm], alpha=0.10)
        ax.grid(alpha=0.2)
        ax.axvline(2304, color="#aaaaaa", ls=":", lw=0.7)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, metric in zip(
        axes.flat,
        ["linear_accuracy_balanced", "linear_ce_balanced", "knn_accuracy_balanced", "knn_ce_balanced"],
        strict=True,
    ):
        for arm in ARMS:
            line(ax, arm, metric)
        ax.set(title=metric.replace("_", " "), xlabel="Offered frames after common parent")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Stopped run: five identical completed comparisons; three histories")
    fig.savefig(folder / "stopped-reader-trajectories.png", dpi=140)
    plt.close(fig)
    images.append("stopped-reader-trajectories.png")
    for bank in ("evaluation", "selector"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        for ax, metric in zip(axes.flat, [*HEALTH, "objective"], strict=True):
            for arm in ARMS:
                line(ax, arm, bank + "_" + metric)
            ax.set(title=metric.replace("_", " "), xlabel="Offered frames after common parent")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Stopped run: {bank} fixed-bank health; same five completed comparisons")
        name = f"stopped-{bank}-health.png"
        fig.savefig(folder / name, dpi=140)
        plt.close(fig)
        images.append(name)
    text = "# 72 — Extended temporal gate: results at user stop\n\n"
    text += (
        "**Stopped immediately at the user's request: 27 completed trajectories; "
        "the 30-trajectory target was not reached.**\n\n"
        "The main table and every curve use the same five completed history/seed comparisons for all five policies, "
        "each at 19,200 arrivals. Two histories have both seeds; the third has one. "
        "The two extra completed original/half-gate trajectories are used only in the separate six-pair gate comparison. "
        "The unfinished zero-gate trajectory is excluded from endpoint results.\n\n"
        "| Policy | Mean accepted updates | Accuracy % | Change from fork, pp | Linear CE | Charged policy minutes |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for arm, r in result["arms"].items():
        text += (
            f"| {arm} | {r['mean_accepted']:.1f} | {r['after']['linear_accuracy_balanced']:.3f} | "
            f"{r['change']['linear_accuracy_balanced']:+.3f} | {r['after']['linear_ce_balanced']:.4f} | "
            f"{r['mean_policy_seconds'] / 60:.2f} |\n"
        )
    text += (
        "\nThe original gate has the highest mean linear accuracy in the common comparisons. "
        "Its superiority is uncertain: exploratory paired intervals below include zero. "
        "Reduced updates do not imply reduced computation; scoring is expensive. "
        "CE, kNN and health rank policies differently.\n\n"
    )
    text += "| Paired accuracy contrast | Mean pp | Exploratory 95% interval |\n|---|---:|---|\n"
    for key in ["gate_original__full", "gate_original__independent_25", "gate_original__gate_half"]:
        r = result["comparisons"][key]["linear_accuracy_balanced"]
        text += f"| {key} | {r['mean']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |\n"
    text += (
        "\nEach complete pair has equal weight. Whole-history bootstrap resampling retains all available "
        "seed/panel/checkpoint observations; draws are weighted by their realized pair counts so the point estimates "
        "match the main table. Earlier live-report intervals used equal history means despite unequal seed coverage; "
        "use stopped-results.json for the consistent final analysis.\n\n"
    )
    extra = result["additional_gate_only_comparison"]
    text += (
        f"Across all six completed original/half-gate pairs, original accuracy is {extra['original_accuracy']:.3f}% "
        f"and half-threshold accuracy is {extra['half_accuracy']:.3f}%. "
        "This comparison includes both seeds in all three histories; baseline comparisons remain restricted to five pairs.\n\n"
        "The same evaluation/support panels and one warm initialization are reused. "
        "The user stopped after inspecting interim results. The frozen gate's arrival coordinate extrapolates beyond "
        "calibration, and its admissions vary substantially across runs. At least one probe fit emitted a convergence "
        "warning. These limitations make the ranking exploratory. Retired matched controls are preserved as ancillary "
        "data and were not continued for this analysis.\n\n"
        "Verification reconstructs every saved gate decision, independent schedule, final-state fingerprint, reader "
        "metric, panel mean and fixed-bank reconstruction mean for all 27 completed trajectories. "
        "This is a numerical saved-data audit, not a new physical replay of every training update. "
        "Inputs and original source locks remain preserved.\n\n"
    )
    for name in images:
        text += f"![{name}](figures/{name})\n\n"
    (root / "results-at-stop.md").write_text(text, encoding="utf-8")
    cells = [dict(cell_type="markdown", metadata={}, source=text.split("![", 1)[0].splitlines(keepends=True))]
    for name in images:
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=[f"![{name}](attachment:{name})"],
                attachments={name: {"image/png": base64.b64encode((folder / name).read_bytes()).decode()}},
            )
        )
    notebook = dict(
        cells=cells,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
        nbformat=4,
        nbformat_minor=4,
    )
    archive = root / "quality/user-stop-now"
    for path in [
        root / "72_single_image_extended_gate.ipynb",
        ROOT / "notebooks/sums/72_single_image_extended_gate.ipynb",
    ]:
        target = archive / ("previous-" + path.name)
        if not target.exists():
            shutil.copy2(path, target)
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    atomic_json(
        root / "stopped-results-verification.json",
        dict(
            passed=True,
            collection_complete=False,
            stopped_by_user=True,
            completed_trajectories=27,
            primary_trajectories=25,
            primary_paired_sets=5,
            checkpoint_readouts_checked=audit_count,
            source_and_data_locks_valid=True,
            numerical_audit=True,
            full_training_replay=False,
            analysis_script_sha256=digest(Path(__file__)),
            retained_file_hashes=manifests,
        ),
    )
    print(json.dumps({k: v for k, v in result.items() if k not in {"curves"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    finalize(parser.parse_args().output.resolve())
