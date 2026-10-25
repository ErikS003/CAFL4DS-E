"""N74: report three pilots against the existing N73 full trajectory."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_extended_gate import load_checkpoint
from cafl4ds.single_image_gate_protection import ARMS
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
COLORS = dict(corrected_J="#2166ac", corrected_covariance="#b85c22", corrected_development="#904b9a", full="#35965f")


def report(root):
    lock = read(root / "input-lock.json")
    start, target = lock["settings"]["start"], lock["settings"]["offered"]
    directory = root / "trajectories" / lock["history"] / f"seed_{lock['seed']}"
    data = {}
    for arm in ARMS:
        branch = directory / arm
        if (branch / "trajectory.json").exists():
            record = read(branch / "trajectory.json")
            trace_path = branch / "trace.npz"
            complete = True
        else:
            checkpoint = load_checkpoint(branch)
            if checkpoint is None:
                continue
            generation, record = checkpoint
            trace_path, complete = generation / "trace.npz", False
        clocks = sorted(map(int, record["checkpoints"]))
        with np.load(trace_path) as trace:
            accepted, tried = trace["accepted"].copy(), trace["tried"].copy()
            anchors = trace["anchors_before"].copy()
            dev = trace["development_after"].copy()
        n = clocks[-1] - start
        data[arm] = dict(
            record=record,
            clocks=clocks,
            accepted=accepted,
            tried=tried,
            anchors=anchors,
            development=dev,
            n=n,
            complete=complete,
            updates=[int(accepted[: clock - start].sum()) for clock in clocks],
        )
    full = read(Path(lock["inherited_full_reference"]))
    full_clocks = sorted(map(int, full["checkpoints"]))
    summary = dict(
        single_trajectory_each=True,
        history=lock["history"],
        seed=lock["seed"],
        target_offered_each=target,
        new_accept_all=0,
        arms={},
        complete=False,
        original_threshold=read(root / "calibration.json")["threshold"],
        covariance_cap=lock["covariance_cap"],
        development_tolerance_pp=2.0,
    )
    shared = sorted(set.intersection(*(set(row["clocks"]) for row in data.values()))) if len(data) == 3 else []
    common_clock = shared[-1] if shared else None
    text = (
        "# 74 — Corrected calibration and measured protection\n\n"
        f"Exactly three pilot trajectories: history {lock['history']}, seed {lock['seed']}; "
        f"the same complete 576-update parent and {target:,}-image continuation as N73. "
        "No new accept-all trajectory: green references reuse N73's saved full exposure run.\n\n"
        "All three gates refit unchanged delta J after excluding both raw clocks and bounding retained "
        "coordinates to development fit ranges. The covariance variant additionally checks a real candidate "
        "update against 1.25 times warm covariance. The probe variant checks development overall/day/night "
        "accuracy against two-checkpoint-confirmed anchors, with a two-point tolerance. Failed candidates "
        "restore the complete learning state. These are measured hard constraints, not learned risk forecasts.\n\n"
        "The final query bank never controls admissions or anchors. Soft error is logged diagnostically. "
        "Recipes are frozen; this pilot neither establishes long-path calibration nor across-run uncertainty.\n\n"
    )
    if not data:
        text += "Calibration prepared; all-policy GPU preflight and training measurements are pending.\n"
    else:
        text += "| Policy | Offered | Accepted | Rejected trials | Latest accuracy | Max drawdown, pp | Finished |\n"
        text += "|---|---:|---:|---:|---:|---:|---|\n"
        for arm, row in data.items():
            acc = np.array(
                [row["record"]["checkpoints"][str(t)]["metrics"]["linear_accuracy_balanced"] for t in row["clocks"]]
            )
            dd = np.maximum.accumulate(acc) - acc
            summary["arms"][arm] = dict(
                offered=row["n"],
                accepted=int(row["accepted"][: row["n"]].sum()),
                rejected_trials=int(row["tried"][: row["n"]].sum() - row["accepted"][: row["n"]].sum()),
                latest_accuracy=float(acc[-1]),
                max_drawdown_pp=float(dd.max()),
                complete=row["complete"],
                policy_seconds=sum(row["record"]["costs"].values()),
            )
            r = summary["arms"][arm]
            text += (
                f"| {arm} | {row['n']:,} | {r['accepted']:,} | {r['rejected_trials']:,} | "
                f"{acc[-1]:.3f}% | {dd.max():.3f} | {row['complete']} |\n"
            )
        if common_clock is not None:
            summary["shared_offered"] = common_clock - start
            summary["at_shared_clock"] = {
                arm: row["record"]["checkpoints"][str(common_clock)]["metrics"] for arm, row in data.items()
            }
            summary["existing_full_at_shared_clock"] = full["checkpoints"][str(common_clock)]["metrics"]
            text += f"\nAll-three comparison available through **{common_clock - start:,}** offered arrivals.\n"
    summary["complete"] = len(data) == 3 and all(row["complete"] for row in data.values())
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    images = []
    if data:
        for name, keys in (
            (
                "readers",
                ["linear_accuracy_balanced", "linear_ce_balanced", "linear_accuracy_day", "linear_accuracy_night"],
            ),
            ("health", ["selector_rank", "selector_variance", "selector_covariance", "selector_objective"]),
            (
                "development",
                [
                    "development_accuracy_balanced",
                    "development_accuracy_day",
                    "development_accuracy_night",
                    "development_soft_error_balanced",
                ],
            ),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
            for ax, key in zip(axes.flat, keys, strict=True):
                for arm, row in data.items():
                    ax.plot(
                        np.array(row["clocks"]) - start,
                        [row["record"]["checkpoints"][str(t)]["metrics"][key] for t in row["clocks"]],
                        label=arm.replace("_", " "),
                        color=COLORS[arm],
                    )
                if not key.startswith("development"):
                    ax.plot(
                        np.array(full_clocks) - start,
                        [full["checkpoints"][str(t)]["metrics"][key] for t in full_clocks],
                        label="Existing N73 full",
                        color=COLORS["full"],
                        alpha=0.55,
                    )
                if key == "selector_covariance":
                    ax.axhline(
                        lock["covariance_cap"],
                        color=COLORS["corrected_covariance"],
                        ls="--",
                        alpha=0.7,
                        label="Covariance cap",
                    )
                if key.startswith("development_accuracy") and "corrected_development" in data:
                    group = key.removeprefix("development_accuracy_")
                    row = data["corrected_development"]
                    ax.plot(
                        np.array(row["clocks"]) - start,
                        [
                            row["record"]["checkpoints"][str(t)]["metrics"]["development_anchor_" + group] - 2
                            for t in row["clocks"]
                        ],
                        ls="--",
                        color=COLORS["corrected_development"],
                        alpha=0.6,
                        label="Confirmed anchor − 2 pp",
                    )
                ax.set(title=key.replace("_", " "), xlabel="Offered continuation images")
                ax.grid(alpha=0.2)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle("N74: one trajectory per policy; current prefixes may differ")
            path = figures / f"{name}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            images.append(path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for arm, row in data.items():
            axes[0].plot(np.array(row["clocks"]) - start, row["updates"], label=arm, color=COLORS[arm])
            counts = [int(row["tried"][: t - start].sum()) for t in row["clocks"]]
            axes[1].plot(
                np.array(row["clocks"]) - start, np.array(counts) - row["updates"], label=arm, color=COLORS[arm]
            )
        for ax, title in zip(axes, ("Actual accepted singleton updates", "Rejected candidate updates"), strict=True):
            ax.set(title=title, xlabel="Offered continuation images")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        path = figures / "admission.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        images.append(path)
    atomic_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        text + "\n" + "\n".join(f"![{p.stem}](figures/{p.name})\n" for p in images), encoding="utf-8"
    )
    cells = [dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))]
    for path in images:
        cells.append(
            dict(
                cell_type="markdown",
                metadata={},
                source=[f"![{path.stem}](attachment:{path.name})"],
                attachments={path.name: {"image/png": base64.b64encode(path.read_bytes()).decode()}},
            )
        )
    notebook = dict(
        cells=cells,
        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
        nbformat=4,
        nbformat_minor=5,
    )
    atomic_json(root / "74_single_image_gate_protection.ipynb", notebook)
    atomic_json(ROOT / "notebooks/sums/74_single_image_gate_protection.ipynb", notebook)
