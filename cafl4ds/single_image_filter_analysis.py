"""Paired endpoint, cost and live calibration summaries for the admission pilot."""

from __future__ import annotations

import numpy as np

from cafl4ds.single_image_filter import objective
from cafl4ds.single_image_forecast import HEALTH

ARMS = ("image_gate", "state_gate", "random_matched", "full", "hold")


def paired(values, groups, settings):
    names = np.unique(groups)
    choice = np.random.default_rng(settings["bootstrap_seed"]).integers(
        len(names), size=(settings["bootstrap_draws"], len(names))
    )
    sums = np.array([values[groups == g].sum(0) for g in names])
    counts = np.array([(groups == g).sum() for g in names])
    sample = sums[choice].sum(1) / counts[choice].sum(1)[:, None]
    return dict(mean=values.mean(0).tolist(), ci95=np.quantile(sample, [0.025, 0.975], axis=0).T.tolist())


def diagnostic_scores(records, tolerance):
    y, p = [np.array([r[k] for r in records]) for k in ("actual_delta", "prediction")]
    changed = abs(y) > tolerance
    negative, positive = y < -tolerance, y > tolerance
    recall_down = float((p[negative] < 0).mean()) if negative.any() else None
    recall_up = float((p[positive] >= 0).mean()) if positive.any() else None
    return dict(
        n=len(y),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        mae=float(abs(p - y).mean()),
        zero_rmse=float(np.sqrt(np.mean(y**2))),
        zero_mae=float(abs(y).mean()),
        actual_improves_fraction=float(negative.mean()),
        predicted_improves_fraction=float((p < 0).mean()),
        sign_accuracy=float(((p[changed] < 0) == (y[changed] < 0)).mean()) if changed.any() else None,
        balanced_sign_accuracy=(recall_down + recall_up) / 2
        if recall_down is not None and recall_up is not None
        else None,
    )


def summarize(root, read, settings):
    refs = {k: np.array(v) for k, v in read(root / "references.json").items() if k in ("selector", "heldout")}
    collection = read(root / "collection.json")["records"]
    pairs = sorted({(r["history"], r["seed"]) for r in collection})
    assert len(pairs) == 12 and len(collection) == 60
    before, after, panels_before, panels_after, curves, accepted, per_clip, costs, diagnostics, health_ood = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    def vector(directory):
        record = read(directory / "readout.json")
        metric_names = list(record["metrics"][str(settings["reader_draws"][0])])
        panels = np.array([[record["metrics"][str(d)][m] for m in metric_names] for d in settings["reader_draws"]])
        health = np.array([record["metrics"][str(settings["reader_draws"][0])][m] for m in HEALTH])
        heldout = np.array(read(directory / "heldout.json")["values"])
        values = np.r_[
            objective(health, refs["selector"], settings),
            objective(heldout, refs["heldout"], settings),
            heldout,
            panels.mean(0),
        ]
        return (
            values,
            panels,
            ["selector_objective", "heldout_objective"] + ["heldout_" + k for k in HEALTH] + metric_names,
        )

    for history, seed in pairs:
        directory = root / "trajectories" / history / f"seed_{seed}"
        b, pb, metric_names = vector(directory / "common")
        before.append(b)
        panels_before.append(pb)
        ar, pa, cv, ac, pc, co, oo = [], [], [], [], [], [], []
        for arm in ARMS:
            d = directory / arm
            end, panels, _ = vector(d / "endpoint")
            trajectory = read(d / "trajectory.json")
            ar.append(end)
            pa.append(panels)
            cv.append(
                [[b[0], b[1]]] + [[v["objective"], v["heldout_objective"]] for v in trajectory["checkpoints"].values()]
            )
            ac.append(trajectory["accepted"])
            pc.append(
                [
                    sum(r["accepted"] for r in trajectory["records"] if r["origin"] // 96 == clip)
                    for clip in range(settings["start"] // 96, settings["stop"] // 96)
                ]
            )
            co.append(
                [
                    trajectory["costs"][k]
                    for k in ("feature_seconds", "prediction_seconds", "health_seconds", "training_seconds")
                ]
            )
            oo.append(
                float(np.mean([r["outside_health_range"] for r in trajectory["records"]]))
                if arm in ("image_gate", "state_gate")
                else 0.0
            )
            diagnostics.extend(dict(history=history, seed=seed, arm=arm, **r) for r in trajectory["diagnostics"])
        after.append(ar)
        panels_after.append(pa)
        curves.append(cv)
        accepted.append(ac)
        per_clip.append(pc)
        costs.append(co)
        health_ood.append(oo)
    a = dict(
        before=np.array(before),
        after=np.array(after),
        panels_before=np.array(panels_before),
        panels_after=np.array(panels_after),
        curves=np.array(curves),
        accepted=np.array(accepted),
        per_clip=np.array(per_clip),
        costs=np.array(costs),
        health_ood=np.array(health_ood),
        metrics=np.array(metric_names),
        groups=np.array([g for g, _ in pairs]),
        seeds=np.array([s for _, s in pairs]),
        arms=np.array(ARMS),
    )
    summary = dict(
        arms={},
        comparisons={},
        diagnostics={},
        settings=settings,
        source_clusters=6,
        training_seeds=2,
        exploratory=True,
    )
    for i, arm in enumerate(ARMS):
        summary["arms"][arm] = dict(
            accepted=int(a["accepted"][:, i].sum()),
            offered=len(pairs) * (settings["stop"] - settings["start"]),
            admission_rate=float(a["accepted"][:, i].mean() / (settings["stop"] - settings["start"])),
            mean_policy_seconds=float(a["costs"][:, i].sum(1).mean()),
            total_policy_seconds=float(a["costs"][:, i].sum()),
            cost_components=a["costs"][:, i].sum(0).tolist(),
            health_range_violation_fraction=float(a["health_ood"][:, i].mean()),
            metrics={
                m: dict(
                    before=float(a["before"][:, j].mean()),
                    after=float(a["after"][:, i, j].mean()),
                    change=float((a["after"][:, i, j] - a["before"][:, j]).mean()),
                    per_seed_change={
                        str(s): float((a["after"][a["seeds"] == s, i, j] - a["before"][a["seeds"] == s, j]).mean())
                        for s in np.unique(a["seeds"])
                    },
                )
                for j, m in enumerate(metric_names)
            },
        )
        rows = [r for r in diagnostics if r["arm"] == arm]
        if rows:
            summary["diagnostics"][arm] = {
                name: diagnostic_scores(
                    [r for r in rows if mode is None or r["accepted"] == mode], settings["numeric_tolerance"]
                )
                if any(mode is None or r["accepted"] == mode for r in rows)
                else None
                for name, mode in (("all", None), ("accepted", True), ("rejected", False))
            }
    for control in ARMS[1:]:
        j = ARMS.index(control)
        result = paired(a["after"][:, 0] - a["after"][:, j], a["groups"], settings)
        summary["comparisons"]["image_gate__" + control] = {
            m: dict(mean=result["mean"][k], ci95=result["ci95"][k]) for k, m in enumerate(metric_names)
        }
    return a, diagnostics, summary
