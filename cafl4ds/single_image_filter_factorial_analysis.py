"""Paired whole-history summaries for the singleton factorial filter study."""

from __future__ import annotations

import joblib
import numpy as np

from cafl4ds.single_image_filter_factorial import ARMS, MAIN


def clustered_interval(values, groups, draws):
    """Resample whole source histories, preserving every seed/panel/checkpoint."""
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    means = np.array([values[groups == group].mean(0) for group in unique])
    sampled = means[draws].mean(1)
    return dict(mean=np.mean(means, axis=0).tolist(), ci95=np.quantile(sampled, [0.025, 0.975], axis=0).tolist())


def forecast_scores(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    nonzero = np.abs(actual) > 1e-10
    classes = [actual < -1e-10, actual > 1e-10]
    balanced = (
        float(np.mean([np.mean((predicted[take] < 0) == (actual[take] < 0)) for take in classes]))
        if all(t.any() for t in classes)
        else None
    )
    return dict(
        n=len(actual),
        rmse=float(np.sqrt(np.mean((actual - predicted) ** 2))),
        mae=float(np.mean(np.abs(actual - predicted))),
        zero_rmse=float(np.sqrt(np.mean(actual**2))),
        zero_mae=float(np.mean(np.abs(actual))),
        balanced_sign_accuracy=balanced,
        nonzero=int(nonzero.sum()),
        improves_fraction=float(np.mean(actual < -1e-10)),
        mean_actual=float(actual.mean()),
    )


def summarize(root, read, settings):
    roster = read(root / "collection.json")["records"]
    keys = sorted({(r["history"], r["seed"]) for r in roster})
    assert len(keys) == 12 and len(roster) == 12 * len(ARMS)
    common = read(root / "trajectories" / keys[0][0] / f"seed_{keys[0][1]}" / "common/readout.json")
    metrics = list(common["metrics"])
    reader_metrics = list(common["panels"][0])
    clocks = [settings["start"] + settings["checkpoint_stride"] * j for j in range(5)]
    shape = (len(keys), len(ARMS), len(metrics))
    before, after = np.zeros((len(keys), len(metrics))), np.zeros(shape)
    curves = np.full((len(keys), len(ARMS), len(clocks), len(metrics)), np.nan)
    panel_before = np.zeros((len(keys), 3, len(reader_metrics)))
    panel_after = np.zeros((len(keys), len(ARMS), 3, len(reader_metrics)))
    accepted = np.zeros((len(keys), len(ARMS)), int)
    accepted_curves = np.zeros((len(keys), len(ARMS), len(clocks)), int)
    costs = np.zeros((len(keys), len(ARMS), 4))
    groups, seeds, diagnostics = [], [], []
    models = joblib.load(root / "forecasters.joblib")
    range_violations = {arm: [] for arm in (*MAIN, "J_state", "probe_state")}
    for i, (history, seed) in enumerate(keys):
        groups.append(history)
        seeds.append(seed)
        directory = root / "trajectories" / history / f"seed_{seed}"
        record = read(directory / "common/readout.json")
        before[i] = [record["metrics"][m] for m in metrics]
        panel_before[i] = [[p[m] for m in reader_metrics] for p in record["panels"]]
        for j, arm in enumerate(ARMS):
            tr = read(directory / arm / "trajectory.json")
            if arm in range_violations:
                inputs = np.array([r["features"] for r in tr["records"]])
                forecaster = models[arm][seed]
                health_outside = np.any(
                    (inputs[:, :14] < forecaster.minimum[:14]) | (inputs[:, :14] > forecaster.maximum[:14]), axis=1
                )
                range_violations[arm].append([health_outside.mean(), forecaster.outside(inputs).mean()])
            accepted[i, j] = tr["accepted"]
            costs[i, j] = [tr["costs"][name] for name in ("features", "prediction", "observations", "training")]
            endpoint = tr["checkpoints"][str(settings["stop"])]
            after[i, j] = [endpoint["metrics"][m] for m in metrics]
            panel_after[i, j] = [[p[m] for m in reader_metrics] for p in endpoint["panels"]]
            for c, clock in enumerate(clocks):
                rec = tr["checkpoints"][str(clock)]
                curves[i, j, c] = [rec["metrics"].get(m, np.nan) for m in metrics]
                accepted_curves[i, j, c] = sum(r["accepted"] for r in tr["records"] if r["origin"] < clock)
            for diagnostic in tr["diagnostics"]:
                measured = diagnostic["measures"]
                row = dict(
                    history=history,
                    seed=seed,
                    arm=arm,
                    origin=diagnostic["origin"],
                    accepted=diagnostic["accepted"],
                    predictions=diagnostic["predictions"],
                )
                for name in measured["before"]:
                    row["H1_" + name] = measured["H1"][name] - measured["before"][name]
                    row["H32_" + name] = measured["H32_accept"][name] - measured["H32_skip"][name]
                diagnostics.append(row)
    groups, seeds = np.array(groups), np.array(seeds)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = rng.integers(0, len(np.unique(groups)), size=(settings["bootstrap_draws"], len(np.unique(groups))))

    def interval_dict(values):
        record = clustered_interval(values, groups, draws)
        means, bounds = np.array(record["mean"]), np.array(record["ci95"])
        return {m: dict(mean=float(means[k]), ci95=bounds[:, k].tolist()) for k, m in enumerate(metrics)}

    summary = dict(
        arms={},
        comparisons={},
        factorial={},
        forecast={},
        persistence={},
        curves={},
        comparison_curves={},
        settings=settings,
        clusters=6,
        seeds=2,
        reader_panels=3,
        exploratory=True,
    )
    for j, arm in enumerate(ARMS):
        changes = after[:, j] - before
        summary["arms"][arm] = dict(
            accepted=int(accepted[:, j].sum()),
            offered=len(keys) * (settings["stop"] - settings["start"]),
            admission=float(accepted[:, j].mean() / (settings["stop"] - settings["start"])),
            mean_policy_seconds=float(costs[:, j].sum(1).mean()),
            mean_costs=dict(
                zip(("features", "prediction", "observations", "training"), costs[:, j].mean(0).tolist(), strict=True)
            ),
            metrics={
                m: dict(
                    before=float(before[:, k].mean()),
                    after=float(after[:, j, k].mean()),
                    change=float(changes[:, k].mean()),
                )
                for k, m in enumerate(metrics)
            },
            changes=interval_dict(changes),
        )
        if arm in range_violations:
            summary["arms"][arm]["health_history_range_violation_fraction"] = float(
                np.mean(range_violations[arm], axis=0)[0]
            )
            summary["arms"][arm]["any_feature_range_violation_fraction"] = float(
                np.mean(range_violations[arm], axis=0)[1]
            )
        summary["curves"][arm] = {}
        for m in (
            "linear_accuracy_balanced",
            "linear_ce_balanced",
            "knn_accuracy_balanced",
            "knn_ce_balanced",
            "evaluation_objective",
            "selector_objective",
            "development_probe_ce",
        ):
            summary["curves"][arm][m] = clustered_interval(curves[:, j, :, metrics.index(m)], groups, draws)
    for arm in MAIN:
        j = ARMS.index(arm)
        random_index = ARMS.index("random_" + arm)
        summary["comparison_curves"][arm] = {
            m: clustered_interval(
                curves[:, j, :, metrics.index(m)] - curves[:, random_index, :, metrics.index(m)], groups, draws
            )
            for m in ("linear_accuracy_balanced", "linear_ce_balanced")
        }
        for control in ("random_" + arm, "full", "hold", arm.split("_")[0] + "_state"):
            summary["comparisons"][arm + "__" + control] = interval_dict(after[:, j] - after[:, ARMS.index(control)])
        rows = [r for r in diagnostics if r["arm"] == arm]
        summary["forecast"][arm] = {}
        for target in ("J", "probe"):
            for subset, take in (
                ("all", rows),
                ("accepted", [r for r in rows if r["accepted"]]),
                ("rejected", [r for r in rows if not r["accepted"]]),
            ):
                summary["forecast"][arm][target + "_" + subset] = (
                    forecast_scores([r["H1_" + target] for r in take], [r["predictions"][target] for r in take])
                    if take
                    else None
                )
        summary["persistence"][arm] = {}
        for metric in ("J", "probe", "evaluation_probe_ce", "evaluation_probe_accuracy"):
            for horizon in ("H1", "H32"):
                values = np.array([r[horizon + "_" + metric] for r in rows])
                row_groups = np.array([r["history"] for r in rows])
                summary["persistence"][arm][horizon + "_" + metric] = clustered_interval(values, row_groups, draws)
    contrasts = dict(
        features_at_J=("J_rich", "J_base"),
        features_at_probe=("probe_rich", "probe_base"),
        objective_at_base=("probe_base", "J_base"),
        objective_at_rich=("probe_rich", "J_rich"),
    )
    for name, (a, b) in contrasts.items():
        raw = after[:, ARMS.index(a)] - after[:, ARMS.index(b)]
        content = raw - (after[:, ARMS.index("random_" + a)] - after[:, ARMS.index("random_" + b)])
        summary["factorial"][name] = dict(raw=interval_dict(raw), above_matched_random=interval_dict(content))
    interaction = (
        after[:, ARMS.index("probe_rich")]
        - after[:, ARMS.index("probe_base")]
        - after[:, ARMS.index("J_rich")]
        + after[:, ARMS.index("J_base")]
    )
    random_interaction = (
        after[:, ARMS.index("random_probe_rich")]
        - after[:, ARMS.index("random_probe_base")]
        - after[:, ARMS.index("random_J_rich")]
        + after[:, ARMS.index("random_J_base")]
    )
    summary["factorial"]["objective_feature_interaction"] = dict(
        raw=interval_dict(interaction), above_matched_random=interval_dict(interaction - random_interaction)
    )
    arrays = dict(
        before=before,
        after=after,
        curves=curves,
        panels_before=panel_before,
        panels_after=panel_after,
        accepted=accepted,
        accepted_curves=accepted_curves,
        costs=costs,
        groups=groups,
        seeds=seeds,
        arms=np.array(ARMS),
        metrics=np.array(metrics),
        reader_metrics=np.array(reader_metrics),
        clocks=np.array(clocks),
    )
    return arrays, summary, diagnostics
