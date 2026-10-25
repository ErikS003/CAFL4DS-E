"""Paired whole-history summaries for the singleton factorial filter study."""

from __future__ import annotations

import joblib
import numpy as np
from scipy.stats import spearmanr

from cafl4ds.single_image_filtered_paths import predict, probe_scores
from cafl4ds.single_image_state_control import ARMS, MAIN, TARGETS, DESIGNS


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
    same = np.where(np.abs(predicted) <= 1e-10, 0, np.sign(predicted)) == np.sign(actual)
    balanced = float(np.mean([same[take].mean() for take in classes])) if all(t.any() for t in classes) else None
    return dict(
        n=len(actual),
        rmse=float(np.sqrt(np.mean((actual - predicted) ** 2))),
        mae=float(np.abs(actual - predicted).mean()),
        zero_rmse=float(np.sqrt(np.mean(actual**2))),
        zero_mae=float(np.abs(actual).mean()),
        balanced_sign_accuracy=balanced,
        nonzero=int(nonzero.sum()),
        zero=int((~nonzero).sum()),
        positive=int(classes[1].sum()),
        negative=int(classes[0].sum()),
        predicted_zero=int((np.abs(predicted) < 1e-10).sum()),
        mean_actual=float(actual.mean()),
    )


def error_skill(actual, better, baseline, groups, draws):
    """Paired reduction in error on a common history-clustered population."""
    actual, better, baseline, groups = map(np.asarray, (actual, better, baseline, groups))
    by_history = np.array(
        [
            [
                np.mean((actual[groups == g] - better[groups == g]) ** 2),
                np.mean((actual[groups == g] - baseline[groups == g]) ** 2),
                np.mean(np.abs(actual[groups == g] - better[groups == g])),
                np.mean(np.abs(actual[groups == g] - baseline[groups == g])),
            ]
            for g in np.unique(groups)
        ]
    )
    sampled = by_history[draws].mean(1)
    all_mean = by_history.mean(0)
    return dict(
        rmse=dict(
            mean=float(np.sqrt(all_mean[1]) - np.sqrt(all_mean[0])),
            ci95=np.quantile(np.sqrt(sampled[:, 1]) - np.sqrt(sampled[:, 0]), [0.025, 0.975]).tolist(),
        ),
        mae=dict(
            mean=float(all_mean[3] - all_mean[2]),
            ci95=np.quantile(sampled[:, 3] - sampled[:, 2], [0.025, 0.975]).tolist(),
        ),
    )


def target_alignment(a, settings):
    """Measured target association, explicitly distinct from forecasting."""
    out = {}
    held = np.isin(a["histories"], settings["validation_histories"])
    for recipe in ("full", "mixed"):
        for role, is_role in (("fit", ~held), ("validation", held)):
            take = (a["recipes"] == recipe) & is_role
            y = a["targets"][take, 0]
            for j, target in enumerate(TARGETS):
                x = a["targets"][take, j]
                usable = (np.abs(y) > 1e-10) & (np.abs(x) > 1e-10)
                both_vary = np.std(x) > 1e-12 and np.std(y) > 1e-12
                out[f"{recipe}/{role}/{target}"] = dict(
                    n=int(take.sum()),
                    actual_zero=int((np.abs(y) <= 1e-10).sum()),
                    pearson=float(np.corrcoef(x, y)[0, 1]) if both_vary else None,
                    spearman=float(spearmanr(x, y).statistic) if both_vary else None,
                    sign_agreement=float(np.mean(np.sign(x[usable]) == np.sign(y[usable]))) if usable.any() else None,
                    sign_rows=int(usable.sum()),
                )
    return out


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
    models = {
        k: v
        for k, v in joblib.load(root / "forecasters.joblib").items()
        if k in ("full/base/accuracy", "full/state/accuracy")
    }
    selection = read(root / "model-selection.json")
    development = __import__("torch").load(root / "intake/development-panel.pt", weights_only=True)
    labels = development["query_labels"].numpy()
    range_violations = {arm: [] for arm in MAIN}
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
                forecaster = models[selection["gates"][arm]["key"]][seed]["regressor"]
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
                raw = np.load(directory / arm / "diagnostics" / f"{diagnostic['origin']}.npz")
                scores = {
                    name: probe_scores(
                        raw[name + "_development_probabilities"], labels, settings["soft_accuracy_temperature"]
                    )
                    for name in ("before", "H1", "H32_accept", "H32_skip")
                }
                for ti, target in enumerate(TARGETS):
                    row["H1_" + target] = float(scores["H1"][ti] - scores["before"][ti])
                    row["H32_" + target] = float(scores["H32_accept"][ti] - scores["H32_skip"][ti])
                x = np.asarray(tr["records"][diagnostic["origin"] - settings["start"]]["audit_vectors"]["base"])
                row["all_predictions"] = {
                    key: float(predict(by_seed[seed], x[:23] if "/state/" in key else x)[0])
                    for key, by_seed in models.items()
                }
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
        for control in ("random_" + arm, "full", "hold"):
            summary["comparisons"][arm + "__" + control] = interval_dict(after[:, j] - after[:, ARMS.index(control)])
        rows = [r for r in diagnostics if r["arm"] == arm]
        summary["forecast"][arm] = {}
        for design in DESIGNS:
            for target in TARGETS:
                key = f"full/{design}/{target}"
                summary["forecast"][arm][design + "/" + target] = forecast_scores(
                    [r["H1_accuracy"] for r in rows], [r["all_predictions"][key] for r in rows]
                )
        summary["persistence"][arm] = {}
        for metric in (
            "J",
            "probe",
            "accuracy",
            "evaluation_probe_ce",
            "evaluation_probe_accuracy",
        ):
            for horizon in ("H1", "H32"):
                values = np.array([r[horizon + "_" + metric] for r in rows])
                row_groups = np.array([r["history"] for r in rows])
                summary["persistence"][arm][horizon + "_" + metric] = clustered_interval(values, row_groups, draws)
    contrasts = dict(image_vs_state=("image", "state"))
    for name, (a, b) in contrasts.items():
        raw = after[:, ARMS.index(a)] - after[:, ARMS.index(b)]
        content = raw - (after[:, ARMS.index("random_" + a)] - after[:, ARMS.index("random_" + b)])
        summary["factorial"][name] = dict(raw=interval_dict(raw), above_matched_random=interval_dict(content))
    summary["common_parent_forecast"] = {}
    values = np.array([r["H1_accuracy"] for r in diagnostics])
    row_groups = np.array([r["history"] for r in diagnostics])
    predictions = {key: np.array([r["all_predictions"][key] for r in diagnostics]) for key in models}
    for key in models:
        predicted = predictions[key]
        result = forecast_scores(values, predicted)
        row_groups = np.array([r["history"] for r in diagnostics])
        result["mae_gain_over_zero"] = clustered_interval(
            np.abs(values) - np.abs(values - predicted), row_groups, draws
        )
        summary["common_parent_forecast"][key] = result
    summary["forecast_skill"] = {}
    for first, second in (("base", "state"), ("base", "zero"), ("state", "zero")):
        baseline = predictions["full/" + second + "/accuracy"] if second != "zero" else np.zeros_like(values)
        summary["forecast_skill"][first + "_minus_" + second] = error_skill(
            values, predictions["full/" + first + "/accuracy"], baseline, row_groups, draws
        )
    summary["selection"] = {arm: selection["gates"][arm] for arm in MAIN}
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
