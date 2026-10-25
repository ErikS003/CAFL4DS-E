"""Rebuild cached diagnostic geometry, raw scores and excluded-history regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.linalg import lstsq
from scipy.stats import t
from threadpoolctl import threadpool_limits

from scripts.run_dnd_signal_diagnostic import CONFIG, ROOT, check, digest, read, write


def require(condition: bool, message: str) -> None:
    """Raise on an independent reconstruction mismatch."""
    if not condition:
        raise ValueError(message)


def geometry(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Reconstruct the three instruments using independent matrix expressions."""
    rank, variance = [], []
    for array in (reference.astype(float), current.astype(float)):
        singular = np.linalg.svd(array, compute_uv=False)
        probability = singular / max(singular.sum(), 1e-12) + 1e-7
        rank.append(np.exp(-(probability * np.log(probability)).sum()))
        unit = array / np.maximum(np.sqrt((array**2).sum(1, keepdims=True)), 1e-12)
        variance.append(np.cov(unit, rowvar=False).diagonal().mean())
    a, b = [array.astype(float) - array.mean(0, dtype=float) for array in (reference, current)]
    cka = np.linalg.norm(a.T @ b) ** 2 / max(np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b), 1e-30)
    return np.array([1 - rank[1] / rank[0], 1 - variance[1] / variance[0], max(0, 1 - cka)])


def same_interval(record: dict, values: np.ndarray) -> None:
    """Recompute source units, means and t intervals."""
    mean = values.mean(0)
    half = t.ppf(0.975, 7) * np.std(values, axis=0, ddof=1) / np.sqrt(8)
    np.testing.assert_allclose(record["history_values"], values, atol=1e-10)
    np.testing.assert_allclose(record["mean"], mean, atol=1e-10)
    np.testing.assert_allclose(record["ci95"], np.stack([mean - half, mean + half], axis=-1), atol=1e-10)


def verify_warning(record: dict, truth: np.ndarray, weights: np.ndarray) -> None:
    """Rebuild warning benefits and positive learning costs without cancellation."""
    count = float(weights.sum())
    harm = truth[..., 0] >= 1
    learning = (truth[..., 1] > 0.5) | (truth[..., 2] > 1) | (truth[..., 3] > 1)
    np.testing.assert_allclose(record["warning_count"], count, atol=1e-10)
    np.testing.assert_allclose(record["warning_fraction"], count / weights.size, atol=1e-10)
    for key, event, denominator in (
        ("harm_precision", harm, count),
        ("harm_recall", harm, harm.sum()),
        ("useful_hold_precision", harm & ~learning, count),
        ("material_learning_fraction", learning, count),
    ):
        if denominator:
            np.testing.assert_allclose(record[key], (weights * event).sum() / denominator, atol=1e-10)
        else:
            require(record[key] is None, f"Undefined statistic must be null: {key}")
    if count:
        np.testing.assert_allclose(
            record["mean_targets_when_warned_pp"], np.sum(truth * weights[..., None], axis=(0, 1)) / count, atol=1e-10
        )
        np.testing.assert_allclose(
            record["mean_positive_balanced_gain_when_warned_pp"],
            (np.clip(truth[..., 1], 0, None) * weights).sum() / count,
            atol=1e-10,
        )
    else:
        require(record["mean_targets_when_warned_pp"] is None, "No warning has no conditional outcome")
        require(record["mean_positive_balanced_gain_when_warned_pp"] is None, "No warning has no learning cost")
    np.testing.assert_allclose(record["protected_loss_per_decision_pp"], (truth[..., 0] * weights).mean(1), atol=1e-10)
    np.testing.assert_allclose(
        record["forgone_positive_learning_per_decision_pp"],
        (np.clip(truth[..., 1], 0, None) * weights).mean(1),
        atol=1e-10,
    )


def main() -> None:  # noqa: C901 - independent checks share the same frozen inputs
    """Audit every source, horizon, fold and warning allocation."""
    config = OmegaConf.load(CONFIG)
    out, source = ROOT / config.output, ROOT / config.source
    require(not (out / "independent-verification.json").exists(), "Preserve the first verification")
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    records = read(source / "collection.json")["records"]
    summary = read(out / "summary.json")
    folds = read(out / "folds.json")["fits"]
    monitor = [132, *range(140, 445, 8)]
    points = sorted({0, *monitor, 448, 148, 264, 160, 192, 224, 256, 288, 320})
    with np.load(ROOT / config.evaluation / "data.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    saved = np.load(out / "derived_windows.npz")
    predictions = np.load(out / "predictions.npz")
    feature_rows, alarm_rows, scores = [], [], []
    raw_files = set()
    for record in records:
        with np.load(source / record["name"] / "embeddings.npz") as a:
            levels = np.array([geometry(a["132"], a[str(point)]) for point in monitor])
        feature = np.zeros((len(monitor), 6))
        feature[:, ::2] = levels
        feature[1:, 1::2] = levels[1:] - levels[:-1]
        flag = levels >= 0.1 - 1e-12
        flag[0] = False
        alarms = np.zeros((len(monitor), 4), bool)
        for i in range(1, len(monitor)):
            alarms[i, :3] = flag[i] & flag[i - 1]
            alarms[i, 3] = flag[i].any() and flag[i - 1].any()
        feature_rows.append(feature)
        alarm_rows.append(alarms)
        panel_curve = []
        for point in points:
            observation = record["observations"][str(point)]
            panel = []
            for draw in config.draws:
                path = Path(observation["readout"]) / f"panel_{draw}.npz"
                metadata = read(path.with_suffix(".json"))
                require(digest(path) == metadata["sha256"], "Changed classifier probabilities")
                require(metadata["encoder_fingerprint"] == observation["encoder_fingerprint"], "Readout encoder")
                require(metadata["learner_fingerprint"] == observation["learner_fingerprint"], "Readout learner")
                with np.load(path) as a:
                    predicted = a["probabilities"][:3663].argmax(1)
                accuracy = []
                for domain in (0, 1):
                    cells = [np.mean(predicted[(domains == domain) & (labels == cls)] == cls) for cls in (0, 1)]
                    accuracy.append(100 * np.mean(cells))
                panel.append([np.mean(accuracy), *accuracy])
                raw_files.add(str(path))
            np.testing.assert_allclose(panel, observation["panel_accuracy_pp"], atol=1e-10)
            panel_curve.append(panel)
        scores.append(panel_curve)
    features, alarms, panels = np.array(feature_rows), np.array(alarm_rows), np.array(scores)
    means = panels.mean(2)
    np.testing.assert_allclose(saved["features_all"], features, atol=1e-10)
    np.testing.assert_array_equal(saved["alarms_all"], alarms)
    np.testing.assert_allclose(saved["panel_curve"], panels, atol=1e-10)
    np.testing.assert_allclose(saved["mean_curve"], means, atol=1e-10)
    names = ("constant", "age", "rank", "variance", "cka", "all")
    fits_checked, decision_count = 0, 0
    candidates = []
    for horizon in config.horizons:
        steps = [point for point in monitor if point >= 140 and point + horizon in monitor]
        np.testing.assert_array_equal(saved[f"steps_{horizon}"], steps)
        ids = [monitor.index(point) for point in steps]
        truth, support = [], []
        for i in range(8):
            rows, panel_rows = [], []
            for point in steps:
                a, b = points.index(point), points.index(point + horizon)
                risk = max(0, *(means[i, a, d] - means[i, k, d] for k in range(a + 1, b + 1) for d in (1, 2)))
                rows.append([risk, *(means[i, b] - means[i, a])])
                panel_rows.append(
                    [
                        [
                            max(
                                0,
                                *(panels[i, a, j, d] - panels[i, k, j, d] for k in range(a + 1, b + 1) for d in (1, 2)),
                            ),
                            *(panels[i, b, j] - panels[i, a, j]),
                        ]
                        for j in range(3)
                    ]
                )
            truth.append(rows)
            support.append(panel_rows)
        truth, support = np.array(truth), np.array(support)
        fx, alarm = features[:, ids], alarms[:, ids]
        for key, expected in (("targets", truth), ("support_targets", support), ("features", fx)):
            np.testing.assert_allclose(saved[f"{key}_{horizon}"], expected, atol=1e-10)
        np.testing.assert_array_equal(saved[f"alarms_{horizon}"], alarm)
        pred = np.empty((6, *truth.shape))
        for excluded in range(8):
            train = [i for i in range(8) if i != excluded]
            mx, my = fx[train].mean(0), truth[train].mean(0)
            pred[0, excluded] = truth[train].mean((0, 1))
            pred[1, excluded] = my
            for j, columns in enumerate(([0, 1], [2, 3], [4, 5], list(range(6))), 2):
                centered = fx[train][:, :, columns] - mx[:, columns]
                flat = centered.reshape(-1, len(columns))
                scale = flat.std(0)
                scale[scale <= 1e-12] = 1
                z = flat / scale
                residual = (truth[train] - my).reshape(-1, 4)
                augmented = np.vstack([z, np.sqrt(len(z)) * np.eye(len(columns))])
                response = np.vstack([residual, np.zeros((len(columns), 4))])
                coef = lstsq(augmented, response)[0]
                pred[j, excluded] = my + ((fx[excluded][:, columns] - mx[:, columns]) / scale) @ coef
                fold = next(
                    row
                    for row in folds
                    if row["horizon"] == horizon and row["heldout"] == excluded and row["model"] == names[j]
                )
                require(fold["train"] == train and excluded not in fold["train"], "History leakage")
                np.testing.assert_array_equal(fold["columns"], columns)
                for key, expected in (
                    ("feature_mean_by_clock", mx[:, columns]),
                    ("outcome_mean_by_clock", my),
                    ("scale", scale),
                    ("coef", coef),
                ):
                    np.testing.assert_allclose(fold[key], expected, atol=1e-9)
                fits_checked += 1
        pred[..., 0] = np.clip(pred[..., 0], 0, None)
        np.testing.assert_allclose(predictions[f"predictions_{horizon}"], pred, atol=1e-9)
        result = summary["horizons"][str(horizon)]
        rmse = np.sqrt(np.mean((pred - truth[None]) ** 2, axis=2))
        mae = np.mean(np.abs(pred - truth[None]), axis=2)
        require(
            result["total_decisions"] == 8 * len(steps) and result["decisions_per_history"] == len(steps), "Grid count"
        )
        decision_count += 8 * len(steps)
        np.testing.assert_allclose(result["harm_prevalence"], (truth[..., 0] >= 1).mean())
        panel_harm = support[..., 0] >= 1
        np.testing.assert_allclose(
            result["support_draw_harm_disagreement_fraction"], np.any(panel_harm != panel_harm[..., :1], axis=-1).mean()
        )
        for j, name in enumerate(names):
            model = result["models"][name]
            same_interval(model["rmse_pp"], rmse[j])
            same_interval(model["mae_pp"], mae[j])
            difference = rmse[1, :, 0] - rmse[j, :, 0]
            same_interval(model["risk_rmse_improvement_vs_age"], difference)
            relative = difference.mean() / max(rmse[1, :, 0].mean(), 1e-12)
            np.testing.assert_allclose(model["fractional_risk_rmse_improvement"], relative, atol=1e-10)
            count = int((difference > 0).sum())
            require(model["histories_improved"] == count, "Consistency count")
            low = difference.mean() - t.ppf(0.975, 7) * difference.std(ddof=1) / np.sqrt(8)
            screen = bool(j >= 2 and relative >= 0.1 and low > 0 and count >= 6)
            require(model["exploratory_screen"] == screen, "Forecast screen")
            if screen and horizon == 32:
                candidates.append(name)
            verify_warning(model["risk_warning"], truth, pred[j, ..., 0] >= 1)
        for j, instrument in enumerate(("rank", "variance", "cka", "combined")):
            result_alarm = result["alarms"][instrument]
            mask = alarm[..., j]
            age = np.zeros_like(mask)
            for i in range(8):
                order = np.argsort(-pred[1, i, :, 0], kind="stable")
                age[i, order[: int(mask[i].sum())]] = True
            np.testing.assert_array_equal(age.sum(1), mask.sum(1))
            np.testing.assert_array_equal(result_alarm["warning_counts_by_history"], mask.sum(1))
            np.testing.assert_array_equal(predictions[f"matched_age_{horizon}_{instrument}"], age)
            random = np.repeat(mask.mean(1)[:, None], len(steps), axis=1)
            for key, weights in (("actual", mask), ("matched_age", age), ("matched_uniform_expected", random)):
                verify_warning(result_alarm[key], truth, weights)
            same_interval(
                result_alarm["protected_loss_advantage_vs_age"], (truth[..., 0] * (mask.astype(float) - age)).mean(1)
            )
            same_interval(
                result_alarm["positive_learning_cost_vs_age"],
                (np.clip(truth[..., 1], 0, None) * (mask.astype(float) - age)).mean(1),
            )
        for row in [r for r in summary["actual_first_alarm_outcomes"] if r["horizon"] == horizon]:
            i = summary["history_names"].index(row["history"])
            require(row["stop"] == records[i]["stop"], "Actual stop identity")
            require(row["full_horizon_available"] == (row["stop"] in steps), "Actual alarm censoring")
            if row["full_horizon_available"]:
                np.testing.assert_allclose(row["targets_pp"], truth[i, steps.index(row["stop"])], atol=1e-10)
            else:
                require(row["targets_pp"] is None, "Censored outcome must be null")
    require(candidates == summary["primary_candidates"], "Primary candidate list")
    require(fits_checked == len(folds) == summary["statistical_model_fits"] == 96, "Statistical fit count")
    require(summary["new_ssl_updates"] == summary["new_classifier_fits"] == 0, "No learner work authorized here")
    require(summary["exploratory_reuse"] and not summary["controller_qualified"], "Exploratory scope")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "source_sha256": digest(Path(__file__)),
            "raw_classifier_files": len(raw_files),
            "history_horizon_decisions": decision_count,
            "geometry_states": 8 * len(monitor),
            "regressions_rebuilt_with_augmented_lstsq": fits_checked,
            "warning_counts_matched": True,
            "censoring_verified": True,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
        },
    )
    print({"independent_verification": "passed", "regressions": fits_checked, "decisions": decision_count}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
