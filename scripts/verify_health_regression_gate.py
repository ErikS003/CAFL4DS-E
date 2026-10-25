"""Independent CPU reconstruction and sklearn-SVD audit of the regression gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import sem, t
from sklearn.linear_model import Ridge
import torch

from cafl4ds.artifacts import atomic_json
from scripts.verify_prediction_pilots import torch_geometry

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(a, b):
    if not np.allclose(a, b, atol=1e-9, rtol=1e-8):
        raise ValueError("Independent health regression audit differs")


def regression(x, y, query, saved):
    """SVD fitting is independent of the collector's normal-equation solver."""
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    model = Ridge(alpha=10, fit_intercept=True, solver="svd").fit((x - center) / scale, y)
    check(center, saved["center"])
    check(scale, saved["scale"])
    check(model.coef_.T, saved["beta"])
    check(model.intercept_, saved["intercept"])
    if saved["rows"] != len(x):
        raise ValueError("Fit row count differs")
    return model.predict((query - center) / scale)


def check_interval(values, saved, family=16):
    mean = float(np.mean(values))
    half = float(t.isf(0.05 / (2 * family), 5) * sem(values))
    check(mean, saved["mean"])
    check([mean - half, mean + half], saved["ci95"])
    check(values, saved["source_values"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    torch.set_num_threads(4)
    lock, proof = read(out / "input_lock.json"), read(out / "verification.json")
    for p, h in {**lock["hashes"], **proof["hashes"]}.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError("Frozen data/source changed")
    data, fitted, summary = (read(out / n) for n in ("data.json", "regressions.json", "summary.json"))
    rows = [
        r
        for r in read(ROOT / "outputs/health-warning/20261013-gpu/collection.json")["records"]
        if r["role"] == "six_stream"
    ]
    lookup = {(r["stream"], r["checkpoint"], r["seed"]): r for r in rows}
    expected = {
        (f"fresh_{g:02}", checkpoint, seed)
        for g in range(6)
        for checkpoint in ("before_shift", "stable_second", "before_return")
        for seed in (5003, 5009, 5011)
    }
    if len(data["identities"]) != 54 or set(lookup) != expected:
        raise ValueError("Cohort grid differs")
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    x, r, y = (np.array(data[k]) for k in ("x", "health_response", "utility"))
    groups = np.array(data["groups"])
    unique = sorted(set(groups))
    if len(unique) != 6:
        raise ValueError("Six source histories required")
    for index, identity in enumerate(data["identities"]):
        row = lookup[identity["stream"], identity["checkpoint"], identity["seed"]]
        if identity["stream"] != groups[index]:
            raise ValueError("Group identity differs")
        with np.load(row["embeddings_path"]) as embeddings:
            h = np.array(
                [list(torch_geometry(embeddings[f"at_{v}"], embeddings["at_0"]).values()) for v in (0, 4, 8, 16)]
            )
        check(h, data["health"][index])
        scale = np.array([*h[2, :3], 1.0, 1.0])
        expected_x = np.concatenate(
            (
                [row["parent_updates"] + 8, np.mean([v[0] for v in row["trace"][4:8]])],
                h[2],
                (h[2] - h[1]) / scale,
                (h[1] - h[0]) / scale,
            )
        )
        check(expected_x, x[index])
        check((h[3] - h[2]) / scale, r[index])
        outcomes = {}
        for landmark in (8, 16):
            directory = Path(row["landmarks"][str(landmark)]["readout"])
            with np.load(directory / "predictions.npz") as p:
                outcomes[landmark] = {
                    reader: np.array(
                        [
                            [
                                100 * np.mean((p[f"{draw}_{reader}"] == labels)[mask])
                                for mask in (np.ones(len(labels), bool), domains == 0, domains == 1)
                            ]
                            for draw in (51011, 51017, 51023)
                        ]
                    )
                    for reader in ("finetune", "linear", "knn")
                }
        panel = outcomes[16]["finetune"] - outcomes[8]["finetune"]
        check(panel, data["panel_utility"][index])
        check(panel.mean(0), y[index])
        for reader in ("linear", "knn"):
            check((outcomes[16][reader] - outcomes[8][reader]).mean(0), data["reader_utility"][index][reader])
    # Independently fit every outer and nested model, preserving group masks.
    predicted = {name: np.zeros_like(y) for name in fitted["predictions"]}
    health_prediction = np.zeros_like(r)
    fit_count = 0
    for g in unique:
        fold = fitted["folds"][str(g)]
        train, held = groups != g, groups == g
        if fold["excluded_group"] != g or set(fold["fit_groups"]) != set(unique) - {g}:
            raise ValueError("Outer source exclusion differs")
        hr = regression(x[train], r[train], x[held], fold["outer_health"])
        health_prediction[held] = hr
        generated = np.zeros_like(r)
        fit_count += 1
        for k in unique:
            if k == g:
                continue
            allowed = (groups != g) & (groups != k)
            inner = fold["inner_health"][str(k)]
            if set(inner["excluded_groups"]) != {g, k}:
                raise ValueError("Inner exclusions differ")
            generated[groups == k] = regression(x[allowed], r[allowed], x[groups == k], inner["model"])
            fit_count += 1
        check(generated[train], fold["generated_training_health"])
        fit_design = {
            "cheap": x[:, :2],
            "history": x,
            "measured": np.column_stack((x, r)),
            "predicted": np.column_stack((x, generated)),
            "response_only": r,
            "measured_interaction": np.column_stack((x, r, x[:, 2:7] * r)),
        }
        for name, design in fit_design.items():
            query = np.column_stack((x[held], hr)) if name == "predicted" else design[held]
            predicted[name][held] = regression(design[train], y[train], query, fold["utility_models"][name])
            fit_count += 1
        predicted["mean"][held] = y[train].mean(0)
        saved = fold["utility_models"]["measured"]
        predicted["plugin"][held] = np.array(saved["intercept"]) + (
            (np.column_stack((x[held], hr)) - saved["center"]) / saved["scale"]
        ) @ np.array(saved["beta"])
    for name, values in predicted.items():
        check(values, fitted["predictions"][name])
    check(health_prediction, fitted["predicted_health"])
    if fit_count != 72:
        raise ValueError("Scientific fit count differs")
    errors = {}
    for name, values in {**predicted, "zero": np.zeros_like(y)}.items():
        errors[name] = {g: np.mean((values[groups == g, 0] - y[groups == g, 0]) ** 2) for g in unique}
        check([errors[name][g] for g in unique], [summary["scores"][name]["source_mse"][g] for g in unique])
        check(np.mean(list(errors[name].values())), summary["scores"][name]["mse_pp2"])
    stage_pass = {}
    for model in ("measured", "predicted"):
        passed = []
        for baseline in ("mean", "history"):
            values = [errors[baseline][g] - errors[model][g] for g in unique]
            c = summary["comparisons"][model][baseline]
            reduction = 1 - np.mean(list(errors[model].values())) / np.mean(list(errors[baseline].values()))
            check(reduction, c["reduction_fraction"])
            check_interval(values, c["adjusted"])
            check_interval(values, c["ordinary"], 1)
            expected_pass = reduction >= 0.1 and c["adjusted"]["ci95"][0] > 0
            if bool(expected_pass) != c["passed"]:
                raise ValueError("MSE gate differs")
            passed.append(expected_pass)
        stage_pass[model] = all(passed)

    def choose(p, threshold=0.25):
        return (p[:, 0] >= threshold) & (np.min(p[:, 1:], axis=1) >= -1)

    choices = {name: choose(predicted[name]) for name in ("predicted", "history", "measured")}
    utility = {name: choice[:, None] * y for name, choice in choices.items()}
    utility.update({"accept_all": y, "hold_all": np.zeros_like(y)})
    random = np.zeros_like(y)
    for g in unique:
        selected = groups == g
        random[selected] = choices["predicted"][selected].mean() * y[selected].mean(0)
    utility["matched_random_expectation"] = random
    for name, value in utility.items():
        check(value.mean(0), summary["policies"][name]["mean_gain_vs_hold_pp"])
        if name in choices and int(choices[name].sum()) != summary["policies"][name]["accepted_rows"]:
            raise ValueError("Choice count differs")
    for baseline in ("accept_all", "hold_all", "history", "matched_random_expectation"):
        for j, domain in enumerate(("balanced", "day", "night")):
            values = [(utility["predicted"][groups == g, j] - utility[baseline][groups == g, j]).mean() for g in unique]
            check_interval(values, summary["policy_contrasts"][baseline][domain])
            mean, width = np.mean(values), t.isf(0.025, 5) * sem(values)
            check([mean - width, mean + width], summary["policy_contrasts"][baseline][domain]["ordinary_ci95"])
    population = {"beneficial_parents": 0, "harmful_balanced_parents": 0}
    for saved in summary["parent_sign_stability"]:
        indices = [
            i
            for i, d in enumerate(data["identities"])
            if d["stream"] == saved["stream"] and d["checkpoint"] == saved["checkpoint"]
        ]
        means = y[indices].mean(0)
        panels = np.array(data["panel_utility"])[indices, :, 0].mean(0)
        check(means[0], saved["mean_balanced_pp"])
        check(y[indices, 0], saved["seed_values_pp"])
        check(panels, saved["panel_means_pp"])
        population["beneficial_parents"] += int(means[0] >= 0.5 and min(means[1:]) >= -1)
        population["harmful_balanced_parents"] += int(means[0] <= -0.5)
    if population != summary["population"]:
        raise ValueError("Population screen differs")
    contrasts = summary["policy_contrasts"]
    gates = {
        "two_sided_population": all(v >= 2 for v in population.values()),
        "measured_health_information": stage_pass["measured"],
        "regression_health_pipeline": stage_pass["predicted"],
        "shadow_policy_gain": all(
            c["balanced"]["mean"] >= 0.25 and c["balanced"]["ci95"][0] > 0 for c in contrasts.values()
        ),
        "shadow_domain_retention": all(c[d]["ci95"][0] >= -1 for c in contrasts.values() for d in ("day", "night")),
    }
    if gates != summary["gates"] or all(gates.values()) != summary["development_screen_passed"]:
        raise ValueError("Overall gate differs")
    for j, metric in enumerate(("variance", "covariance", "rank", "cka", "cosine")):
        for name, p in (
            ("mse", health_prediction[:, j]),
            ("zero_mse", np.zeros(len(r))),
            ("recent_trend_mse", 2 * x[:, 7 + j]),
        ):
            expected = np.mean([np.mean((p[groups == g] - r[groups == g, j]) ** 2) for g in unique])
            check(expected, summary["health_prediction"][metric][name])
    for threshold, saved in summary["threshold_tradeoff_secondary"].items():
        a = choose(predicted["predicted"], float(threshold))
        if int(a.sum()) != saved["accepted"]:
            raise ValueError("Threshold acceptance differs")
        check((a[:, None] * y).mean(0), saved["mean_gain_vs_hold_pp"])
    good = (y[:, 0] >= 0.5) & (np.min(y[:, 1:], axis=1) >= -1)
    bad = (y[:, 0] <= -0.5) | (np.min(y[:, 1:], axis=1) < -1)
    expected_classification = {
        "good_rows": int(good.sum()),
        "bad_rows": int(bad.sum()),
        "neutral_rows": int((~good & ~bad).sum()),
        "accepted_bad": int((choices["predicted"] & bad).sum()),
        "rejected_good": int((~choices["predicted"] & good).sum()),
        "acceptance_is_not_confidence": True,
    }
    if expected_classification != summary["classification"]:
        raise ValueError("Classification differs")
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "cohort_rows": 54,
            "health_scalars": 1080,
            "raw_prediction_arrays": 972,
            "independent_sklearn_svd_fits": fit_count,
            "nested_exclusions": True,
            "primary_intervals": 16,
            "thresholds_unchanged": True,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
        },
    )
    print({"passed": True, "independent_fits": fit_count, "primary_intervals": 16})


if __name__ == "__main__":
    main()
