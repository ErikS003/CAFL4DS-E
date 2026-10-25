"""Independent CPU reconstruction of matched-batch capability and frozen decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import t
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("uniform", "rank_low", "rank_high", "cka_low", "cka_high")
RECIPES = ("action", "history", "bridge")
DOMAINS = ("balanced", "A", "B")
METRICS = ("variance", "covariance", "rank", "cka", "cosine")


def read(path: Path) -> Any:
    """Read preserved evidence."""
    return json.loads(path.read_text(encoding="utf-8"))


def check(a: Any, b: Any) -> None:
    """Allow only floating point solver differences."""
    if not np.allclose(a, b, atol=1e-8, rtol=1e-8):
        raise ValueError(f"Independent bridge reconstruction differs: {a!r} / {b!r}")


def features(o: dict, action: str, recipe: str) -> np.ndarray:
    """Reconstruct causal contrasts directly, without collector feature helpers."""
    flags = [float(action == a) for a in ACTIONS[1:]]
    if recipe == "action":
        return np.array(flags)
    dr = o["history"][4][2] - o["history"][3][2]
    dc = o["history"][4][3] - o["history"][3][3]
    first, reference = o["previews"][action], o["previews"]["uniform"]
    values = flags + [v * o["age"] for v in flags] + [v * dr for v in flags] + [v * dc for v in flags]
    values += [first[k] - reference[k] for k in ("loss", "rgb_mean", "rgb_std")]
    if recipe == "bridge":
        response = [first["predicted"][m] - reference["predicted"][m] for m in METRICS]
        values += response + [v * dr for v in response] + [v * dc for v in response]
    return np.array(values)


def context(series: list[dict], index: int, excluded: set[str], saved: dict) -> dict:
    """Independently fit nested delta AR2 contexts using sklearn's SVD ridge."""
    result = {}
    if saved["excluded"] != sorted(excluded):
        raise ValueError("Nested history exclusion differs")
    for k, name in enumerate(METRICS):
        x, y = [], []
        for s in series:
            if s["stream"] in excluded:
                continue
            h = np.array(s["health"])[:, k]
            for origin in range(4, 17):
                x.append([h[origin] - h[origin - 1], h[origin - 1] - h[origin - 2]])
                y.append(h[origin + 4] - h[origin])
        scaler = StandardScaler().fit(x)
        regression = Ridge(alpha=10, solver="svd").fit(scaler.transform(x), y)
        h = np.array(series[index]["health"])[:, k]
        query = [[h[4] - h[3], h[3] - h[2]]]
        result[name] = float(regression.predict(scaler.transform(query))[0])
        check(result[name], saved["predicted"][name])
        fit = saved["fits"][name]
        check(regression.coef_, fit["beta"])
        check(regression.intercept_, fit["intercept"])
        check(scaler.mean_, fit["center"])
        check(scaler.scale_, fit["scale"])
        if fit["fit_rows"] != len(y):
            raise ValueError("Context row count differs")
    return result


def net_features(o: dict, c: dict) -> list[float]:
    """Only observed parent history and nested forecasts enter net utility."""
    h = o["history"]
    return [o["age"], h[4][2], h[4][3], h[4][2] - h[3][2], h[4][3] - h[3][3], c["rank"], c["cka"]]


def decision(prediction: dict, net: dict, stop: bool) -> str:
    """Reimplement fixed eligibility and stable menu priority."""
    if stop and all(net["balanced"] + prediction[a]["balanced"] <= -0.25 for a in ACTIONS):
        return "hold"
    best, best_gain = "uniform", float("-inf")
    for a in ACTIONS[1:]:
        p = prediction[a]
        eligible = p["balanced"] >= 0.25 and net["balanced"] + p["balanced"] >= 0.25
        eligible &= all(p[d] >= -1 and net[d] + p[d] >= -1 for d in ("A", "B"))
        if eligible and p["balanced"] > best_gain:
            best, best_gain = a, p["balanced"]
    return best


def interval(values: list[float]) -> tuple[float, list[float]]:
    """Eight-comparison family, six history units."""
    a = np.array(values)
    mean = float(a.mean())
    half = float(t.isf(0.05 / 16, 5) * np.std(a, ddof=1) / np.sqrt(6))
    return mean, [mean - half, mean + half]


def main() -> None:
    """Check raw predictions, source-isolated fitting, decisions and qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    lock = read(out / "input_lock.json")
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Locked inputs changed")
    if not read(out / "verification.json")["passed"]:
        raise ValueError("Collector replay audit missing")
    obs = [read(out / f"observations{d}.json")["records"] for d in (0, 1)]
    choices = read(out / "choices.json")["records"]
    outcomes = read(out / "outcomes.json")["records"]
    holds = read(out / "holds.json")["records"]
    models = read(out / "models.json")
    frozen = read(out / "evaluation_decisions.json")
    summary = read(out / "summary.json")
    series = read(ROOT / "outputs/prediction-health-timeseries/20261014-pilot/series.json")["records"]
    if not frozen["frozen_before_draw1_outcomes"] or frozen["model_sha256"] != digest(out / "models.json"):
        raise ValueError("Model freeze differs")
    if frozen["observation_sha256"] != digest(out / "observations1.json"):
        raise ValueError("Observation freeze differs")
    if len(outcomes) != 180 or len(holds) != 18 or len(choices) != 18:
        raise ValueError("Incomplete intervention grid")
    lookup = {(r["parent"], r["draw"], r["action"]): r for r in outcomes}
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as arrays:
        query = arrays["query"]
        labels, domains = arrays["labels"][query], arrays["domains"][query]
    checked_predictions = 0
    for r in outcomes + holds:
        directory = out / r["directory"] if "directory" in r else Path(r["source"])
        with np.load(directory / "predictions.npz") as predictions:
            for reader in ("finetune", "linear", "knn"):
                for domain, mask in (
                    ("balanced", np.ones(len(labels), bool)),
                    ("A", domains == 0),
                    ("B", domains == 1),
                ):
                    panels = []
                    for panel in (51011, 51017, 51023):
                        prediction = predictions[f"{panel}_{reader}"]
                        accuracy = float(np.mean(prediction[mask] == labels[mask]))
                        # Legacy panel means were calculated in torch float32.
                        # Raw-label primary scores and every fit retain strict checks.
                        if abs(accuracy - r["panels"][str(panel)][reader][domain]) > 5e-8:
                            raise ValueError("Float32 panel accuracy differs")
                        panels.append(100 * accuracy)
                        checked_predictions += len(labels[mask])
                    check(np.mean(panels), r["scores"][reader][domain])
        if "draw" in r:
            p, d, a = r["parent"], r["draw"], r["action"]
            preview = obs[d][p]["previews"][a]
            if not r["replay_passed"] or r["trace"][0] != [preview["loss"], preview["gradient_norm"]]:
                raise ValueError("Paired preview/outcome replay differs")
            if r["alias"]:
                alias = next(v for v in outcomes if v["directory"] == r["alias"])
                if alias["fingerprint"] != r["fingerprint"] or alias["selected_slots"] != r["selected_slots"]:
                    raise ValueError("Readout alias state or content differs")
                with np.load(directory / "endpoint.npz") as x, np.load(out / r["alias"] / "endpoint.npz") as y:
                    check(x["embeddings"], y["embeddings"])
    projection_count = 0
    for p, c in enumerate(choices):
        generator = torch.Generator().manual_seed(113003 + p)
        menus = [list(range(1, 64, 2))]
        while len(menus) < 16:
            selected = sorted(torch.randperm(64, generator=generator)[:32].tolist())
            if selected not in menus:
                menus.append(selected)
        if menus != c["menus"]:
            raise ValueError("Visible candidate menu differs")
        records = c["preview_records"]
        selected = {"uniform": 0}
        for metric, prefix in (("rank", "rank"), ("cka", "cka")):
            values = [r["predicted"][metric] for r in records]
            selected[prefix + "_low"] = int(np.argmin(values))
            selected[prefix + "_high"] = int(np.argmax(values))
        if selected != c["selected"]:
            raise ValueError("Unlabeled action selection differs")
        for draw in (0, 1):
            check(obs[draw][p]["history"], series[p]["health"][:5])
            if obs[draw][p]["age"] != lock["cases"][p]["parent_updates"] + 4:
                raise ValueError("Observed age differs")
            for a in ACTIONS:
                if lookup[p, draw, a]["selected_slots"] != menus[selected[a]]:
                    raise ValueError("Draws changed candidate identities")
            for r in records if draw == 0 else list(obs[1][p]["previews"].values()):
                for metric in METRICS:
                    denominator = max(abs(c["parent_values"][metric]), 1e-12) if metric in METRICS[:3] else 1
                    check(sum(r["projection_terms"][metric]) / denominator, r["predicted"][metric])
                    projection_count += 1
    # Every utility fit uses other source histories; contexts additionally exclude
    # each fitting observation's own history. Refit all coefficients independently.
    refits, context_fits = 0, 0
    independent_predictions = {}
    for group, fold in models["folds"].items():
        training = [o for o in obs[0] if o["stream"] != group]
        if fold["training_parents"] != [o["parent"] for o in training]:
            raise ValueError("Held history entered fitting rows")
        contexts = {}
        for o in obs[0]:
            p = o["parent"]
            contexts[p] = context(series, p, {group, o["stream"]}, fold["contexts"][str(p)])
            context_fits += 5
        held = [o for o in obs[1] if o["stream"] == group]
        fitted_net = {}
        xn = [net_features(o, contexts[o["parent"]]) for o in training]
        scaler = StandardScaler().fit(xn)
        for domain in DOMAINS:
            y = [
                lookup[o["parent"], 0, "uniform"]["scores"]["finetune"][domain]
                - holds[o["parent"]]["scores"]["finetune"][domain]
                for o in training
            ]
            fit = Ridge(alpha=10, solver="svd").fit(scaler.transform(xn), y)
            saved = fold["net"][domain]
            check(fit.coef_, saved["beta"])
            check(fit.intercept_, saved["intercept"])
            check(scaler.mean_, saved["center"])
            check(scaler.scale_, saved["scale"])
            fitted_net[domain] = fit
            refits += 1
        for o in held:
            p = o["parent"]
            row = frozen["records"][p]
            net = {
                d: float(fitted_net[d].predict(scaler.transform([net_features(o, contexts[p])]))[0]) for d in DOMAINS
            }
            check(list(net.values()), [row["uniform_net"][d] for d in DOMAINS])
            independent_predictions[p] = {}
        for recipe in RECIPES:
            x = np.array([features(o, a, recipe) for o in training for a in ACTIONS[1:]])
            scale = np.sqrt(np.mean(np.square(x), axis=0))
            scale[scale < 1e-12] = 1
            for domain in DOMAINS:
                y = [
                    lookup[o["parent"], 0, a]["scores"]["finetune"][domain]
                    - lookup[o["parent"], 0, "uniform"]["scores"]["finetune"][domain]
                    for o in training
                    for a in ACTIONS[1:]
                ]
                fit = Ridge(alpha=10, fit_intercept=False, solver="svd").fit(x / scale, y)
                saved = fold["recipes"][recipe][domain]
                check(scale, saved["scale"])
                check(fit.coef_, saved["beta"])
                if saved["fit_rows"] != len(y):
                    raise ValueError("Utility fitting row count differs")
                for o in held:
                    p = o["parent"]
                    pred = fit.predict(np.array([features(o, a, recipe) for a in ACTIONS]) / scale)
                    check(pred, [frozen["records"][p]["recipes"][recipe]["predictions"][a][domain] for a in ACTIONS])
                    independent_predictions[p].setdefault(recipe, {})[domain] = pred
                refits += 1
    for p, recipes in independent_predictions.items():
        row = frozen["records"][p]
        for recipe, values in recipes.items():
            pred = {a: {d: float(values[d][i]) for d in DOMAINS} for i, a in enumerate(ACTIONS)}
            for stop in (False, True):
                key = "stop_choice" if stop else "content_choice"
                if decision(pred, row["uniform_net"], stop) != row["recipes"][recipe][key]:
                    raise ValueError("Frozen policy choice differs")
    groups = sorted(models["folds"])

    def quality(p: int, draw: int, action: str, domain: str, reader: str = "finetune") -> float:
        r = holds[p] if action == "hold" else lookup[p, draw, action]
        return r["scores"][reader][domain]

    errors = {}
    for recipe in ("zero", *RECIPES):
        source = {}
        for g in groups:
            residuals = []
            for o in obs[1]:
                if o["stream"] != g:
                    continue
                p = o["parent"]
                for i, a in enumerate(ACTIONS[1:], 1):
                    target = quality(p, 1, a, "balanced") - quality(p, 1, "uniform", "balanced")
                    prediction = 0 if recipe == "zero" else independent_predictions[p][recipe]["balanced"][i]
                    residuals.append((target - prediction) ** 2)
            source[g] = float(np.mean(residuals))
        errors[recipe] = source
        check(list(source.values()), [summary["forecast_scores"][recipe]["source_mse"][g] for g in groups])
        check(np.mean(list(source.values())), summary["forecast_scores"][recipe]["mse"])
    gates = {}
    for recipe in ("action", "history"):
        mean, bounds = interval([errors[recipe][g] - errors["bridge"][g] for g in groups])
        fraction = 1 - np.mean(list(errors["bridge"].values())) / np.mean(list(errors[recipe].values()))
        saved = summary["forecast_contrasts"][recipe]
        check(mean, saved["mse_improvement"])
        check(bounds, saved["adjusted_95_interval"])
        check(fraction, saved["fractional_reduction"])
        gates["forecast_vs_" + recipe] = bool(fraction >= 0.1 and bounds[0] > 0)
    for name, policy in summary["policy_scores"].items():
        for reader in ("finetune", "linear", "knn"):
            for domain in DOMAINS:
                for control in ("uniform", "hold"):
                    for g in groups:
                        values = [
                            quality(o["parent"], 1, policy["choices"][str(o["parent"])], domain, reader)
                            - quality(o["parent"], 1, control, domain, reader)
                            for o in obs[1]
                            if o["stream"] == g
                        ]
                        check(np.mean(values), policy["readers"][reader][domain][control][g])
        switches = sum(
            a != "hold" and lookup[int(p), 1, a]["selected_slots"] != lookup[int(p), 1, "uniform"]["selected_slots"]
            for p, a in policy["choices"].items()
        )
        if switches != policy["nonuniform_choices"]:
            raise ValueError("Physical content-switch count differs")
        if name.endswith("_content") or name.endswith("_stop"):
            recipe, kind = name.split("_")
            if any(
                policy["choices"][str(p)] != frozen["records"][p]["recipes"][recipe][kind + "_choice"]
                for p in range(18)
            ):
                raise ValueError("Scored policy differs from frozen decision")
    bridge = summary["policy_scores"]["bridge_content"]["readers"]["finetune"]
    history = summary["policy_scores"]["history_content"]["readers"]["finetune"]
    contrast_values = {
        "balanced_vs_uniform": list(bridge["balanced"]["uniform"].values()),
        "balanced_vs_history": [bridge["balanced"]["uniform"][g] - history["balanced"]["uniform"][g] for g in groups],
        **{d + "_vs_" + c: list(bridge[d][c].values()) for d in ("A", "B") for c in ("uniform", "hold")},
    }
    for name, values in contrast_values.items():
        mean, bounds = interval(values)
        check(mean, summary["policy_contrasts"][name]["mean"])
        check(bounds, summary["policy_contrasts"][name]["adjusted_95_interval"])
        gates[name] = bool(mean >= 0.25 and bounds[0] > 0) if name.startswith("balanced") else bool(bounds[0] >= -1)
    gates["content_switches"] = summary["policy_scores"]["bridge_content"]["nonuniform_choices"] > 0
    if gates != summary["gates"] or all(gates.values()) != summary["bridge_qualified"]:
        raise ValueError("Qualification gates differ")
    costs = read(out / "costs.json")
    if (
        costs["total_experiment_updates"] != 2340
        or costs["new_primary_finetune_fits"] + costs["reused_duplicate_fits"] != 540
    ):
        raise ValueError("Work accounting differs")
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "raw_prediction_comparisons": checked_predictions,
            "projection_arithmetic_checks": projection_count,
            "independent_utility_refits": refits,
            "independent_nested_context_refits": context_fits,
            "frozen_policy_decisions": 108,
            "primary_adjusted_contrasts": 8,
            "input_hashes_verified": len(lock["hashes"]),
            "additional_gpu_updates": 0,
            "limits": "CPU reconstruction checks saved predictions and all fitting/selection arithmetic; copied readouts are not independently retrained. Full learner replay is the collector audit.",
        },
    )
    print(json.dumps({"passed": True, "utility_refits": refits, "context_refits": context_fits}))


if __name__ == "__main__":
    main()
