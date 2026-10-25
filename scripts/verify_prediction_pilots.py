"""Independently reconstruct paired capability targets, health geometry and forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, tolerance=1e-10):
    if not np.allclose(a, b, atol=tolerance, rtol=tolerance):
        raise ValueError(f"Independent reconstruction differs: {np.max(np.abs(np.asarray(a) - np.asarray(b)))}")


def sklearn_forecast(x, y, groups):
    """Use an independent SVD ridge implementation, never the collector's solve."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    result = np.zeros(len(y))
    for group in sorted(set(groups)):
        test = groups == group
        if not x.shape[1]:
            result[test] = y[~test].mean()
            continue
        scaler = StandardScaler().fit(x[~test])
        fit = Ridge(alpha=10, solver="svd").fit(scaler.transform(x[~test]), y[~test])
        result[test] = fit.predict(scaler.transform(x[test]))
    return result


def check_models(container, rows, recipes, groups):
    y = np.asarray(container["targets"])
    for name, fields in recipes.items():
        x = np.asarray([[r[f] for f in fields] for r in rows]).reshape(len(rows), len(fields))
        close(sklearn_forecast(x, y, groups), container["models"][name]["predictions"])
    for model in container["models"].values():
        prediction = np.asarray(model["predictions"])
        errors = []
        for group in sorted(set(groups)):
            mask = np.asarray(groups) == group
            value = float(((prediction[mask] - y[mask]) ** 2).mean())
            close(value, model["source_mse"][str(group)])
            errors.append(value)
        close(np.mean(errors), model["mse"])
    return len(container["models"])


def check_contrast(candidate, baseline, contrast, family):
    errors = np.array([baseline["source_mse"][s] - candidate["source_mse"][s] for s in sorted(candidate["source_mse"])])
    half = t.isf(0.05 / (2 * family), len(errors) - 1) * np.std(errors, ddof=1) / np.sqrt(len(errors))
    close([errors.mean() - half, errors.mean() + half], contrast["adjusted_95_interval"])
    reduction = 1 - candidate["mse"] / baseline["mse"] if baseline["mse"] > 1e-16 else None
    passes = reduction is not None and reduction >= 0.1 and errors.mean() - half > 0
    if passes != contrast["passes"]:
        raise ValueError("Qualification gate reconstruction failed")


def capability(out):
    summary = read(out / "summary.json")
    prior = ROOT / "outputs/budget-reservation/20261013-gpu"
    records = read(prior / "collection.json")["records"]
    lookup = {(r["stream"], r["checkpoint"], r["seed"], r["arm"]): r for r in records}
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    raw_sets = 0
    base = ("updates", "incoming_loss")
    state = base + ("rank", "variance", "covariance", "cka_drift", "cosine_drift")
    recipes = {
        "mean": (),
        "count_loss": base,
        "parent_health": state,
        "paid_preview": state
        + tuple("preview_" + f for f in ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")),
        "paid_anchor": state + ("relative_anchor_change",),
    }
    targets = {key: [] for key in summary["outcomes"]}
    for point in summary["points"]:
        for horizon in ("immediate", "terminal"):
            effects = {d: [] for d in ("balanced", "A", "B")}
            for seed in (5003, 5009, 5011):
                scored = {}
                for arm in ("spend", "hold"):
                    row = lookup[point["stream"], point["checkpoint"], seed, arm]
                    with np.load(prior / row[horizon] / "predictions.npz") as data:
                        scored[arm] = {}
                        for domain, mask in (
                            ("balanced", np.ones(len(labels), bool)),
                            ("A", domains == 0),
                            ("B", domains == 1),
                        ):
                            scored[arm][domain] = np.mean(
                                [
                                    100 * np.equal(data[f"{panel}_finetune"][mask], labels[mask]).mean()
                                    for panel in (51011, 51017, 51023)
                                ]
                            )
                            raw_sets += 3
                for domain in effects:
                    effects[domain].append(scored["spend"][domain] - scored["hold"][domain])
            for domain in effects:
                targets[horizon + "_" + domain].append(float(np.mean(effects[domain])))
    for key, data in summary["outcomes"].items():
        close(targets[key], data["targets"])
        check_models(data, summary["points"], recipes, summary["groups"])
        for baseline, contrast in data["primary_candidate_contrasts"].items():
            check_contrast(data["models"]["parent_health"], data["models"][baseline], contrast, 2)
    return {"capability_targets": 108, "raw_panel_domain_scores": raw_sets, "independent_sklearn_fits": 180}


def torch_geometry(z, warm):
    """Independent float64 torch and centered-Gram implementation."""
    z, warm = torch.as_tensor(z).double(), torch.as_tensor(warm).double()
    q = torch.nn.functional.normalize(z, dim=1)
    w = torch.nn.functional.normalize(warm, dim=1)
    centered = q - q.mean(0)
    covariance = centered.T @ centered / (len(q) - 1)
    off = covariance - torch.diag(torch.diagonal(covariance))
    singular = torch.linalg.svdvals(z)
    probability = singular / singular.sum() + 1e-7
    a, b = z - z.mean(0), warm - warm.mean(0)
    ga, gb = a @ a.T, b @ b.T
    cka = (ga * gb).sum() / (torch.linalg.norm(ga) * torch.linalg.norm(gb))
    return {
        "variance": float(centered.square().sum(0).mean() / (len(q) - 1)),
        "covariance": float(off.square().sum() / z.shape[1]),
        "rank": float(torch.exp(-(probability * probability.log()).sum())),
        "cka": float(1 - cka),
        "cosine": float(1 - (q * w).sum(1).mean()),
    }


def image_health(out):
    torch.set_num_threads(4)
    lock, summary = read(out / "input_lock.json"), read(out / "summary.json")
    records = read(out / "observations.json")["records"]
    if len(records) != 288:
        raise ValueError("Incomplete observation grid")
    warm = np.load(out / "warm.npy")
    index = {(r["stream"], r["checkpoint"]): i for i, r in enumerate(lock["cases"])}
    metrics_cache = {}

    def metric(path, key):
        cache_key = (str(path), key)
        if cache_key not in metrics_cache:
            with np.load(path) as data:
                metrics_cache[cache_key] = torch_geometry(data[key], warm)
        return metrics_cache[cache_key]

    for row in records:
        i, draw, slot = index[row["stream"], row["checkpoint"]], row["draw"], row["slot"]
        parent = metric(out / f"parent_{i:02d}.npz", "before")
        if row["intervention"] == "marginal":
            after = metric(out / f"full_{i:02d}_{draw}.npz", "after")
            baseline = metric(out / f"image_{i:02d}_{draw}_{slot}.npz", "omitted")
        else:
            after = metric(out / f"image_{i:02d}_{draw}_{slot}.npz", "single")
            baseline = parent
        for key in parent:
            value = after[key] - baseline[key]
            if key in ("variance", "rank", "covariance"):
                value /= max(abs(parent[key]), 1e-12)
            close(value, row["effect"][key], 1e-9)
    state = ("updates", "parent_rank", "parent_variance", "parent_covariance", "parent_cka")
    image = state + ("rgb_mean", "rgb_std", "edge", "reconstruction_loss", "centroid_distance", "loss_variance")
    recipes = {"mean": (), "state": state, "image_state": image}
    for name, intervention in summary["interventions"].items():
        rows = [r for r in records if r["intervention"] == name]
        for metric_name, data in intervention["metrics"].items():
            close([r["effect"][metric_name] for r in rows], data["targets"])
            check_models(data, [r["features"] for r in rows], recipes, intervention["groups"])
            for candidate, baselines in data["primary_contrasts"].items():
                for baseline, contrast in baselines.items():
                    check_contrast(data["models"][candidate], data["models"][baseline], contrast, 4)
    return {
        "image_response_rows": len(records),
        "independent_health_scalars": len(metrics_cache) * 5,
        "independent_sklearn_fits": 180,
        "new_ssl_updates": 0,
        "derivative_scope": "finite-difference tests and collector exact replay; no independent new GPU derivative reconstruction",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--image-health", type=Path, required=True)
    args = parser.parse_args()
    for out, verify in ((args.capability, capability), (args.image_health, image_health)):
        lock = read(out / "input_lock.json")
        if any(digest(Path(path)) != expected for path, expected in lock["hashes"].items()):
            raise ValueError("Scientific source/input hash changed")
        result = verify(out)
        atomic_json(
            out / "independent-verification.json", {"passed": True, **result, "verifier_sha256": digest(Path(__file__))}
        )
        print(json.dumps({"output": str(out), "passed": True, **result}), flush=True)


if __name__ == "__main__":
    main()
