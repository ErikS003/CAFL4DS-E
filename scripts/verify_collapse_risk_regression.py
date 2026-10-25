"""Independently reconstruct raw geometry, probes, grouped forecasts and warning gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(actual, expected, atol=1e-9):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=atol)


def independent_geometry(raw):
    z = torch.from_numpy(raw).double()
    singular = torch.linalg.svdvals(z)
    mass = singular / singular.sum()
    rank = float(torch.exp(-torch.sum(mass[mass > 0] * mass[mass > 0].log())))
    variance = float(torch.nn.functional.normalize(z, dim=1).var(0, correction=1).mean())
    centered = z - z.mean(0)
    return rank, variance, centered


def uncertainty(values, family=1):
    v = np.asarray(values, float)
    radius = t.isf(0.025 / family, df=len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
    return [float(np.mean(v)), float(np.mean(v) - radius), float(np.mean(v) + radius)]


def sklearn_fit(x, y, query):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-12] = 1
    model = Ridge(alpha=10.0, fit_intercept=True, solver="svd").fit((x - mean) / std, y)
    return model.predict((query - mean) / std), mean, std, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    torch.set_num_threads(2)
    lock, collection = read(out / "input_lock.json"), read(out / "collection.json")
    assert collection["complete"]
    for path, expected in lock["hashes"].items():
        assert sha(path) == expected, path
    records, parents = collection["records"], collection["parents"]
    assert len(records) == 108 and len(parents) == 12
    assert len({(r["group"], r["seed"], r["arm"]) for r in records}) == 108
    fitted, summary = read(out / "regressions.json"), read(out / "summary.json")
    groups = sorted({r["group"] for r in records})
    with np.load(out / "evaluation_arrays.npz") as source:
        arrays = {k: source[k] for k in source.files}
    target = np.eye(int(arrays["labels"].max()) + 1)[arrays["labels"]]
    query = arrays["query"]
    health, point_rows, probe_count, scalar_count = [], [], 0, 0

    def probe_check(features, cached, prefix, scores):
        nonlocal probe_count
        for k, draw in enumerate((51011, 51017, 51023)):
            support = arrays[f"support_{draw}"]
            assert not set(support).intersection(query)
            prediction, _, _, _ = sklearn_fit(
                features[support].astype(float), target[support], features[query].astype(float)
            )
            predicted = prediction.argmax(1)
            np.testing.assert_array_equal(predicted, cached[f"{prefix}{draw}"])
            close(100 * np.mean(predicted == arrays["labels"][query]), scores[k])
            probe_count += 1

    for parent in parents:
        assert parent["exact_replay"]
        for p, expected in parent["artifact_hashes"].items():
            assert sha(p) == expected
        path = next(Path(p) for p in parent["artifact_hashes"] if p.endswith(".npz"))
        with np.load(path) as cached:
            probe_check(cached["initial"], cached, "initial_", parent["initial_scores"])
            probe_check(cached["warm"], cached, "warm_", parent["warm_scores"])
    for index, row in enumerate(records):
        assert row["exact_replay"]
        for p, expected in row["artifact_hashes"].items():
            assert sha(p) == expected
        with np.load(row["arrays"]) as cached:
            raw = cached["projector"]
            assert raw.shape == (17, 256, 128)
            probe_check(cached["evaluation"], cached, "prediction_", row["scores"])
            values = []
            _, _, reference = independent_geometry(raw[0])
            for z in raw:
                rank, variance, centered = independent_geometry(z)
                numerator = torch.linalg.matrix_norm(centered.T @ reference) ** 2
                denominator = torch.linalg.matrix_norm(centered.T @ centered) * torch.linalg.matrix_norm(
                    reference.T @ reference
                )
                drift = float(torch.clamp(1 - numerator / denominator.clamp_min(1e-30), 0, 1))
                values.append([rank, variance, drift])
                scalar_count += 3
        close(values, row["health"], atol=1e-9)
        h = np.array(values)
        health.append(h)
        rank = np.log(h[:, 0] / h[0, 0])
        for origin in range(16, 121, 8):
            j = origin // 8
            x = [
                rank[j],
                rank[j] - rank[j - 1],
                rank[j - 1] - rank[j - 2],
                np.log(h[j, 1] / h[0, 1]),
                h[j, 2],
                np.mean(row["losses"][origin - 8 : origin]),
                origin / 128,
            ]
            point_rows.append(
                {
                    "record": index,
                    "group": row["group"],
                    "arm": row["arm"],
                    "origin": origin,
                    "x": x,
                    "y": rank[j + 1] - rank[j],
                    "current": rank[j],
                }
            )
    assert len(point_rows) == len(fitted["points"]) == 1512
    x, y = np.array([p["x"] for p in point_rows]), np.array([p["y"] for p in point_rows])
    close(x, [p["x"] for p in fitted["points"]])
    close(y, [p["y"] for p in fitted["points"]])
    prediction = np.zeros(len(y))
    for group in groups:
        training = np.array([p["group"] != group and p["arm"][:4] != "soft" for p in point_rows])
        testing = np.array([p["group"] == group for p in point_rows])
        fold = fitted["folds"][group]
        assert fold["fit_points"] == np.flatnonzero(training).tolist()
        assert fold["test_points"] == np.flatnonzero(testing).tolist()
        predicted, center, scale, model = sklearn_fit(x[training], y[training], x[testing])
        prediction[testing] = predicted
        close(center, fold["model"]["center"])
        close(scale, fold["model"]["scale"])
        close(model.coef_, fold["model"]["beta"])
        close(model.intercept_, fold["model"]["intercept"])
        ratios = {}
        for arm in ("healthy", "blend100"):
            ratios[arm] = np.median(
                [
                    h[-1, 0] / h[0, 0]
                    for h, r in zip(health, records, strict=True)
                    if r["group"] != group and r["arm"] == arm
                ]
            )
        close(np.sqrt(ratios["healthy"] * ratios["blend100"]), fold["boundary"])
    close(prediction, [p["predicted"] for p in fitted["points"]])
    close(x[:, 1], [p["trend"] for p in fitted["points"]])
    endpoints = []
    for parent in parents:
        pairs = {
            r["arm"]: (r, h)
            for r, h in zip(records, health, strict=True)
            if r["group"] == parent["group"] and r["seed"] == parent["seed"]
        }
        a, ah = pairs["healthy"]
        b, bh = pairs["blend100"]
        endpoints.append(
            {
                "group": parent["group"],
                "separation": ah[-1, 0] / bh[-1, 0],
                "ratio": ah[-1, 0] / ah[0, 0],
                "anchor": np.mean(a["scores"]) - np.mean(b["scores"]),
                "benefit": np.mean(a["scores"]) - np.mean(parent["warm_scores"]),
            }
        )
    anchor = uncertainty([np.mean([e["anchor"] for e in endpoints if e["group"] == g]) for g in groups])
    benefit = uncertainty([np.mean([e["benefit"] for e in endpoints if e["group"] == g]) for g in groups])
    close(anchor, [summary["downstream_anchor"][k] for k in ("mean", "low", "high")])
    close(benefit, [summary["beneficial_control"][k] for k in ("mean", "low", "high")])
    mask = np.array([p["arm"].startswith("soft") for p in point_rows])
    errors = {}
    for name, values in (("predicted", prediction), ("persistence", np.zeros(len(y))), ("trend", x[:, 1])):
        errors[name] = np.array(
            [
                np.mean(
                    (
                        values[mask & np.array([p["group"] == g for p in point_rows])]
                        - y[mask & np.array([p["group"] == g for p in point_rows])]
                    )
                    ** 2
                )
                for g in groups
            ]
        )
        close(errors[name].mean(), summary["soft_mechanism_mse"][name])
    forecast_pass = True
    for baseline in ("persistence", "trend"):
        bounds = uncertainty(errors[baseline] - errors["predicted"], family=2)
        close(bounds, [summary["mse_improvement"][baseline][k] for k in ("mean", "low", "high")])
        skill = 100 * (1 - errors["predicted"].mean() / errors[baseline].mean())
        close(skill, summary["skill_percent"][baseline])
        forecast_pass &= skill >= 10 and bounds[1] > 0
    for report in [summary["warnings"], *summary["secondary_buffers"]]:
        event_count = eligible_count = warned_count = 0
        nominal = {g: [] for g in groups}
        for i, (row, h) in enumerate(zip(records, health, strict=True)):
            cutoff = np.log(fitted["folds"][row["group"]]["boundary"])
            ranks = np.log(h[:, 0] / h[0, 0])
            event_indices = [j for j in range(16) if ranks[j] <= cutoff and ranks[j + 1] <= cutoff]
            if row["arm"].startswith("soft") and event_indices:
                event_count += 1
                first = event_indices[0] * 8
                if 24 <= first <= 120:
                    eligible_count += 1
                    k = next(k for k, p in enumerate(point_rows) if p["record"] == i and p["origin"] == first - 8)
                    warned_count += int(
                        point_rows[k]["current"] > cutoff
                        and point_rows[k]["current"] + prediction[k] <= cutoff + report["buffer"]
                    )
            if row["arm"] == "healthy":
                for k, point in enumerate(point_rows):
                    if point["record"] == i:
                        j = point["origin"] // 8
                        if ranks[j] > cutoff and ranks[j + 1] > cutoff:
                            nominal[row["group"]].append(bool(ranks[j] + prediction[k] <= cutoff + report["buffer"]))
        assert report["all_events"] == event_count and report["eligible_events"] == eligible_count
        close(report["episode_recall"], warned_count / eligible_count if eligible_count else 0)
        close(report["eligible_fraction"], eligible_count / event_count if event_count else 0)
        opportunities = [v for values in nominal.values() for v in values]
        assert report["nominal_opportunities"] == len(opportunities)
        assert report["false_warnings"] == sum(opportunities)
        for g, values in nominal.items():
            if values:
                close(np.mean(values), report["cohort_false_warning_rate"][g])
            else:
                assert report["cohort_false_warning_rate"][g] is None
    w = summary["warnings"]
    gates = {
        "instrument": min(e["separation"] for e in endpoints) >= 2
        and np.median([e["ratio"] for e in endpoints]) >= 0.8,
        "downstream_anchor": anchor[0] >= 1 and anchor[1] > 0,
        "beneficial_control": benefit[1] > 0,
        "forecast": forecast_pass,
        "warning": w["eligible_events"] >= 6
        and w["episode_recall"] >= 0.8
        and w["eligible_fraction"] >= 0.8
        and w["false_warning_rate"] is not None
        and w["false_warning_rate"] <= 0.05
        and all(v is not None and v <= 0.1 for v in w["cohort_false_warning_rate"].values()),
    }
    assert gates == summary["gates"]
    assert bool(all(gates.values())) == summary["ready_for_separate_controller_experiment"]
    assert not summary["live_filter_qualified"]
    assert summary["scientific_updates"] == summary["replay_updates"] == 17664
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "locked_inputs": len(lock["hashes"]),
            "raw_health_scalars": scalar_count,
            "independent_probe_fits": probe_count,
            "independent_regression_fits": 6,
            "forecast_points": len(point_rows),
            "all_complete_trajectory_replays": True,
            "gates": gates,
        },
    )
    print(json.dumps(read(out / "independent-verification.json")))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
