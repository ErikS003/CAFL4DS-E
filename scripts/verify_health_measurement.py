"""Independent numerical and source-group checks for the health measurement."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def properties(z):
    x = z.astype(np.float64)
    s = np.linalg.svd(x, compute_uv=False)
    p = s / s.sum() + 1e-7
    rank = np.exp(-np.sum(p * np.log(p)))
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    centered = normalized - normalized.mean(0)
    cov = centered.T @ centered / (len(x) - 1)
    covariance = (np.sum(cov**2) - np.sum(np.diag(cov) ** 2)) / x.shape[1]
    return rank, normalized.var(0, ddof=1).mean(), covariance


def difference(a, b):
    x, y = a.astype(np.float64), b.astype(np.float64)
    ra, va, ca = properties(a)
    rb, vb, cb = properties(b)
    xc, yc = x - x.mean(0), y - y.mean(0)
    cka = np.sum((yc.T @ xc) ** 2) / np.sqrt(np.sum((xc.T @ xc) ** 2) * np.sum((yc.T @ yc) ** 2))
    cosine = np.mean(np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))
    return {
        "rank_drop": (ra - rb) / ra,
        "variance_drop": (va - vb) / va,
        "covariance_rise": (cb - ca) / ca,
        "cka_drift": max(0.0, 1 - cka),
        "cosine_drift": 1 - cosine,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    lock = read(out / "input_lock.json")
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    ref = lock["reference"]["images"]
    assert len(ref) == len({r["path"] for r in ref}) == 256
    assert all(
        sum(r["cell"] == [light, scene] for r in ref) == 64
        for light in ("daytime", "night")
        for scene in ("city street", "highway")
    )
    root = Path(__file__).resolve().parents[1]
    old = read(root / "outputs/retention-anchor/20261013-gpu/input_lock.json")
    excluded = {Path(p).stem for p in old["hashes"] if Path(p).suffix.lower() in (".jpg", ".jpeg")}
    assert all(Path(r["path"]).stem not in excluded for r in ref)
    observations = read(out / "observations.json")
    assert observations["complete"] and len(observations["records"]) == 63
    warm = np.load(out / "warm_embeddings.npy")
    geometry_checks = 0
    for row in observations["records"]:
        with np.load(out / row["features"], allow_pickle=False) as z:
            assert all(z[name].shape == (256, 96) for name in ("before", "virtual", "actual", "hold"))
            assert np.array_equal(z["before"], z["hold"])
            values = {name: difference(z["before"], z[name]) for name in ("virtual", "actual", "hold")}
            values["snapshot"] = difference(warm, z["before"])
            for name, measurements in values.items():
                expected = row["snapshot"] if name == "snapshot" else row["changes"][name]
                for metric, value in measurements.items():
                    assert abs(value - expected[metric]) < 3e-5, (name, metric, value, expected[metric])
                    geometry_checks += 1
    summary = read(out / "summary.json")
    points = summary["points"]
    models = {
        "mean": (),
        "age_loss": ("updates", "incoming_loss"),
        "virtual_cka": ("updates", "incoming_loss", "v_cka_drift"),
        "snapshot_cka": ("updates", "incoming_loss", "s_cka_drift"),
        "virtual_bundle": (
            "updates",
            "incoming_loss",
            "v_rank_drop",
            "v_variance_drop",
            "v_covariance_rise",
            "v_cka_drift",
            "v_cosine_drift",
        ),
    }
    for point in points:
        rows = [
            r
            for r in observations["records"]
            if r["role"] == "six_stream" and r["stream"] == point["stream"] and r["checkpoint"] == point["checkpoint"]
        ]
        assert len(rows) == 3
        for prefix, key in (("v", "virtual"), ("a", "actual"), ("s", "snapshot")):
            for field in ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift"):
                value = np.mean([r["snapshot"][field] if key == "snapshot" else r["changes"][key][field] for r in rows])
                assert np.isclose(point[prefix + "_" + field], value, atol=1e-12, rtol=0)
    groups = sorted({p["stream"] for p in points})
    y = np.array([p["terminal_damage"] for p in points])
    errors, slopes = {}, []
    for name, columns in models.items():
        x = np.array([[p[f] for f in columns] for p in points]).reshape(18, len(columns))
        prediction = np.zeros(18)
        for group in groups:
            test = np.array([p["stream"] == group for p in points])
            mean, std = x[~test].mean(0), x[~test].std(0)
            std[std == 0] = 1
            z = (x - mean) / std
            design = np.column_stack((z[~test], np.ones(15)))
            penalty = np.eye(len(columns) + 1)
            penalty[-1, -1] = 0
            beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[~test])
            prediction[test] = np.column_stack((z[test], np.ones(3))) @ beta
            if name == "virtual_cka":
                slopes.append(beta[-2])
        assert np.allclose(prediction, summary["models"][name]["predictions"], atol=1e-9, rtol=0)
        errors[name] = {
            g: np.mean((prediction[[p["stream"] == g for p in points]] - y[[p["stream"] == g for p in points]]) ** 2)
            for g in groups
        }
        assert np.isclose(np.mean(list(errors[name].values())), summary["models"][name]["mse"], atol=1e-9, rtol=0)
    improvement = True
    for baseline in ("mean", "age_loss"):
        delta = np.array([errors[baseline][g] - errors["virtual_cka"][g] for g in groups])
        width = t.ppf(0.9875, 5) * delta.std(ddof=1) / np.sqrt(6)
        reduction = 1 - np.mean(list(errors["virtual_cka"].values())) / np.mean(list(errors[baseline].values()))
        assert np.allclose(
            [delta.mean() - width, delta.mean() + width],
            summary["primary_contrasts"][baseline]["adjusted_ci95"],
            atol=1e-9,
            rtol=0,
        )
        improvement &= reduction >= 0.1 and delta.mean() - width > 0
    control = summary["controls"]
    separation = all(
        control["day_end"]["cka_drift"][i] < control[bad]["cka_drift"][i]
        for i in range(3)
        for bad in ("night_end", "return_early")
    )
    assert summary["health_candidate_qualified"] == (improvement and sum(v > 0 for v in slopes) >= 5 and separation)
    result = {
        "passed": True,
        "reference_images": 256,
        "geometry_scalars": geometry_checks,
        "hold_checks": 63,
        "grouped_models": 5,
        "primary_contrasts": 2,
        "source_groups": 6,
        "controls_separate": True,
    }
    atomic_json(out / "independent-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
