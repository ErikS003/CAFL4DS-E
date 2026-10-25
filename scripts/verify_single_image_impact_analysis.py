"""Reconstruct isolated targets, predictor exclusions and mature-label regressions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_forecast import HEALTH
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_signal import read


def raw_metrics(path, warm, labels, domains):
    """Independent float64 raw-feature/probability reconstruction, without live readers."""
    record = read(path)
    saved = np.load(path.with_name("raw.npz"), allow_pickle=False)
    z = torch.from_numpy(saved["bank_features"]).double()
    normalized = torch.nn.functional.normalize(z, dim=1)
    centered = z - z.mean(0)
    gram = centered @ centered.T
    wc = warm - warm.mean(0)
    warm_gram = wc @ wc.T
    cov = torch.cov(normalized.T)
    off = cov - torch.diag(torch.diag(cov))
    singular = torch.linalg.svdvals(z)
    p = singular / singular.sum() + 1e-7
    geometry = {
        "rank": float(torch.exp(-(p * p.log()).sum())),
        "variance": float(normalized.var(0, unbiased=True).mean()),
        "covariance": float(off.square().sum() / z.shape[1]),
        "cka": float(1 - (gram * warm_gram).sum() / (gram.norm() * warm_gram.norm())),
        "cosine": float(1 - (normalized * torch.nn.functional.normalize(warm, dim=1)).sum(1).mean()),
        "feature_norm": float(z.norm(dim=1).mean()),
        "bank_reconstruction": float(saved["bank_reconstruction"].mean()),
    }
    reconstructed = {}
    counts = {"geometry_values": 0, "reader_values": 0}
    for draw, original in record["metrics"].items():
        values = dict(geometry)
        for key in [k for k in saved.files if k.startswith(draw + "_")]:
            reader = key.removeprefix(draw + "_")
            probabilities = saved[key].astype(float)
            correct = probabilities.argmax(1) == labels
            loss = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1))
            for domain, mask in (
                ("balanced", np.ones(len(labels), bool)),
                ("day", domains == 0),
                ("night", domains == 1),
            ):
                values[f"{reader}_accuracy_{domain}"] = float(100 * correct[mask].mean())
                values[f"{reader}_ce_{domain}"] = float(loss[mask].mean())
                counts["reader_values"] += 2
        for key, value in values.items():
            np.testing.assert_allclose(
                value,
                original[key],
                rtol=2e-10 if key in geometry else 0,
                atol=2e-12 if key in geometry else 1e-12,
                err_msg=str(path) + "/" + key,
            )
        counts["geometry_values"] += len(geometry)
        reconstructed[draw] = values
    saved.close()
    return reconstructed, counts


def independent_fit(x, y, normalization, fit, query, penalty):
    center, scale = x[normalization].mean(0), x[normalization].std(0)
    scale[scale < 1e-12] = 1
    yc, ys = y[normalization].mean(0), y[normalization].std(0)
    ys[ys < 1e-12] = 1
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    prior = np.diag(np.sqrt(np.r_[1e-8, np.full(x.shape[1], penalty)]))
    beta = np.linalg.lstsq(
        np.vstack((design[fit], prior)), np.vstack(((y[fit] - yc) / ys, np.zeros((len(prior), y.shape[1])))), rcond=None
    )[0]
    return yc + (design[query] @ beta) * ys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    lock = read(root / "input_lock.json")
    prior = Path(lock["prior"])
    settings = lock["settings"]
    for path, expected in {**lock["prior_lock"]["hashes"], **lock["hashes"]}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input/source changed: {path}")
    torch.set_num_threads(1)
    panel = torch.load(prior / "panel.pt", weights_only=True)
    labels, domains = panel["query_labels"].numpy(), panel["query_domains"].numpy()
    warm = torch.from_numpy(np.load(prior / "warm_bank.npy")).double()
    a = np.load(root / "inputs.npz", allow_pickle=False)
    f = np.load(root / "forecasts.npz", allow_pickle=False)
    metrics = a["metrics"].tolist()
    rows = read(root / "collection.json")["records"]
    counts = {"raw_state_files": 0, "geometry_values": 0, "reader_values": 0, "isolated_target_values": 0}
    series_cache = {}
    for i, row in enumerate(rows):
        actual_path = Path(row["actual_readout"])
        assert digest(actual_path) == row["actual_readout_sha256"]
        assert digest(actual_path.with_name("raw.npz")) == row["actual_raw_sha256"]
        outcomes = []
        for path in (actual_path, root / row["zero_readout"]):
            reconstructed, c = raw_metrics(path, warm, labels, domains)
            counts["raw_state_files"] += 1
            for k, v in c.items():
                counts[k] += v
            outcomes.append(np.array([reconstructed[row["reader_draw"]][k] for k in metrics]))
        np.testing.assert_allclose(a["actual"][i], outcomes[0], rtol=0, atol=2e-11)
        np.testing.assert_allclose(a["zero"][i], outcomes[1], rtol=0, atol=2e-11)
        np.testing.assert_allclose(a["targets"][i], outcomes[0] - outcomes[1], rtol=0, atol=2e-11)
        counts["isolated_target_values"] += len(metrics)
        g, t = row["history"], row["origin"]
        if g not in series_cache:
            series_cache[g] = read(prior / "trajectories" / g / "series.json")
        series = series_cache[g]
        before = np.array([series["states"][str(t)][k] for k in metrics])
        previous = np.array([series["states"][str(t - 1)][k] for k in metrics])
        feature = series["features"][str(t)]
        state = np.r_[before, before - previous, feature["optimizer"], t]
        selected = [metrics.index(k) for k in HEALTH]
        health = np.r_[before[selected], (before - previous)[selected], feature["optimizer"], t]
        for recipe, design in (
            ("state", state),
            ("image", np.r_[state, feature["image"]]),
            ("gradient", np.r_[state, feature["image"], feature["gradient"]]),
            ("label_free_state", health),
            ("label_free_image", np.r_[health, feature["image"]]),
            ("label_free_gradient", np.r_[health, feature["image"], feature["gradient"]]),
        ):
            np.testing.assert_array_equal(a["x_" + recipe][i], design)
        if i % 50 == 0:
            print(f"Raw isolated-effect pairs reconstructed: {i + 1}/{len(rows)}", flush=True)
    y, origins, ends, held = [a[k] for k in ("targets", "origins", "ends", "held")]
    calibration = ends <= settings["prefix_images"]
    test = held & (origins >= settings["test_origin_start"])
    np.testing.assert_array_equal(y[test], f["targets"])
    maximum, checked = 0.0, 0
    for recipe in [key.removeprefix("x_") for key in a.files if key.startswith("x_")]:
        x = a["x_" + recipe]
        for mode, initial in (("transfer", calibration & ~held), ("frozen", calibration)):
            expected = independent_fit(x, y, initial, initial, test, settings["ridge_penalty"])
            maximum = max(maximum, float(np.max(abs(expected - f[mode + "_" + recipe]))))
            np.testing.assert_allclose(expected, f[mode + "_" + recipe], rtol=3e-7, atol=2e-9)
            checked += expected.size
        expected = np.zeros_like(y)
        for t in sorted(set(origins[test])):
            fit = calibration | (held & (ends <= t))
            query = test & (origins == t)
            expected[query] = independent_fit(x, y, calibration, fit, query, settings["ridge_penalty"])
        maximum = max(maximum, float(np.max(abs(expected[test] - f["online_" + recipe]))))
        np.testing.assert_allclose(expected[test], f["online_" + recipe], rtol=3e-7, atol=2e-9)
        checked += expected[test].size
    np.testing.assert_array_equal(f["zero_effect"], np.zeros_like(y[test]))
    np.testing.assert_array_equal(f["transfer_mean"], np.tile(y[calibration & ~held].mean(0), (test.sum(), 1)))
    np.testing.assert_array_equal(f["prefix_mean"], np.tile(y[calibration].mean(0), (test.sum(), 1)))
    np.testing.assert_array_equal(
        f["online_mean"], np.array([y[calibration | (held & (ends <= t))].mean(0) for t in origins[test]])
    )
    for j, i in enumerate(np.flatnonzero(test)):
        available = np.flatnonzero((a["histories"] == a["histories"][i]) & (ends <= origins[i]))
        np.testing.assert_array_equal(f["previous_effect"][j], y[available[np.argmax(ends[available])]])
    summary = read(root / "summary.json")
    for j, metric in enumerate(metrics):
        for model, result in summary["outcomes"][metric]["models"].items():
            error = f[model][:, j] - y[test, j]
            np.testing.assert_allclose(
                [result["rmse"], result["mae"]], [np.sqrt(np.mean(error**2)), abs(error).mean()], rtol=1e-12
            )
    if not read(root / "physical-verification.json")["passed"]:
        raise ValueError("Physical replay must pass")
    atomic_json(
        root / "analysis-verification.json",
        {
            "passed": True,
            **counts,
            "independent_regression_values": checked,
            "maximum_prediction_difference": maximum,
            "all_designs_reconstructed_from_pre_update_observations": True,
            "all_five_baselines_independently_reconstructed": True,
            "source_sha256": digest(Path(__file__)),
            "supervised_fits_independently_retrained": False,
        },
    )


if __name__ == "__main__":
    main()
