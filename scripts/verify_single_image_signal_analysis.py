"""Independent raw-reader, geometry and mature-label regression reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_forecast import HEALTH
from scripts.run_short_buffer_experiment import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_fit(x, residual, calibration, fit, query):
    center, scale = x[calibration].mean(0), x[calibration].std(0)
    scale[scale < 1e-12] = 1
    yc, ys = residual[calibration].mean(0), residual[calibration].std(0)
    ys[ys < 1e-12] = 1
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    penalty = np.diag(np.sqrt(np.r_[1e-8, np.full(x.shape[1], 10.0)]))
    beta = np.linalg.lstsq(
        np.vstack((design[fit], penalty)),
        np.vstack(((residual[fit] - yc) / ys, np.zeros((len(penalty), residual.shape[1])))),
        rcond=None,
    )[0]
    return yc + (design[query] @ beta) * ys


def verify_forecasts(root, settings):
    summary = read(root / "forecast_summary.json")
    checked = 0
    maxdiff = 0.0
    for horizon in settings["horizons"]:
        a = np.load(root / f"forecast_inputs_h{horizon}.npz", allow_pickle=False)
        f = np.load(root / f"forecasts_h{horizon}.npz", allow_pickle=False)
        metrics = a["metrics"].tolist()
        y = a["targets"]
        base = a["baseline"]
        origins = a["origins"]
        ends = a["ends"]
        held = a["held"]
        calibration = ends <= settings["prefix_images"]
        transfer = calibration & ~held
        test = held & (origins >= settings["test_origin_start"])
        np.testing.assert_array_equal(y[test], f["targets"])
        for recipe in [name.removeprefix("x_") for name in a.files if name.startswith("x_")]:
            x = a["x_" + recipe]
            offset = base.copy()
            if recipe.startswith("direct_"):
                offset[:] = 0
            if recipe.startswith("label_free"):
                offset[:, [j for j, k in enumerate(metrics) if k not in HEALTH]] = 0
            residual = y - offset
            for mode, initial in (("transfer", transfer), ("frozen", calibration)):
                predicted = offset[test] + independent_fit(x, residual, initial, initial, test)
                diff = float(np.max(abs(predicted - f[f"{mode}_{recipe}"])))
                maxdiff = max(maxdiff, diff)
                np.testing.assert_allclose(predicted, f[f"{mode}_{recipe}"], rtol=3e-7, atol=2e-8)
                checked += predicted.size
            online = np.zeros_like(y)
            for t in sorted(set(origins[test])):
                fit = calibration | (held & (ends <= t))
                query = test & (origins == t)
                online[query] = offset[query] + independent_fit(x, residual, calibration, fit, query)
            diff = float(np.max(abs(online[test] - f["online_" + recipe])))
            maxdiff = max(maxdiff, diff)
            np.testing.assert_allclose(online[test], f["online_" + recipe], rtol=3e-7, atol=2e-8)
            checked += online[test].size
        np.testing.assert_array_equal(f["no_change"], np.zeros_like(y[test]))
        np.testing.assert_array_equal(f["last_change"], base[test])
        for j, metric in enumerate(metrics):
            for name, score in summary["forecast_outcomes"][str(horizon)][metric]["models"].items():
                error = f[name][:, j] - y[test, j]
                np.testing.assert_allclose(
                    [score["rmse"], score["mae"]], [np.sqrt(np.mean(error**2)), np.mean(abs(error))], rtol=1e-12
                )
    return {"independent_regression_values": checked, "max_absolute_prediction_difference": maxdiff}


def raw_measurements(root):
    panel = torch.load(root / "panel.pt", weights_only=True)
    labels = panel["query_labels"].numpy()
    domains = panel["query_domains"].numpy()
    warm = torch.from_numpy(np.load(root / "warm_bank.npy")).double()
    wc = warm - warm.mean(0)
    wg = wc @ wc.T
    wn = torch.nn.functional.normalize(warm, dim=1)
    files = list((root / "trajectories").glob("*/states/*/readout.json")) + list(
        (root / "diagnostic").rglob("readout.json")
    )
    read_values = 0
    geometry_values = 0
    maximum = 0.0
    for n, path in enumerate(files):
        record = read(path)
        a = np.load(path.with_name("raw.npz"), allow_pickle=False)
        z = torch.from_numpy(a["bank_features"]).double()
        q = torch.nn.functional.normalize(z, dim=1)
        centered = z - z.mean(0)
        gram = centered @ centered.T
        cov = torch.cov(q.T)
        off = cov - torch.diag(torch.diag(cov))
        s = torch.linalg.svdvals(z)
        p = s / s.sum() + 1e-7
        geometry = {
            "rank": float(torch.exp(-(p * p.log()).sum())),
            "variance": float(q.var(0, unbiased=True).mean()),
            "covariance": float(off.square().sum() / z.shape[1]),
            "cka": float(1 - (gram * wg).sum() / (gram.norm() * wg.norm())),
            "cosine": float(1 - (q * wn).sum(1).mean()),
            "feature_norm": float(z.norm(dim=1).mean()),
            "bank_reconstruction": float(a["bank_reconstruction"].mean()),
        }
        for draw, values in record["metrics"].items():
            for metric, value in geometry.items():
                actual = values[metric]
                maximum = max(maximum, abs(actual - value))
                if not np.isclose(actual, value, rtol=2e-10, atol=2e-12):
                    raise ValueError(f"Geometry mismatch {path}/{metric}")
                geometry_values += 1
            for name in (k for k in a.files if k.startswith(draw + "_")):
                reader = name.removeprefix(draw + "_")
                probs = a[name].astype(float)
                correct = probs.argmax(1) == labels
                loss = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1))
                for domain, mask in (
                    ("balanced", np.ones(len(labels), bool)),
                    ("day", domains == 0),
                    ("night", domains == 1),
                ):
                    for metric, value in (
                        (f"{reader}_accuracy_{domain}", 100 * correct[mask].mean()),
                        (f"{reader}_ce_{domain}", loss[mask].mean()),
                    ):
                        if not np.isclose(values[metric], value, rtol=0, atol=1e-12):
                            raise ValueError(f"Reader mismatch {path}/{metric}")
                        read_values += 1
        if n % 100 == 0:
            print(json.dumps({"raw_state_files": n + 1, "total": len(files)}), flush=True)
    return {
        "raw_state_files": len(files),
        "reader_values": read_values,
        "geometry_values": geometry_values,
        "maximum_geometry_difference": maximum,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    lock = read(root / "input_lock.json")
    torch.set_num_threads(1)
    for p, value in lock["hashes"].items():
        if digest(Path(p)) != value:
            raise ValueError(f"Frozen source/input changed: {p}")
    measurements = raw_measurements(root)
    forecasts = verify_forecasts(root, lock["settings"])
    for path in (root / "physical-verification.json", root / "diagnostic/physical-verification.json"):
        if not read(path)["passed"]:
            raise ValueError("Physical replay missing or failed")
    atomic_json(
        root / "analysis-verification.json",
        {
            "passed": True,
            **measurements,
            **forecasts,
            "analysis_source_sha256": digest(Path(__file__)),
            "limitations": "The independently reconstructed measurements use saved reader probabilities/features; diagnostic training and longitudinal states are separately replayed. No independent repeat of every supervised fit.",
        },
    )


if __name__ == "__main__":
    main()
