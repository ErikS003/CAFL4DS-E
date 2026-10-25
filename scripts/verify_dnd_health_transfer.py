"""Independent prospective alarms, raw scores, risk duration and exact shuffle audit."""

import itertools
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import t
from threadpoolctl import threadpool_limits

from scripts.run_dnd_health_transfer import CONFIG
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read, write


def changes(a, b):
    a, b = a.astype(float), b.astype(float)
    ranks = []
    variances = []
    for z in (a, b):
        singular = np.linalg.svd(z, compute_uv=False)
        p = singular / max(singular.sum(), 1e-12) + 1e-7
        ranks.append(np.exp(-(p * np.log(p)).sum()))
        u = z / np.maximum(np.sqrt((z * z).sum(1, keepdims=True)), 1e-12)
        variances.append(np.cov(u, rowvar=False).diagonal().mean())
    x, y = a - a.mean(0), b - b.mean(0)
    cka = np.linalg.norm(x.T @ y) ** 2 / max(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y), 1e-30)
    return np.array([1 - ranks[1] / ranks[0], 1 - variances[1] / variances[0], max(0, 1 - cka)])


def main():
    started = time.perf_counter()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert not (out / "independent-verification.json").exists()
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    assert read(ROOT / config.intake / "independent-verification.json")["passed"]
    replay = read(out / "training-replay-verification.json")
    assert replay["passed"] and replay["ssl_updates"] == 3584 and replay["exact_classifier_fit_replays"] == 2
    monitor = [132, *range(140, 445, 8)]
    points = sorted({0, *monitor, 448, 148, 264, 160, 192, 224, 256, 288, 320})
    assert len(points) == 49
    with np.load(ROOT / config.evaluation / "data.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
        assert len(labels) == 3663
        assert all(
            len(a[f"support_{draw}"]) == 50 and len(np.intersect1d(a[f"support_{draw}"], a["query"])) == 0
            for draw in (61001, 61007, 61013)
        )
    collection = read(out / "collection.json")
    summary = read(out / "summary.json")
    full = []
    stops = []
    prediction_sets = 0
    for i, r in enumerate(collection["records"]):
        assert r["name"] == f"transfer_{i:02d}"
        unit = out / r["name"]
        check(read(unit / "complete.json")["hashes"])
        curve = []
        with np.load(unit / "embeddings.npz") as bank:
            streak = 0
            events = []
            stop = None
            for point in monitor[1:]:
                v = changes(bank["132"], bank[str(point)])
                alarm = bool(np.any(v >= 0.1 - 1e-12))
                streak = streak + 1 if alarm else 0
                events.append({"step": point, "values": v, "alarm": alarm, "streak": streak, "stopped": streak >= 2})
                if streak >= 2:
                    stop = point
                    break
            assert r["stop"] == (stop or 448) and r["alarm_fired"] == (stop is not None)
            assert len(events) == len(r["events"])
            for actual, expected in zip(r["events"], events, strict=True):
                for key in ("step", "alarm", "streak", "stopped"):
                    assert actual[key] == expected[key]
                np.testing.assert_allclose(actual["values"], expected["values"], atol=1e-11)
        for point in points:
            o = r["observations"][str(point)]
            assert o["on_controller_schedule"] == (point in monitor)
            assert o["controller_observed"] == (point in monitor and (stop is None or point <= stop))
            scores = []
            for draw in (61001, 61007, 61013):
                path = Path(o["readout"]) / f"panel_{draw}.npz"
                meta = read(path.with_suffix(".json"))
                assert digest(path) == meta["sha256"] and meta["encoder_fingerprint"] == o["encoder_fingerprint"]
                assert meta["learner_fingerprint"] == o["learner_fingerprint"]
                with np.load(path) as a:
                    probability = a["probabilities"]
                    assert probability.shape == (4063, 2)
                    prediction = probability[:3663].argmax(1)
                values = []
                for d in (0, 1):
                    cells = []
                    for c in (0, 1):
                        mask = (domains == d) & (labels == c)
                        cells.append(np.count_nonzero(prediction[mask] == labels[mask]) / mask.sum())
                    values.append(50 * sum(cells))
                scores.append([(values[0] + values[1]) / 2, *values])
                prediction_sets += 1
            np.testing.assert_allclose(scores, o["panel_accuracy_pp"], atol=1e-12)
            curve.append(np.mean(scores, axis=0))
        full.append(curve)
        stops.append(r["stop"])
        snapshot = torch.load(unit / "stopped_state.pt", weights_only=True, map_location="cpu")
        assert (
            fingerprint(snapshot)
            == r["hold_verification"]["state_fingerprint"]
            == r["observations"][str(r["stop"])]["learner_fingerprint"]
        )
        assert r["hold_verification"]["passed"] and r["hold_verification"]["skipped_batches"] == 448 - r["stop"]
    full = np.array(full)

    def path(i, k):
        return np.array([full[i, points.index(min(p, k))] for p in points])

    def metrics(curve):
        deficits = np.array([curve[: j + 1].max(0) - value for j, value in enumerate(curve)])
        lengths = np.array([b - a if a >= 132 else 0 for a, b in zip(points[:-1], points[1:], strict=True)])
        drawdown = deficits.max(0)
        return np.r_[
            curve[-1],
            drawdown,
            (deficits[:-1] * lengths[:, None]).sum(0) / 316,
            ((deficits[:-1] > 1) * lengths[:, None]).sum(0) * 2,
            max(drawdown[1:]),
        ]

    def same_interval(saved, array):
        array = np.asarray(array)
        mean = array.mean(0)
        half = t.ppf(0.975, 7) * array.std(0, ddof=1) / np.sqrt(8)
        np.testing.assert_allclose(saved["mean"], mean, atol=1e-10)
        np.testing.assert_allclose(saved["history_values"], array, atol=1e-10)
        np.testing.assert_allclose(saved["ci95"], np.stack([mean - half, mean + half], axis=-1), atol=1e-10)

    rule_metrics = {}
    for name, selected in {
        "frozen": [0] * 8,
        "full": [448] * 8,
        "fixed148": [148] * 8,
        "fixed264": [264] * 8,
        "health": stops,
    }.items():
        rule = summary["rules"][name]
        assert rule["stops"] == selected and rule["total_updates"] == sum(selected)
        paths = np.array([path(i, k) for i, k in enumerate(selected)])
        np.testing.assert_allclose(paths, rule["history_trajectories_pp"], atol=1e-11)
        np.testing.assert_allclose(paths.mean(0), rule["trajectory_mean_pp"], atol=1e-11)
        value = np.array([metrics(p) for p in paths])
        rule_metrics[name] = value
        for j, key in enumerate(summary["metric_order"]):
            same_interval(rule["metrics"][key], value[:, j])
        same_interval(rule["terminal_minus_warm"], value[:, :3] - full[:, 0])
    donor = np.array([[metrics(path(i, k)) for k in stops] for i in range(8)])
    permutations = np.array(list(itertools.permutations(range(8))), dtype=np.int16)
    with np.load(out / "shuffle_assignments.npz") as a:
        np.testing.assert_array_equal(a["permutations"], permutations)
        means = np.array([donor[np.arange(8), p].mean(0) for p in permutations])
        np.testing.assert_allclose(a["assignment_mean_metrics"], means, atol=1e-11)
    assert (np.array(stops)[permutations].sum(1) == sum(stops)).all()
    shuffle = summary["matched_work_shuffle"]
    assert shuffle["assignments"] == 40320 and shuffle["total_updates_each"] == sum(stops)
    expected = donor.mean(1)
    np.testing.assert_allclose(shuffle["history_expected_metrics"], expected, atol=1e-11)
    np.testing.assert_allclose(list(shuffle["mean_metrics"].values()), means.mean(0), atol=1e-11)
    np.testing.assert_allclose(
        shuffle["assignment_quantiles"], np.quantile(means, [0.025, 0.5, 0.975], axis=0), atol=1e-11
    )
    actual = rule_metrics["health"]
    risk = expected[:, -1] - actual[:, -1]
    quality = actual[:, :3] - expected[:, :3]
    same_interval(summary["health_vs_shuffle"]["risk_reduction"], risk)
    same_interval(summary["health_vs_shuffle"]["terminal_difference_pp"], quality)
    fractional = 1 - actual[:, -1].mean() / max(means[:, -1].mean(), 1e-30)
    np.testing.assert_allclose(summary["health_vs_shuffle"]["fractional_risk_reduction"], fractional, atol=1e-10)
    low = risk.mean() - t.ppf(0.975, 7) * risk.std(ddof=1) / np.sqrt(8)
    assert summary["added_timing_information_flag"] == bool(
        fractional >= 0.2
        and risk.mean() >= 0.5
        and low > 0
        and quality.mean(0)[0] >= -0.5
        and (quality.mean(0)[1:] >= -1).all()
    )
    qfull = actual[:, :3] - rule_metrics["full"][:, :3]
    assert summary["full_quality_tolerances_met"] == bool(qfull.mean(0)[0] >= -0.5 and (qfull.mean(0)[1:] >= -1).all())
    for name in ("full", "fixed148", "fixed264"):
        same_interval(summary["contrasts"][name]["terminal_difference_pp"], actual[:, :3] - rule_metrics[name][:, :3])
        same_interval(
            summary["contrasts"][name]["worst_drawdown_reduction_pp"], rule_metrics[name][:, -1] - actual[:, -1]
        )
    attempts = [read(p) for p in (out / "attempts").glob("collect_*.json")]
    assert sum(a["fits_started"] for a in attempts) == sum(a["fits_completed"] for a in attempts) == 1155
    assert sum(a["ssl_started"] for a in attempts) == sum(a["ssl_completed"] for a in attempts) == 3584
    assert len(list(out.glob("transfer_*/at_*/panel_*.npz"))) + len(list((out / "warm").glob("panel_*.npz"))) == 1155
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "fresh_histories": 8,
            "primary_prediction_set_uses": prediction_sets,
            "canonical_fit_files": 1155,
            "dense_states": 392,
            "guard_decisions": 8,
            "shuffle_assignments": 40320,
            "time_weighting_rebuilt": True,
            "ssl_collection_updates": 3584,
            "ssl_replay_updates": 3584,
            "extra_classifier_replay_fits": 2,
            "source_sha256": digest(__file__),
            "seconds": time.perf_counter() - started,
        },
    )
    print({"independent_verification": "passed", "histories": 8, "shuffle_assignments": 40320}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
