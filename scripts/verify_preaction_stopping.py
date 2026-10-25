"""Independent parent features, least-squares fits, envelopes and stopping scores."""

from __future__ import annotations

import itertools
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import t
from threadpoolctl import threadpool_limits

from scripts.run_preaction_stopping import CONFIG, ROOT, check, digest, read, write


def geometry(z, warm, domains):
    def raw(a, b):
        a, b = a.astype(float), b.astype(float)
        u = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        v = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        p = np.linalg.svd(a, compute_uv=False)
        p = p / max(p.sum(), 1e-12) + 1e-7
        covariance = np.cov(u, rowvar=False)
        off = covariance.copy()
        np.fill_diagonal(off, 0)
        ac, bc = a - a.mean(0), b - b.mean(0)
        cka = np.sum((ac.T @ bc) ** 2) / max(np.linalg.norm(ac.T @ ac) * np.linalg.norm(bc.T @ bc), 1e-30)
        return np.array(
            [
                np.exp(-sum(p * np.log(p))),
                covariance.diagonal().mean(),
                (off**2).sum() / a.shape[1],
                1 - cka,
                1 - (u * v).sum(1).mean(),
            ]
        )

    values = []
    for mask in (np.ones(len(z), bool), domains == 0, domains == 1):
        current, anchor = raw(z[mask], warm[mask]), raw(warm[mask], warm[mask])
        values.extend([*np.log(np.maximum(current[:3], 1e-30) / np.maximum(anchor[:3], 1e-30)), *current[3:]])
    return np.array(values)


def fit(x, y, q):
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    z = np.c_[np.ones(len(x)), (x - center) / scale]
    regularizer = np.c_[np.zeros(x.shape[1]), np.sqrt(10) * np.eye(x.shape[1])]
    beta = np.linalg.lstsq(np.r_[z, regularizer], np.r_[y, np.zeros((x.shape[1], y.shape[1]))], rcond=None)[0]
    return np.c_[np.ones(len(q)), (q - center) / scale] @ beta


def decision(values):
    allowed = [(0.0, 0)]
    for k in (1, 2, 3, 4):
        if np.all(values[:k] >= -1) and np.mean(values[k - 1]) >= 0.25:
            allowed.append((float(np.tanh(np.mean(values[k - 1]) / 0.5)), -k))
    return -max(allowed)[1]


def scores(choices, y, lower, upper):
    zero = np.zeros((6, 1, 2))
    gains = np.concatenate([zero, y], axis=1)[np.arange(6), choices]
    states = []
    for i, k in enumerate(choices):
        states.append(
            "within"
            if k == 0 or np.all(lower[i, :k] >= -1)
            else "violation"
            if np.any(upper[i, :k] < -1)
            else "unknown"
        )
    return np.r_[gains.mean(), gains.mean(0)], states, gains


def main():
    started = time.perf_counter()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    assert not (out / "independent-verification.json").exists(), "Completed audit is immutable"
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    with np.load(out / "causal_inputs.npz") as a:
        losses, health = a["losses"], a["geometry"]
    with np.load(ROOT / config.reference / "warm_features.npz") as a:
        warm = a["bank"]
    with np.load(ROOT / config.reference / "reference_targets.npz") as a:
        domains = a["domains"]
    for i, source in enumerate(config.sources):
        directory = ROOT / config.guard / "runs" / source / "seed_5003"
        with np.load(directory / "banks.npz") as a:
            np.testing.assert_allclose(health[i], geometry(a["128"], warm, domains), atol=1e-11)
        past = np.array(read(directory / "parent-path.json")["trace"])[120:128, 0]
        np.testing.assert_allclose(losses[i], [past.mean(), np.polyfit(np.arange(8), past, 1)[0]], atol=1e-12)
    with np.load(out / "offline_targets.npz") as a:
        targets, lower, upper = a["gain"], a["lower"], a["upper"]
    stage = read(ROOT / config.outcomes / "stage_512.json")
    for i, episode in enumerate(stage["episodes"]):
        assert episode["source"] == config.sources[i]
        for j, point in enumerate(episode["points"][1:]):
            np.testing.assert_array_equal(targets[i, j], point["acquired"]["mean_gain_pp"][1:])
            for array, key in ((lower, "lower"), (upper, "upper")):
                np.testing.assert_array_equal(array[i, j], point["acquired"]["crossed"][key][1:])
    y = targets.reshape(24, 2)
    h = np.tile([0.25, 0.5, 0.75, 1], 6)
    cheap = np.c_[h, h**2, h[:, None] * np.repeat(losses, 4, axis=0)]
    full = np.c_[cheap, h[:, None] * np.repeat(health, 4, axis=0)]
    saved, summary = read(out / "forecasts.json"), read(out / "summary.json")
    errors, reconstructed, envelope_checks = {}, {}, 0
    for name, prediction in saved["predictions"].items():
        result = np.zeros((24, 2))
        for held, fold in enumerate(saved["folds"]):
            train = np.flatnonzero(np.arange(24) // 4 != held)
            query = np.arange(held * 4, held * 4 + 4)
            np.testing.assert_array_equal(fold["train_rows"], train)
            np.testing.assert_array_equal(fold["held_rows"], query)
            operator = np.asarray(saved["operators"][name])
            assert not operator[np.ix_(query, query)].any()
            if name == "zero":
                continue
            if name == "horizon_mean":
                forecast = targets[np.arange(6) != held].mean(0)
            else:
                x = cheap if name == "controls" else full
                if name.startswith("health_sham_"):
                    groups = np.array([g for g in range(6) if g != held])
                    rng = np.random.default_rng(91601 + 1000 * held + int(name[-2:]))
                    while True:
                        donors = rng.permutation(groups)
                        if np.all(donors != groups):
                            break
                    assert fold["sham_donors"][name] == dict(zip(map(str, groups), map(int, donors), strict=True))
                    sham = health.copy()
                    sham[groups] = health[donors]
                    x = np.c_[cheap, h[:, None] * np.repeat(sham, 4, axis=0)]
                forecast = fit(x[train], y[train], x[query])
                model = fold["models"][name]
                np.testing.assert_allclose(model["center"], x[train].mean(0), atol=1e-12)
                scale = x[train].std(0)
                scale[scale < 1e-12] = 1
                np.testing.assert_allclose(model["scale"], scale, atol=1e-12)
                direct = np.asarray(model["intercept"]) + (x[query] - model["center"]) / model["scale"] @ model["beta"]
                np.testing.assert_allclose(direct, forecast, atol=1e-10)
            result[query] = forecast
            np.testing.assert_allclose(operator[query] @ y, forecast, atol=1e-10)
            if name in ("health", "controls", "horizon_mean"):
                residuals = []
                for inner in (g for g in range(6) if g != held):
                    itrain = train[train // 4 != inner]
                    iq = np.arange(4 * inner, 4 * inner + 4)
                    pred = (
                        targets[[g for g in range(6) if g not in (held, inner)]].mean(0)
                        if name == "horizon_mean"
                        else fit(x[itrain], y[itrain], x[iq])
                    )
                    residuals.append(abs(pred - y[iq]))
                np.testing.assert_allclose(
                    np.array(saved["envelope_half_width_pp"][name])[query], np.max(residuals, axis=0), atol=1e-10
                )
                envelope_checks += 1
        np.testing.assert_allclose(prediction, result, atol=1e-10)
        reconstructed[name] = result.reshape(6, 4, 2)
        error = (result - y).reshape(6, 4, 2)
        errors[name] = (error**2).mean(1)
        metric = summary["metrics"][name]
        np.testing.assert_allclose(metric["rmse_pp"], np.sqrt((error**2).mean((0, 1))), atol=1e-10)
        np.testing.assert_allclose(metric["source_mse"], errors[name], atol=1e-10)
        np.testing.assert_allclose(metric["bias_pp"], error.mean((0, 1)), atol=1e-10)
    shams = [n for n in errors if n.startswith("health_sham_")]
    winners = [min(shams, key=lambda n: errors[n][:, d].mean()) for d in range(2)]
    assert winners == summary["best_sham_per_domain"]
    passed = []
    for name, comparison in summary["comparisons"].items():
        baseline = (
            np.column_stack([errors[n][:, d] for d, n in enumerate(winners)])
            if name == "best_geometry_sham"
            else errors[name]
        )
        diff = baseline - errors["health"]
        reduction = 1 - errors["health"].mean(0) / baseline.mean(0)
        low = diff.mean(0) - t.ppf(1 - 0.05 / 12, 5) * diff.std(0, ddof=1) / np.sqrt(6)
        np.testing.assert_allclose(comparison["fractional_mse_reduction"], reduction, atol=1e-10)
        gate = (reduction >= 0.1) & ((diff > 0).sum(0) >= 5) & (low > 0)
        np.testing.assert_array_equal(comparison["passed_domains"], gate)
        passed.append(gate)
    calibration = (np.array(summary["metrics"]["health"]["rmse_pp"]) <= 1) & (
        abs(np.array(summary["metrics"]["health"]["bias_pp"])) <= 0.5
    )
    assert summary["exploratory_prediction_gate"] == bool(calibration.all() and np.all(passed))
    oracle = []
    for episode in stage["episodes"]:
        points = episode["points"]
        allowed = [0] + [
            k
            for k in range(1, 5)
            if all(min(p["acquired"]["crossed"]["lower"][1:]) >= -1 for p in points[: k + 1])
            and points[k]["acquired"]["crossed"]["lower"][0] >= 0.25
        ]
        oracle.append(max(allowed, key=lambda k: (points[k]["acquired"]["crossed"]["lower"][0], -k)))
    choices = {"hold": [0] * 6, "always_train": [4] * 6, "acquired_oracle": oracle}
    for name in ("health", "controls", "horizon_mean"):
        p = reconstructed[name]
        width = np.asarray(saved["envelope_half_width_pp"][name]).reshape(6, 4, 2)
        for suffix, values in (("point", p), ("envelope", p - width)):
            choices[f"{name}_{suffix}"] = [decision(v) for v in values]
        np.testing.assert_allclose(
            summary["metrics"][name]["residual_envelope_coverage_fraction"], (abs(p - targets) <= width).mean((0, 1))
        )
    for name, choice in choices.items():
        result = summary["rules"][name]
        assert result["updates_by_history"] == [8 * k for k in choice]
        assert result["total_updates"] == 8 * sum(choice)
        gains, states, source_gains = scores(choice, targets, lower, upper)
        np.testing.assert_allclose(result["mean_gain_pp"], gains, atol=1e-10)
        np.testing.assert_allclose(result["source_domain_gain_pp"], source_gains, atol=1e-10)
        assert result["path_status"] == states
        for state in ("within", "violation", "unknown"):
            assert result[f"{state}_histories"] == states.count(state)
        assignments = [scores(p, targets, lower, upper) for p in itertools.permutations(choice)]
        np.testing.assert_allclose(
            result["matched_random"]["mean_gain_pp"], np.mean([a[0] for a in assignments], axis=0), atol=1e-10
        )
        np.testing.assert_allclose(
            result["matched_random"]["expected_violations"], np.mean([a[1].count("violation") for a in assignments])
        )
        assert result["matched_random"]["assignment_permutations"] == 720
    assert not summary["policy_qualified"] and not summary["controller_activated"]
    assert not summary["inherited_precision_pass"]
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "forecast_models_rebuilt": len(errors),
            "outer_folds_per_model": 6,
            "nested_envelopes_rebuilt": envelope_checks,
            "stopping_rules_rebuilt": len(choices),
            "matched_random_assignments_per_rule": 720,
            "parent_profiles_rebuilt": 6,
            "new_ssl_updates": 0,
            "new_canonical_classifier_fits": 0,
            "source_sha256": digest(__file__),
            "seconds": time.perf_counter() - started,
        },
    )
    print({"independent_verification": "passed", "models": len(errors)}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
