"""Independent raw-outcome, repeat-aggregation and bootstrap reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_signal import read
from scripts.verify_single_image_impact_analysis import raw_metrics


def independent_pairs(e):
    def centered(x):
        return x - x.sum(1, keepdims=True) / 3

    pairs = {}
    for budget in (1, 2, 3):
        a = e[:, :, :budget].sum((2, 3)) / (budget * 3)
        b = e[:, :, 3 : 3 + budget].sum((2, 3)) / (budget * 3)
        pairs[f"views_{budget}_raw"] = (a, b)
        pairs[f"views_{budget}_centered"] = (centered(a), centered(b))
        if budget == 3:
            pairs["current_image_views_3"] = (a[:, :1], b[:, :1])
    one_a, one_b, two_a, two_b, joint_a, joint_b = [], [], [], [], [], []
    for held in range(3):
        other = [r for r in range(3) if r != held]
        held_mean = e[:, :, :, held].sum(2) / 6
        for source in other:
            one_a.append(e[:, :, :, source].sum(2) / 6)
            one_b.append(held_mean)
        two_a.append((e[:, :, :, other[0]].sum(2) + e[:, :, :, other[1]].sum(2)) / 12)
        two_b.append(held_mean)
        for block in (0, 1):
            source_draws = slice(block * 3, block * 3 + 3)
            target_draws = slice((1 - block) * 3, (1 - block) * 3 + 3)
            joint_a.append(e[:, :, source_draws, held].sum(2) / 3)
            joint_b.append((e[:, :, target_draws, other[0]].sum(2) + e[:, :, target_draws, other[1]].sum(2)) / 6)
    for name, a, b in (
        ("readers_1_centered", one_a, one_b),
        ("readers_2_centered", two_a, two_b),
        ("joint_centered", joint_a, joint_b),
    ):
        pairs[name] = (centered(np.stack(a, axis=-1)), centered(np.stack(b, axis=-1)))
    return pairs


def independent_correlation_ci(a, b, groups, draws):
    # Group sufficient statistics produce the same clustered bootstrap without
    # using the scoring code's materialized resampled effect matrices.
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    moments = []
    for indices in members:
        x, y = a[indices].flatten(), b[indices].flatten()
        moments.append([len(x), x.sum(), y.sum(), x @ x, y @ y, x @ y])
    m = np.array(moments)[draws].sum(1)
    n, sx, sy, sxx, syy, sxy = m.T
    xx, yy = sxx - sx**2 / n, syy - sy**2 / n
    xy = sxy - sx * sy / n
    denom = np.sqrt(np.maximum(xx, 0) * np.maximum(yy, 0))
    degenerate = (xx < abs(sxx) * 1e-11) | (yy < abs(syy) * 1e-11)
    for i in np.flatnonzero(degenerate):
        indices = np.concatenate([members[g] for g in draws[i]])
        x, y = a[indices].flatten(), b[indices].flatten()
        x, y = x - x.mean(), y - y.mean()
        denom[i], xy[i] = np.linalg.norm(x) * np.linalg.norm(y), x @ y
    valid = denom > 1e-28
    values = xy[valid] / denom[valid]
    return (np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None), int(valid.sum())


def verify_result(a, b, saved, groups, draws, tolerance):
    x, y = a.flatten(), b.flatten()
    error = x - y
    count = len(x)
    corr = (
        float(np.corrcoef(x, y)[0, 1]) if np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()) > 1e-28 else None
    )
    rmse, mae = np.linalg.norm(error) / np.sqrt(count), abs(error).sum() / count
    zr = np.linalg.norm(np.r_[x, y]) / np.sqrt(2 * count)
    za = (abs(x).sum() + abs(y).sum()) / (2 * count)
    expected = {
        "rmse": rmse,
        "mae": mae,
        "symmetric_zero_rmse": zr,
        "symmetric_zero_mae": za,
        "cross_repeat_signal_power": x @ y / count,
        "half_difference_noise_power": (error @ error) / (2 * count),
        "observed_power": (x @ x + y @ y) / (2 * count),
        "zero_fraction_a": np.mean(abs(x) <= tolerance),
        "zero_fraction_b": np.mean(abs(y) <= tolerance),
    }
    for key, value in expected.items():
        np.testing.assert_allclose(saved[key], value, rtol=1e-10, atol=1e-15, err_msg=key)
    for key, value in (
        ("correlation", corr),
        ("rmse_ratio_to_zero", rmse / zr if zr > 0 else None),
        ("mae_ratio_to_zero", mae / za if za > 0 else None),
    ):
        if value is None:
            assert saved[key] is None
        else:
            np.testing.assert_allclose(saved[key], value, rtol=1e-9, atol=1e-11, err_msg=key)
    both = (abs(x) > tolerance) & (abs(y) > tolerance)
    assert saved["both_nonzero_cases"] == int(both.sum())
    direction = np.mean(np.sign(x[both]) == np.sign(y[both])) if both.any() else None
    assert saved["nonzero_direction_agreement"] == direction
    ci, valid = independent_correlation_ci(a, b, groups, draws)
    assert valid == saved["valid_bootstrap_draws"]
    if ci is None:
        assert saved["correlation_ci95"] is None
    else:
        np.testing.assert_allclose(saved["correlation_ci95"], ci, rtol=1e-8, atol=1e-8)
    return {
        "correlation_ci95": ci,
        "rmse_ratio_to_zero": rmse / zr if zr > 0 else None,
        "mae_ratio_to_zero": mae / za if za > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.resolve()
    lock = read(root / "input_lock.json")
    prior = Path(lock["prior"])
    supports = lock["prior_lock"]["support_indices"]
    keys = sorted(supports)
    query_ids = set(lock["prior_lock"]["query_indices"])
    for i, key in enumerate(keys):
        assert len(supports[key]) == 50 and not set(supports[key]) & query_ids
        for other in keys[i + 1 :]:
            assert not set(supports[key]) & set(supports[other])
    for path, expected in {**lock["prior_lock"]["hashes"], **lock["hashes"]}.items():
        assert digest(Path(path)) == expected, path
    torch.set_num_threads(1)
    panel = torch.load(prior / "panel.pt", weights_only=True)
    labels, domains = panel["query_labels"].numpy(), panel["query_domains"].numpy()
    warm = torch.from_numpy(np.load(prior / "warm_bank.npy")).double()
    collection = read(root / "collection.json")
    parents = sorted(collection["parents"], key=lambda r: (r["history"], r["origin"]))
    parent_indices = {(r["history"], r["origin"]): i for i, r in enumerate(parents)}
    data = np.load(root / "effects.npz", allow_pickle=False)
    metrics, panels = data["metrics"].tolist(), data["reader_panels"].tolist()
    actual, zero = np.zeros_like(data["actual"]), np.zeros_like(data["zero"])
    counts = {"raw_readouts": 0, "reader_values": 0, "geometry_values": 0}

    def reconstruct(path):
        reconstructed, c = raw_metrics(path, warm, labels, domains)
        counts["raw_readouts"] += 1
        for k, v in c.items():
            counts[k] += v
        return np.array([[reconstructed[p][m] for m in metrics] for p in panels])

    for i, parent in enumerate(parents):
        zero[i] = reconstruct(root / parent["zero_readout"])
    for n, row in enumerate(collection["records"]):
        i = parent_indices[row["history"], row["origin"]]
        actual[i, row["candidate"], row["draw_id"]] = reconstruct(root / row["readout"])
        if n % 50 == 0:
            print(f"Raw repeated-image readouts checked: {n + 1}/432", flush=True)
    np.testing.assert_allclose(data["actual"], actual, rtol=0, atol=2e-11)
    np.testing.assert_allclose(data["zero"], zero, rtol=0, atol=2e-11)
    np.testing.assert_array_equal(data["effects"], data["actual"] - data["zero"][:, None, None])
    # Raw reconstruction can differ by a few ULPs in SVD geometry. Aggregation
    # is audited on the exact serialized measurements, whose raw audit passed.
    effects = data["effects"]
    replicates = np.load(root / "measurement-replicates.npz", allow_pickle=False)
    summary = read(root / "summary.json")
    settings = lock["settings"]
    draws = np.random.default_rng(settings["bootstrap_seed"]).integers(0, 6, (settings["bootstrap_draws"], 6))
    checked = 0
    for context, selected in (
        ("within_clip", data["origins"] % 96 != 0),
        ("clip_join", data["origins"] % 96 == 0),
        ("all", np.ones(24, bool)),
    ):
        for j, metric in enumerate(metrics):
            outcomes = summary["by_context"][context][metric]
            audited = {}
            for name, (a, b) in independent_pairs(effects[selected, ..., j]).items():
                prefix = f"{context}__{metric}__{name}"
                np.testing.assert_allclose(replicates[prefix + "__a"], a, rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(replicates[prefix + "__b"], b, rtol=1e-10, atol=1e-14)
                audited[name] = verify_result(
                    replicates[prefix + "__a"],
                    replicates[prefix + "__b"],
                    outcomes["comparisons"][name],
                    data["groups"][selected],
                    draws,
                    settings["tie_tolerance"],
                )
                checked += 1
            current = effects[selected, 0, ..., j].sum((1, 2)) / 18
            np.testing.assert_allclose(outcomes["current_image_parent_effects"], current, rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(outcomes["mean_current_image_effect"], current.mean(), rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(
                outcomes["rms_current_image_effect"],
                np.linalg.norm(current) / np.sqrt(len(current)),
                rtol=1e-10,
                atol=1e-14,
            )
            gate = True
            for name in ("views_3_centered", "joint_centered"):
                a = audited[name]
                gate = (
                    gate
                    and a["correlation_ci95"] is not None
                    and a["correlation_ci95"][0] > 0
                    and a["rmse_ratio_to_zero"] is not None
                    and a["rmse_ratio_to_zero"] < 1
                    and a["mae_ratio_to_zero"] < 1
                )
            assert bool(gate) == outcomes["passes_measurement_prerequisite"]
    assert read(root / "physical-verification.json")["passed"]
    atomic_json(
        root / "analysis-verification.json",
        {
            "passed": True,
            **counts,
            "isolated_effect_values": effects.size,
            "independently_checked_replication_comparisons": checked,
            "independent_bootstrap_intervals": checked,
            "source_sha256": digest(Path(__file__)),
            "raw_auditor_sha256": digest(Path(__file__).with_name("verify_single_image_impact_analysis.py")),
            "all_prerequisite_decisions_independently_recomputed": True,
            "reader_panels_disjoint_and_query_excluded": True,
            "supervised_fits_independently_retrained": False,
        },
    )


if __name__ == "__main__":
    main()
