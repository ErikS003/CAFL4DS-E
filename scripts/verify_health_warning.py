"""Independently reconstruct warning targets, CKA, predictions and grouped fits."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_fit(points, fields, target):
    x = np.asarray([[p[f] for f in fields] for p in points], dtype=float).reshape(18, len(fields))
    y = np.asarray([p[target] for p in points])
    prediction, slopes, errors = np.zeros(18), {}, {}
    for group in sorted({p["stream"] for p in points}):
        held = np.array([p["stream"] == group for p in points])
        center, scale = x[~held].mean(0), x[~held].std(0)
        scale[scale == 0] = 1
        design = np.column_stack(((x[~held] - center) / scale, np.ones(sum(~held))))
        penalty = np.diag([1.0] * len(fields) + [0.0])
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[~held])
        prediction[held] = np.column_stack(((x[held] - center) / scale, np.ones(sum(held)))) @ beta
        slopes[group] = dict(zip(fields, beta[:-1], strict=True))
        errors[group] = float(np.mean((prediction[held] - y[held]) ** 2))
    return prediction, slopes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    root = Path(__file__).resolve().parents[1]
    lock, collection, summary = [read(out / name) for name in ("input_lock.json", "collection.json", "summary.json")]
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    assert read(out / "verification.json")["passed"] and collection["complete"]
    rows = collection["records"]
    assert len(rows) == 63 and sum(r["role"] == "six_stream" for r in rows) == 54
    assert len({r["stream"] for r in rows if r["role"] == "six_stream"}) == 6
    assert len(lock["reference"]["images"]) == 256
    with np.load(root / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    count, cka_count, new_readouts, aliases = 0, 0, 0, 0
    for row in rows:
        assert row["exact_replay"] and len(row["trace"]) == 32
        expected_parent = row["point"] // (64 if row["role"] == "six_stream" else 32)
        assert row["parent_updates"] == expected_parent
        with np.load(row["embeddings_path"], allow_pickle=False) as arrays:
            start = arrays["at_0"].astype(np.float64)
            for j in (0, 4, 8, 16, 32):
                observation = row["landmarks"][str(j)]
                z = arrays[f"at_{j}"].astype(np.float64)
                assert z.shape == start.shape == (256, 96) and np.isfinite(z).all()
                x, y = start - start.mean(0), z - z.mean(0)
                cka = 1 - np.sum((y.T @ x) ** 2) / np.sqrt(np.sum((x.T @ x) ** 2) * np.sum((y.T @ y) ** 2))
                assert abs(max(0, cka) - observation["geometry"]["cka_drift"]) < 3e-5
                cka_count += 1
                if j:
                    assert abs(np.mean([v[0] for v in row["trace"][j - 4 : j]]) - observation["recent_loss"]) < 1e-12
                else:
                    assert observation["recent_loss"] is None and observation["geometry"]["cka_drift"] < 1e-6
                directory = Path(observation["readout"])
                metadata = read(directory / "readout.json")
                assert metadata["ssl_state_unchanged"]
                alias_expected = (
                    j == 0 or (row["role"] == "six_stream" and j == 16) or (row["role"] == "control" and j == 32)
                )
                assert observation["aliased"] == alias_expected
                aliases += int(alias_expected)
                new_readouts += int(not alias_expected)
                with np.load(directory / "predictions.npz", allow_pickle=False) as prediction:
                    for reader in ("finetune", "linear", "knn"):
                        metric_values = {}
                        for metric, mask in (
                            ("balanced", np.ones(400, dtype=bool)),
                            ("A", domains == 0),
                            ("B", domains == 1),
                        ):
                            values = []
                            for draw, panel in metadata["panels"].items():
                                correct = prediction[f"{draw}_{reader}"] == labels
                                assert correct.shape == (400,)
                                value = float(correct[mask].mean())
                                assert abs(value - panel[reader][metric]) < 1e-6
                                values.append(100 * value)
                            metric_values[metric] = float(np.mean(values))
                        assert all(
                            abs(metric_values[k] - observation["scores"][reader][k]) < 1e-10 for k in metric_values
                        )
                        count += 3
    assert new_readouts == 189 and aliases == 126
    grouped = sorted({(r["stream"], r["checkpoint"]) for r in rows if r["role"] == "six_stream"})
    fields_by_model = {
        "mean": (),
        "count_loss": ("updates", "recent_loss"),
        "cka": ("updates", "recent_loss", "cka_drift"),
    }
    for j in (4, 8, 16):
        result = summary["forecasts"][str(j)]
        points = []
        for stream, checkpoint in grouped:
            selected = [r for r in rows if r["stream"] == stream and r["checkpoint"] == checkpoint]
            assert len(selected) == 3 and len({r["seed"] for r in selected}) == 3
            capability = {
                n: {
                    m: np.mean([r["landmarks"][str(n)]["scores"]["finetune"][m] for r in selected])
                    for m in ("balanced", "A", "B")
                }
                for n in (0, j, 32)
            }
            delta = {m: capability[32][m] - capability[j][m] for m in ("balanced", "A", "B")}
            p = {
                "stream": stream,
                "checkpoint": checkpoint,
                **delta,
                "damage": -min(delta["A"], delta["B"]),
                "balanced_damage": -delta["balanced"],
                "already_damage": -min(capability[j][m] - capability[0][m] for m in ("A", "B")),
                "updates": np.mean([r["parent_updates"] + j for r in selected]),
                "recent_loss": np.mean([r["landmarks"][str(j)]["recent_loss"] for r in selected]),
                "cka_drift": np.mean([r["landmarks"][str(j)]["geometry"]["cka_drift"] for r in selected]),
            }
            points.append(p)
        for calculated, saved in zip(points, result["points"], strict=True):
            for key, value in calculated.items():
                assert (value == saved[key]) if isinstance(value, str) else abs(value - saved[key]) < 1e-10
        fits = {}
        for target, block in (
            ("damage", "models"),
            ("already_damage", "already_incurred_diagnostic"),
            ("balanced_damage", "balanced_damage_diagnostic"),
        ):
            for model, fields in fields_by_model.items():
                prediction, slopes, errors = independent_fit(points, fields, target)
                saved = result[block][model]
                assert np.allclose(prediction, saved["predictions"], rtol=0, atol=1e-9)
                assert abs(np.mean(list(errors.values())) - saved["mse"]) < 1e-9
                for g in errors:
                    assert abs(errors[g] - saved["source_mse"][g]) < 1e-9
                    assert all(abs(v - saved["coefficients"][g][f]) < 1e-9 for f, v in slopes[g].items())
                if target == "damage":
                    fits[model] = (slopes, errors)
        incremental = True
        for baseline in ("mean", "count_loss"):
            delta = np.array([fits[baseline][1][g] - fits["cka"][1][g] for g in fits[baseline][1]])
            width = t.ppf(0.9875, 5) * delta.std(ddof=1) / np.sqrt(6)
            reduction = 1 - np.mean(list(fits["cka"][1].values())) / np.mean(list(fits[baseline][1].values()))
            saved = result["contrasts"][baseline]
            assert np.allclose(saved["adjusted_ci95"], [delta.mean() - width, delta.mean() + width], atol=1e-9, rtol=0)
            assert abs(saved["reduction_fraction"] - reduction) < 1e-9
            incremental &= reduction >= 0.1 and delta.mean() - width > 0
        population = {
            "large_future_damage": sum(p["damage"] >= 1 for p in points),
            "no_future_domain_damage": sum(p["damage"] <= 0 for p in points),
            "beneficial_continuations": sum(p["balanced"] >= 0.5 for p in points),
        }
        assert population == result["population"]
        sign = sum(s["cka_drift"] > 0 for s in fits["cka"][0].values()) >= 5
        assert result["gates"] == {
            "incremental_prediction": bool(incremental),
            "positive_cka_folds": sign,
            "two_sided_population": all(v >= 2 for v in population.values()),
        }
        assert result["qualified"] == all(result["gates"].values())
    assert summary["qualified"] == summary["forecasts"]["8"]["qualified"]
    result = {
        "passed": True,
        "cka_scalars": cka_count,
        "prediction_sets": count,
        "new_readouts": new_readouts,
        "aliased_readouts": aliases,
        "grouped_forecasts": 27,
        "history_groups": 6,
        "controls_excluded_from_fits": True,
        "remaining_effect_targets": 54,
        "hashes": len(lock["hashes"]),
    }
    atomic_json(out / "independent-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
