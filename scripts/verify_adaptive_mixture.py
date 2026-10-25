"""Independent causal selector, replay-dose and prediction/statistical audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_plans(stream, seed):
    base = seed + int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(base + 74111)
    pool = list(range(512))
    processed = 512
    result = []
    for block in range(23):
        boundary = 512 + block * 512
        while processed < boundary:
            slot = int(rng.integers(processed + 1))
            if slot < 512:
                pool[slot] = processed
            processed += 1
        ids = np.random.default_rng(base + 75203 + block).choice(pool, size=64, replace=False).tolist()
        assert len(set(ids)) == 64 and max(ids) < boundary
        result.append(ids)
    return result


def verify_selector(row, features):
    plans = reconstruct_plans(row["stream"], row["seed"])
    assert plans == row["plans"]
    base = row["seed"] + int(hashlib.sha256(row["stream"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(base + 76349)
    recent = []
    calibration = []
    threshold = None
    remaining = 0
    cursor = 0
    old_total = 0
    for j, record in enumerate(row["trace"]):
        start = j * 64
        score = None
        assert record["step"] == j and record["offset"] == start
        if row["arm"] == "adaptive" and j:
            assert max(recent) < start
            now = features[start : start + 64].astype(float)
            old = features[recent].astype(float)
            now /= np.linalg.norm(now, axis=1, keepdims=True)
            old /= np.linalg.norm(old, axis=1, keepdims=True)
            a, b = now.mean(0), old.mean(0)
            score = float(np.clip(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 2))
            assert abs(score - record["score"]) < 1e-11
        else:
            assert record["score"] is None
        if j < 8:
            if score is not None:
                calibration.append(score)
            amount = target = 0
            historical = []
        elif row["arm"] == "current":
            amount = target = 0
            historical = []
        else:
            block, position = divmod(j - 8, 8)
            if position == 0:
                remaining, cursor = 64, 0
            if row["arm"] == "fixed":
                target = 8
            elif row["arm"] == "random":
                target = 16 if rng.random() < 0.5 else 0
            else:
                if threshold is None:
                    sorted_scores = sorted(calibration)
                    threshold = sorted_scores[5] + 0.4 * (sorted_scores[6] - sorted_scores[5]) + 1e-8
                target = 16 if score > threshold else 0
            amount = min(max(target, max(0, remaining - 16 * (7 - position))), min(16, remaining))
            historical = plans[block][cursor : cursor + amount]
            cursor += amount
            remaining -= amount
            if position == 7:
                assert remaining == 0
        count = 32 - amount
        current = [start + (k * 31) // (count - 1) for k in range(count)]
        assert record["current"] == current and record["old"] == historical
        assert record["historical_count"] == amount and record["target_count"] == target
        assert record["quota_forced"] == (amount != target) and record["quota_remaining"] == remaining
        if threshold is None:
            assert record["threshold"] is None
        else:
            assert abs(threshold - record["threshold"]) < 1e-11
        assert len(current) + len(historical) == 32 and all(i < start for i in historical)
        recent = (recent + historical + current)[-256:]
        old_total += amount
    assert old_total == (0 if row["arm"] == "current" else 1472)
    for block, ids in enumerate(plans):
        selected = [i for r in row["trace"][8 + 8 * block : 16 + 8 * block] for i in r["old"]]
        assert selected == ([] if row["arm"] == "current" else ids)


def score_directory(directory, labels, domains):
    metadata = read(directory / "readout.json")
    assert metadata["ssl_state_unchanged"]
    result = {}
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
        for reader in ("finetune", "linear", "knn"):
            result[reader] = {}
            for metric, mask in (("balanced", np.ones(400, bool)), ("A", domains == 0), ("B", domains == 1)):
                values = []
                for draw in (51011, 51017, 51023):
                    correct = arrays[f"{draw}_{reader}"] == labels
                    assert correct.shape == (400,)
                    value = float(correct[mask].mean())
                    assert abs(value - metadata["panels"][str(draw)][reader][metric]) < 1e-6
                    values.append(100 * value)
                result[reader][metric] = float(np.mean(values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    root = Path(__file__).resolve().parents[1]
    lock = read(out / "input_lock.json")
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    assert read(out / "verification.json")["passed"]
    collection = read(out / "collection.json")
    assert collection["complete"] and len(collection["records"]) == 72
    rows = [read(Path(p)) for p in collection["records"]]
    expected = {
        (s["name"], seed, arm)
        for s in lock["streams"]
        for seed in (5003, 5009, 5011)
        for arm in ("current", "fixed", "random", "adaptive")
    }
    assert {(r["stream"], r["seed"], r["arm"]) for r in rows} == expected
    with np.load(root / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    warm = {
        seed: score_directory(root / "outputs/coverage-admission/20261013-gpu/warm" / f"seed_{seed}", labels, domains)
        for seed in (5003, 5009, 5011)
    }
    scores = {}
    prediction_sets = 27
    for row in rows:
        assert row["exact_replay"] and len(row["trace"]) == 192
        features = np.load(row["feature_path"], allow_pickle=False) if row["arm"] == "adaptive" else None
        if features is not None:
            assert features.shape == (12288, 96) and np.isfinite(features).all()
        verify_selector(row, features)
        pattern = row["pattern"]
        transitions = [1024 * i for i in range(1, len(pattern)) if pattern[i] != pattern[i - 1]]
        assert row["readouts"]["before_return"]["clock"] == transitions[1]
        for horizon, readout in row["readouts"].items():
            assert readout["clock"] == (12288 if horizon == "terminal" else transitions[1])
            assert readout["aliased"] == (row["arm"] == "current" and horizon == "terminal")
            calculated = score_directory(Path(readout["directory"]), labels, domains)
            for reader in calculated:
                for metric in calculated[reader]:
                    assert abs(calculated[reader][metric] - readout["scores"][reader][metric]) < 1e-10
            scores[row["stream"], row["seed"], row["arm"], horizon] = calculated
            prediction_sets += 9
    for stream in lock["streams"]:
        for seed in (5003, 5009, 5011):
            repeated = [r for r in rows if r["stream"] == stream["name"] and r["seed"] == seed]
            assert len({r["common_calibration_fingerprint"] for r in repeated}) == 1
    summary = read(out / "summary.json")
    names = sorted({r["stream"] for r in rows})
    checked = 0
    for horizon in ("before_return", "terminal"):
        result = summary["horizons"][horizon]
        for arm in ("current", "fixed", "random", "adaptive", "warm"):
            for reader in ("finetune", "linear", "knn"):
                for metric in ("balanced", "A", "B"):
                    vals = [
                        warm[s][reader][metric] if arm == "warm" else scores[n, s, arm, horizon][reader][metric]
                        for n in names
                        for s in (5003, 5009, 5011)
                    ]
                    assert abs(np.mean(vals) - result["means"][arm][reader][metric]) < 1e-10
        for comparator in ("current", "fixed", "random", "warm"):
            for reader in ("finetune", "linear", "knn"):
                for metric in ("balanced", "A", "B"):
                    delta = np.array(
                        [
                            np.mean(
                                [
                                    scores[n, s, "adaptive", horizon][reader][metric]
                                    - (
                                        warm[s][reader][metric]
                                        if comparator == "warm"
                                        else scores[n, s, comparator, horizon][reader][metric]
                                    )
                                    for s in (5003, 5009, 5011)
                                ]
                            )
                            for n in names
                        ]
                    )
                    for kind, family in (("ordinary", 1), ("adjusted", 12)):
                        saved = result["contrasts"][comparator][reader][metric][kind]
                        width = t.ppf(1 - 0.05 / (2 * family), 5) * delta.std(ddof=1) / np.sqrt(6)
                        assert abs(delta.mean() - saved["mean_pp"]) < 1e-10
                        assert np.allclose(
                            [delta.mean() - width, delta.mean() + width], saved["ci95_pp"], atol=1e-10, rtol=0
                        )
                        checked += 1
    terminal = summary["horizons"]["terminal"]["contrasts"]
    gates = {
        "adaptive_content_lead": all(
            terminal[a]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.5
            and terminal[a]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0
            for a in ("current", "fixed", "random")
        ),
        "useful_ssl": terminal["warm"]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.1
        and terminal["warm"]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0,
        "domain_retention": all(
            terminal[a]["finetune"][m]["adjusted"]["ci95_pp"][0] >= -1
            for a in ("current", "fixed", "random", "warm")
            for m in ("A", "B")
        ),
    }
    assert gates == summary["gates"] and summary["adaptive_mixture_qualified"] == all(gates.values())
    verification = {
        "passed": True,
        "trajectories": 72,
        "causal_update_exposures": 13824,
        "matched_replay_sequences": 1242,
        "prediction_sets": prediction_sets,
        "statistical_intervals": checked,
        "history_groups": 6,
        "quota_and_context_decisions_verified": True,
        "source_hashes": len(lock["hashes"]),
    }
    atomic_json(out / "independent-verification.json", verification)
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
