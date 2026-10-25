"""Independently reconstruct temporal grouping, health and capability statistics on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import sem, t

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest
from scripts.verify_adaptive_mixture import reconstruct_plans
from scripts.verify_prediction_pilots import torch_geometry

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("current", "chronological", "random", "stratified")
SEEDS = (5003, 5009, 5011)


def read(path: Path):
    """Read evidence without mutating experiment artifacts."""
    return json.loads(path.read_text(encoding="utf-8"))


def check(first, second):
    """Require numerical agreement beyond printing precision."""
    if not np.allclose(first, second, atol=1e-9, rtol=1e-9):
        raise ValueError("Independent grouping reconstruction differs")


def independent_plan(stream: str, seed: int, arm: str):
    """Build exposures using explicit stratum/column loops, not collector reshapes."""
    history = reconstruct_plans(stream, seed)
    rows = [
        {"step": j, "available": (j + 1) * 64, "old": [], "current": list(range(j * 64, j * 64 + 32))} for j in range(8)
    ]
    for b in range(23):
        origin = 512 + b * 512
        matrix = [[origin + row * 64 + int(value) for value in np.linspace(0, 31, 24)] for row in range(8)]
        if arm == "current":
            chosen = [list(range(origin + j * 64, origin + j * 64 + 32)) for j in range(8)]
        elif arm == "stratified":
            chosen = [
                [matrix[stratum][column] for stratum in range(8) for column in range(j * 3, j * 3 + 3)]
                for j in range(8)
            ]
        elif arm == "random":
            base = seed + int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)
            values = np.random.default_rng(base + 88169 + b).permutation([i for row in matrix for i in row]).tolist()
            chosen = [values[j * 24 : j * 24 + 24] for j in range(8)]
        else:
            chosen = matrix
        rows += [
            {
                "step": 8 + b * 8 + j,
                "available": origin + 512,
                "old": [] if arm == "current" else history[b][j * 8 : j * 8 + 8],
                "current": chosen[j],
            }
            for j in range(8)
        ]
    return rows


def verify_scores(directory: Path, labels, domains, saved):
    """Reconstruct exact percentages from integer predictions; allow legacy float32 panels."""
    panel_scores = read(directory / "readout.json")["panels"]
    with np.load(directory / "predictions.npz") as predictions:
        for reader in ("finetune", "linear", "knn"):
            for domain, mask in (("balanced", np.ones(len(labels), bool)), ("A", domains == 0), ("B", domains == 1)):
                values = []
                for draw in (51011, 51017, 51023):
                    score = float(np.mean(predictions[f"{draw}_{reader}"][mask] == labels[mask]))
                    if abs(score - panel_scores[str(draw)][reader][domain]) > 5e-8:
                        raise ValueError("Legacy panel differs beyond float32 rounding")
                    values.append(score * 100)
                check(np.mean(values), saved[reader][domain])


def compare_interval(values, saved, family):
    """Use source means and scipy SEM to independently rebuild both interval families."""
    mean = float(np.mean(values))
    width = float(t.isf(0.05 / (2 * family), 5) * sem(values))
    check(mean, saved["mean_pp"])
    check([mean - width, mean + width], saved["ci95_pp"])
    check(values, saved["source_values_pp"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    torch.set_num_threads(4)
    lock = read(out / "input_lock.json")
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Frozen inputs changed")
    if not read(out / "verification.json")["passed"]:
        raise ValueError("Exact learner replay must pass first")
    rows = [read(Path(p)) for p in read(out / "collection.json")["records"]]
    old_rows = [read(Path(p)) for p in read(ROOT / "outputs/adaptive-mixture/20261014-gpu/collection.json")["records"]]
    old_lookup = {
        (r["stream"], r["seed"], "chronological" if r["arm"] == "fixed" else "current"): r
        for r in old_rows
        if r["arm"] in ("current", "fixed")
    }
    expected = {(s["name"], seed, arm) for s in lock["streams"] for seed in SEEDS for arm in ARMS}
    lookup = {(r["stream"], r["seed"], r["arm"]): r for r in rows}
    if len(rows) != 72 or set(lookup) != expected:
        raise ValueError("Incomplete grouping grid")
    warm = np.load(out / "warm_bank.npz")["embeddings"]
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    warm_scores = read(out / "warm_scores.json")
    for seed in SEEDS:
        verify_scores(
            ROOT / f"outputs/coverage-admission/20261013-gpu/warm/seed_{seed}", labels, domains, warm_scores[str(seed)]
        )
    metrics, exposure_rows = 0, 0
    for r in rows:
        if r["arm"] in ("current", "chronological"):
            ancestor = old_lookup[r["stream"], r["seed"], r["arm"]]
            if [x["loss_gradient"] for x in r["trace"]] != [x["loss_gradient"] for x in ancestor["trace"]]:
                raise ValueError("Cached control losses changed")
            if any(
                r["readouts"][name]["fingerprint"] != ancestor["readouts"][name]["fingerprint"]
                for name in ("before_return", "terminal")
            ):
                raise ValueError("Cached control endpoint changed")
        plan = independent_plan(r["stream"], r["seed"], r["arm"])
        if not r["complete"] or not r["exact_replay"] or len(r["trace"]) != 192:
            raise ValueError("Incomplete trajectory proof")
        if any(digest(Path(p)) != h for p, h in r["artifact_hashes"].items()):
            raise ValueError("Committed trajectory file changed")
        for target, actual in zip(plan, r["trace"], strict=True):
            if any(actual[k] != v for k, v in target.items()):
                raise ValueError("Independent exposure arrangement differs")
            ids = sorted(actual["current"])
            counts = Counter(i // 64 for i in ids)
            n = len(ids)
            m = actual["manipulation"]
            check(len(counts), m["occupied_strata"])
            check(max(counts.values()), m["largest_stratum_count"])
            check(sum(v * (v - 1) for v in counts.values()) / (n * (n - 1)), m["same_stratum_pair_fraction"])
            check(
                sum((2 * j - n + 1) * i for j, i in enumerate(ids)) / (n * (n - 1) / 2),
                m["mean_pair_arrival_separation"],
            )
            check(actual["available"] - 1 - np.mean(ids), m["mean_age_at_release"])
            if len(set(actual["old"] + ids)) != 32 or max(actual["old"] + ids) >= actual["available"]:
                raise ValueError("Exposure causality failed")
            exposure_rows += 1
        if sum(len(x["old"]) for x in plan) != (0 if r["arm"] == "current" else 1472):
            raise ValueError("Historical dose differs")
        for h in r["health"].values():
            z = np.load(h["path"])["embeddings"]
            independent = torch_geometry(z, warm)
            for name, value in independent.items():
                check(value, h["values"][name])
                metrics += 1
        for readout in r["readouts"].values():
            verify_scores(Path(readout["directory"]), labels, domains, readout["scores"])
        if r["arm"] != "current":
            reference = lookup[r["stream"], r["seed"], "chronological"]
            for start in range(8, 192, 8):
                if Counter(i for x in r["trace"][start : start + 8] for i in x["current"]) != Counter(
                    i for x in reference["trace"][start : start + 8] for i in x["current"]
                ):
                    raise ValueError("Current identity multiset changed")
                if [x["old"] for x in r["trace"][start : start + 8]] != [
                    x["old"] for x in reference["trace"][start : start + 8]
                ]:
                    raise ValueError("Replay identity/order changed")
    for stream in lock["streams"]:
        for seed in SEEDS:
            if len({lookup[stream["name"], seed, a]["calibration_fingerprint"] for a in ARMS}) != 1:
                raise ValueError("Common calibration states differ")
    summary = read(out / "summary.json")
    groups = sorted({r["stream"] for r in rows})
    checks = 0
    for horizon in ("before_return", "terminal"):

        def quality(g, a, reader, d, s):
            return (
                warm_scores[str(s)][reader][d]
                if a == "warm"
                else lookup[g, s, a]["readouts"][horizon]["scores"][reader][d]
            )

        saved = summary["horizons"][horizon]
        for arm in (*ARMS, "warm"):
            for reader in ("finetune", "linear", "knn"):
                for d in ("balanced", "A", "B"):
                    check(
                        np.mean([quality(g, arm, reader, d, s) for g in groups for s in SEEDS]),
                        saved["means"][arm][reader][d],
                    )
        for comparator in ("current", "chronological", "random", "warm"):
            for reader in ("finetune", "linear", "knn"):
                for d in ("balanced", "A", "B"):
                    values = [
                        np.mean(
                            [
                                quality(g, "stratified", reader, d, s) - quality(g, comparator, reader, d, s)
                                for s in SEEDS
                            ]
                        )
                        for g in groups
                    ]
                    for family, key in ((1, "ordinary"), (12, "adjusted")):
                        compare_interval(values, saved["contrasts"][comparator][reader][d][key], family)
                        checks += 1
        values = [
            np.mean(
                [
                    quality(g, "random", "finetune", "balanced", s)
                    - quality(g, "chronological", "finetune", "balanced", s)
                    for s in SEEDS
                ]
            )
            for g in groups
        ]
        compare_interval(values, saved["secondary_random_minus_chronological"], 1)
    c = summary["horizons"]["terminal"]["contrasts"]
    gains = [c[a]["finetune"]["balanced"]["adjusted"] for a in ("current", "chronological", "random")]
    useful = c["warm"]["finetune"]["balanced"]["adjusted"]
    gates = {
        "structured_content_value": all(x["mean_pp"] >= 0.25 and x["ci95_pp"][0] > 0 for x in gains),
        "useful_ssl": useful["mean_pp"] >= 0.1 and useful["ci95_pp"][0] > 0,
        "terminal_domain_retention": all(
            c[a]["finetune"][d]["adjusted"]["ci95_pp"][0] >= -1
            for a in ("current", "chronological", "random", "warm")
            for d in ("A", "B")
        ),
    }
    if gates != summary["gates"] or all(gates.values()) != summary["structured_filter_qualified"]:
        raise ValueError("Scientific decision differs")
    audit = read(out / "verification.json")
    if (
        audit["planned_optimizer_updates"] != 2 * len(rows) * 192 + 16
        or audit["new_primary_fits"] != sum(not v["aliased"] for r in rows for v in r["readouts"].values()) * 3
    ):
        raise ValueError("Work/readout accounting differs")
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "trajectories": 72,
            "exposure_decisions_reconstructed": exposure_rows,
            "health_scalars_reconstructed": metrics,
            "intervals_reconstructed": checks,
            "raw_prediction_readouts_checked": 147,
            "additional_gpu_updates": 0,
            "limits": "Raw classifier predictions are independently rescored; copied-encoder fine-tuning is not independently repeated. Exact learner replay is the collector audit.",
        },
    )
    print(json.dumps({"passed": True, "exposure_decisions": exposure_rows, "health_values": metrics}))


if __name__ == "__main__":
    main()
