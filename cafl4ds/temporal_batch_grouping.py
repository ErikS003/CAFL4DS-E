"""Causal temporal regrouping with identical replay-image doses."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.adaptive_mixture import SEEDS, interval, memory_plan, stream_seed

ARMS = ("current", "chronological", "random", "stratified")
METRICS = ("variance", "covariance", "rank", "cka", "cosine")


def batches(stream: str, seed: int, arm: str) -> list[dict[str, Any]]:
    """Choose from each arrived block; never consult pixels or task/health labels."""
    if arm not in ARMS:
        raise ValueError("Unknown temporal grouping arm")
    plans = memory_plan(stream, seed)
    result = []
    for step in range(8):
        current = list(range(step * 64, step * 64 + 32))
        result.append({"step": step, "available": (step + 1) * 64, "old": [], "current": current})
    for block in range(23):
        start = 512 + block * 512
        selected = np.array([start + j * 64 + np.linspace(0, 31, 24, dtype=int) for j in range(8)])
        if arm == "stratified":
            selected = selected.reshape(8, 8, 3).transpose(1, 0, 2).reshape(8, 24)
        elif arm == "random":
            generator = np.random.default_rng(stream_seed(stream, seed) + 88169 + block)
            selected = generator.permutation(selected.reshape(-1)).reshape(8, 24)
        elif arm == "current":
            selected = np.array([np.arange(start + j * 64, start + j * 64 + 32) for j in range(8)])
        for position in range(8):
            old = [] if arm == "current" else plans[block][position * 8 : position * 8 + 8]
            current = selected[position].tolist()
            if any(i >= start for i in old) or len(set(old + current)) != 32:
                raise ValueError("Invalid historical/current exposure identity")
            result.append({"step": 8 + block * 8 + position, "available": start + 512, "old": old, "current": current})
    return result


def manipulation(row: dict[str, Any]) -> dict[str, float]:
    """Describe temporal co-occurrence, without treating it as semantic quality."""
    ids = np.array(row["current"])
    strata = ids // 64
    i, j = np.triu_indices(len(ids), 1)
    counts = np.unique(strata, return_counts=True)[1]
    return {
        "occupied_strata": int(len(counts)),
        "largest_stratum_count": int(counts.max()),
        "same_stratum_pair_fraction": float(np.mean(strata[i] == strata[j])),
        "mean_pair_arrival_separation": float(np.mean(np.abs(ids[i] - ids[j]))),
        "mean_age_at_release": float(np.mean(row["available"] - 1 - ids)),
    }


def summarize(rows: list[dict], warm: dict[str, Any]) -> dict[str, Any]:
    """Keep six source groups as uncertainty units, averaging seeds within source."""
    groups = sorted({r["stream"] for r in rows})
    lookup = {(r["stream"], r["seed"], r["arm"]): r for r in rows}
    result = {}
    for horizon in ("before_return", "terminal"):

        def quality(group: str, arm: str, reader: str, domain: str, seed: int) -> float:
            return (
                warm[str(seed)][reader][domain]
                if arm == "warm"
                else lookup[group, seed, arm]["readouts"][horizon]["scores"][reader][domain]
            )

        means, contrasts = {}, {}
        for arm in (*ARMS, "warm"):
            means[arm] = {
                reader: {
                    d: float(np.mean([quality(g, arm, reader, d, s) for g in groups for s in SEEDS]))
                    for d in ("balanced", "A", "B")
                }
                for reader in ("finetune", "linear", "knn")
            }
        for comparator in ("current", "chronological", "random", "warm"):
            contrasts[comparator] = {}
            for reader in ("finetune", "linear", "knn"):
                contrasts[comparator][reader] = {}
                for d in ("balanced", "A", "B"):
                    values = [
                        float(
                            np.mean(
                                [
                                    quality(g, "stratified", reader, d, s) - quality(g, comparator, reader, d, s)
                                    for s in SEEDS
                                ]
                            )
                        )
                        for g in groups
                    ]
                    contrasts[comparator][reader][d] = {
                        "ordinary": interval(values, 1),
                        "adjusted": interval(values, 12),
                    }
        random_delta = [
            float(
                np.mean(
                    [
                        quality(g, "random", "finetune", "balanced", s)
                        - quality(g, "chronological", "finetune", "balanced", s)
                        for s in SEEDS
                    ]
                )
            )
            for g in groups
        ]
        result[horizon] = {
            "means": means,
            "contrasts": contrasts,
            "secondary_random_minus_chronological": interval(random_delta, 1),
        }
    c = result["terminal"]["contrasts"]
    gates = {
        "structured_content_value": all(
            c[a]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.25
            and c[a]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0
            for a in ("current", "chronological", "random")
        ),
        "useful_ssl": c["warm"]["finetune"]["balanced"]["adjusted"]["mean_pp"] >= 0.1
        and c["warm"]["finetune"]["balanced"]["adjusted"]["ci95_pp"][0] > 0,
        "terminal_domain_retention": all(
            c[a]["finetune"][d]["adjusted"]["ci95_pp"][0] >= -1
            for a in ("current", "chronological", "random", "warm")
            for d in ("A", "B")
        ),
    }
    return {
        "horizons": result,
        "gates": gates,
        "structured_filter_qualified": all(gates.values()),
        "fresh_confirmation": False,
        "adaptive_predictor_fitted": False,
    }
