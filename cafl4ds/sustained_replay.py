"""Sustained historical dose and age selection, independent of learner state."""

from __future__ import annotations

from typing import Any

import numpy as np

from cafl4ds.adaptive_mixture import SEEDS, interval, stream_seed

ARMS = ("current", "uniform25", "uniform50", "recent25", "recent50")
PRIMARY = (("uniform25", "current"), ("uniform50", "uniform25"), ("recent25", "uniform25"))
METRICS = ("variance", "covariance", "rank", "cka", "cosine")


def memories(stream: str, seed: int) -> list[list[int]]:
    """Snapshot the common all-arrival reservoir before each buffered block."""
    rng = np.random.default_rng(stream_seed(stream, seed) + 74111)
    memory = list(range(512))
    seen, snapshots = 512, []
    for start in range(512, 12288, 512):
        for index in range(seen, start):
            slot = int(rng.integers(index + 1))
            if slot < 512:
                memory[slot] = index
        seen = start
        snapshots.append(memory.copy())
    return snapshots


def batches(stream: str, seed: int, arm: str) -> list[dict[str, Any]]:
    """Plan paired old subsets; selections never consult pixels or labels."""
    if arm not in ARMS:
        raise ValueError("Unknown sustained replay arm")
    result = [
        {"step": j, "available": (j + 1) * 64, "old": [], "current": list(range(j * 64, j * 64 + 32))} for j in range(8)
    ]
    amount = 0 if arm == "current" else 8 if arm.endswith("25") else 16
    for block, memory in enumerate(memories(stream, seed)):
        start = 512 + block * 512
        candidates = sorted(memory)[-128:] if arm.startswith("recent") else memory
        rng = np.random.default_rng(stream_seed(stream, seed) + 94103 + block)
        old = rng.permutation(candidates)[:128].reshape(8, 16)
        for position in range(8):
            current = (start + position * 64 + np.linspace(0, 31, 32 - amount, dtype=int)).tolist()
            result.append(
                {
                    "step": 8 + block * 8 + position,
                    "available": start + 512,
                    "old": old[position, :amount].tolist(),
                    "current": current,
                }
            )
    return result


def manipulation(row: dict[str, Any]) -> dict[str, float | int]:
    """Describe exposures by arrival age, without assigning semantic utility."""
    start = (row["step"] // 8) * 512
    ages = start - np.asarray(row["old"], dtype=int)
    return {
        "old_count": len(row["old"]),
        "current_count": len(row["current"]),
        "mean_old_age": float(ages.mean()) if len(ages) else 0.0,
        "mean_current_age_at_release": float(np.mean(row["available"] - 1 - np.array(row["current"]))),
    }


def exposure_summary(plan: list[dict]) -> dict:
    """Keep dose, age, repetition and prepared-clip shares separately visible."""
    old = [i for r in plan for i in r["old"]]
    current = [i for r in plan for i in r["current"]]
    ages = [(r["step"] // 8) * 512 - i for r in plan for i in r["old"]]
    return {
        "old_exposures": len(old),
        "current_exposures": len(current),
        "unique_old": len(set(old)),
        "repeated_old_exposures": len(old) - len(set(old)),
        "unique_current": len(set(current)),
        "unique_total": len(set(old + current)),
        "old_age_quantiles": np.quantile(ages, [0, 0.25, 0.5, 0.75, 1]).tolist() if ages else [],
        "mean_old_age": float(np.mean(ages)) if ages else 0.0,
        "old_exposures_by_clip": [sum(i // 512 == clip for i in old) for clip in range(24)],
        "all_exposures_by_clip": [sum(i // 512 == clip for i in old + current) for clip in range(24)],
        "available_past_coverage": len(set(old)) / 11776,
        "current_arrival_coverage": len(set(current)) / 12288,
    }


def summarize(rows: list[dict], warm: dict) -> dict:
    """Primary three-contrast inference plus simultaneous eighteen-scalar gates."""
    groups = sorted({r["stream"] for r in rows})
    lookup = {(r["stream"], r["seed"], r["arm"]): r for r in rows}
    horizons = {}
    gates = {}
    for horizon in ("before_return", "terminal"):

        def quality(g, seed, arm, reader, domain):
            return (
                warm[str(seed)][reader][domain]
                if arm == "warm"
                else lookup[g, seed, arm]["readouts"][horizon]["scores"][reader][domain]
            )

        means = {
            a: {
                r: {
                    d: float(np.mean([quality(g, s, a, r, d) for g in groups for s in SEEDS]))
                    for d in ("balanced", "A", "B")
                }
                for r in ("finetune", "linear", "knn")
            }
            for a in (*ARMS, "warm")
        }
        contrasts = {}
        pairs = (
            list(PRIMARY)
            + [("recent50", "uniform50")]
            + [(a, c) for a in ARMS[1:] for c in ("current", "warm", "uniform25") if a != c]
        )
        for a, c in dict.fromkeys(pairs):
            key = f"{a}_minus_{c}"
            contrasts[key] = {}
            for reader in ("finetune", "linear", "knn"):
                contrasts[key][reader] = {}
                for domain in ("balanced", "A", "B"):
                    values = [
                        float(
                            np.mean(
                                [quality(g, s, a, reader, domain) - quality(g, s, c, reader, domain) for s in SEEDS]
                            )
                        )
                        for g in groups
                    ]
                    contrasts[key][reader][domain] = {
                        "ordinary": interval(values, 1),
                        "primary_family3": interval(values, 3),
                        "qualification_family18": interval(values, 18),
                    }
        # Paired difference-in-differences; a secondary, never a replacement primary.
        interaction = [
            float(
                np.mean(
                    [
                        quality(g, s, "recent50", "finetune", "balanced")
                        - quality(g, s, "uniform50", "finetune", "balanced")
                        - quality(g, s, "recent25", "finetune", "balanced")
                        + quality(g, s, "uniform25", "finetune", "balanced")
                        for s in SEEDS
                    ]
                )
            )
            for g in groups
        ]
        horizons[horizon] = {"means": means, "contrasts": contrasts, "secondary_interaction": interval(interaction, 1)}
        if horizon == "terminal":
            for a, c in PRIMARY:

                def stat(arm, comparator, domain):
                    return contrasts[f"{arm}_minus_{comparator}"]["finetune"][domain]["qualification_family18"]

                primary = stat(a, c, "balanced")
                useful = stat(a, "warm", "balanced")
                seed_values = [
                    float(
                        np.mean(
                            [
                                quality(g, s, a, "finetune", "balanced") - quality(g, s, c, "finetune", "balanced")
                                for g in groups
                            ]
                        )
                    )
                    for s in SEEDS
                ]
                panel_values = []
                for draw in (51011, 51017, 51023):
                    values = []
                    for g in groups:
                        for s in SEEDS:
                            left = lookup[g, s, a]["readouts"][horizon]["panel_balanced_pp"][str(draw)]
                            right = lookup[g, s, c]["readouts"][horizon]["panel_balanced_pp"][str(draw)]
                            values.append(left - right)
                    panel_values.append(float(np.mean(values)))
                gate = {
                    "primary_family3_signal": contrasts[f"{a}_minus_{c}"]["finetune"]["balanced"]["primary_family3"][
                        "ci95_pp"
                    ][0]
                    > 0,
                    "practical_primary": primary["mean_pp"] >= 0.25 and primary["ci95_pp"][0] > 0,
                    "source_seed_support_stability": min(primary["source_values_pp"]) >= 0
                    and min(seed_values) > 0
                    and min(panel_values) > 0,
                    "useful_ssl": useful["mean_pp"] >= 0.1 and useful["ci95_pp"][0] > 0,
                    "domain_retention": all(
                        stat(a, comparator, d)["ci95_pp"][0] >= -1
                        for comparator in ("current", "warm")
                        for d in ("A", "B")
                    ),
                    "seed_gains_pp": seed_values,
                    "panel_gains_pp": panel_values,
                }
                gate["development_lead"] = all(
                    gate[k]
                    for k in ("practical_primary", "source_seed_support_stability", "useful_ssl", "domain_retention")
                )
                gates[a] = gate
    return {
        "horizons": horizons,
        "gates": gates,
        "development_lead": any(g["development_lead"] for g in gates.values()),
        "fresh_confirmation": False,
        "adaptive_predictor_fitted": False,
    }
