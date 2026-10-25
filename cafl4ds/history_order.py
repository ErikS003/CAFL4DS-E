"""Crossed fixed parent states and incoming clip blocks; paired source-block inference."""

import statistics

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json

PARENTS = {"DN": 8448, "ND": 8448}
BLOCKS = {"day": 12, "night": 5}
CASES = {
    f"{p}_{c}_b{k}": {"parent": p, "context": c, "block": k}
    for p in PARENTS
    for c in ("day", "night")
    for k in range(BLOCKS[c])
}
STATES = {n: PARENTS[v["parent"]] for n, v in CASES.items()}
CHECKPOINTS = (1024,)
ACTIONS = ("hold", "full")
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
METRICS = ("balanced", "A", "B")
READERS = ("finetune", "linear", "knn")


def history_indices(order):
    if order == "DN":
        return list(range(8448))
    if order == "ND":
        return list(range(4224, 8448)) + list(range(4224))
    raise ValueError("Unknown training order")


def paired_interval(values, adjusted=False):
    if len(values) not in (5, 12):
        raise ValueError("Expected five secondary or twelve primary block means")
    mean = statistics.fmean(values)
    half = (
        float(t.ppf(1 - (0.05 / 6 if adjusted else 0.05) / 2, len(values) - 1))
        * statistics.stdev(values)
        / len(values) ** 0.5
    )
    return {"mean_pp": mean, "ci95_pp": [mean - half, mean + half], "source_block_values_pp": values}


def preference(contrasts):
    return {
        "full": contrasts["balanced"]["mean_pp"] >= 0.1
        and contrasts["balanced"]["ci95_pp"][0] > 0
        and all(contrasts[m]["ci95_pp"][0] > -1 for m in ("A", "B")),
        "hold": contrasts["balanced"]["mean_pp"] <= -0.1
        and contrasts["balanced"]["ci95_pp"][1] < 0
        and all(contrasts[m]["ci95_pp"][1] < 1 for m in ("A", "B")),
    }


def report_history(out, data):
    labels = data.labels[data.query].numpy()
    domains = data.domains[data.query].numpy()
    counts = {}
    for name, case in CASES.items():
        for seed in SEEDS:
            for action in ACTIONS:
                directory = (
                    out / "parents" / case["parent"] / f"seed_{seed}"
                    if action == "hold"
                    else out / "runs" / name / "full" / f"seed_{seed}/at_1024"
                )
                with np.load(directory / "predictions.npz") as predictions:
                    for reader in READERS:
                        for m in METRICS:
                            mask = np.ones(len(labels), bool) if m == "balanced" else domains == (0 if m == "A" else 1)
                            counts[name, seed, action, reader, m] = sum(
                                int(((predictions[f"{d}_{reader}"] == labels) & mask).sum()) for d in DRAWS
                            )

    def block_delta(parent, context, k, reader, m):
        name = f"{parent}_{context}_b{k}"
        return (
            100
            * sum(counts[name, s, "full", reader, m] - counts[name, s, "hold", reader, m] for s in SEEDS)
            / (3 * (1200 if m == "balanced" else 600))
        )

    cells = {}
    for parent in PARENTS:
        for context in ("day", "night"):
            key = f"{parent}_{context}"
            means = {
                r: {
                    a: {
                        m: 100
                        * sum(
                            counts[f"{parent}_{context}_b{k}", s, a, r, m]
                            for k in range(BLOCKS[context])
                            for s in SEEDS
                        )
                        / (3 * BLOCKS[context] * (1200 if m == "balanced" else 600))
                        for m in METRICS
                    }
                    for a in ACTIONS
                }
                for r in READERS
            }
            contrasts = {
                r: {
                    level: {
                        m: paired_interval(
                            [block_delta(parent, context, k, r, m) for k in range(BLOCKS[context])],
                            level == "guard_adjusted",
                        )
                        for m in METRICS
                    }
                    for level in ("unadjusted", "guard_adjusted")
                }
                for r in READERS
            }
            cells[key] = {
                "means": means,
                "contrasts": contrasts,
                "preferred": {
                    level: preference(contrasts["finetune"][level]) for level in ("unadjusted", "guard_adjusted")
                },
            }
    interactions = {
        context: {
            r: {
                level: {
                    m: paired_interval(
                        [
                            block_delta("ND", context, k, r, m) - block_delta("DN", context, k, r, m)
                            for k in range(BLOCKS[context])
                        ],
                        level == "guard_adjusted",
                    )
                    for m in METRICS
                }
                for level in ("unadjusted", "guard_adjusted")
            }
            for r in READERS
        }
        for context in ("day", "night")
    }
    flips = {
        c: cells[f"DN_{c}"]["preferred"]["guard_adjusted"]["full"]
        and cells[f"ND_{c}"]["preferred"]["guard_adjusted"]["hold"]
        or cells[f"DN_{c}"]["preferred"]["guard_adjusted"]["hold"]
        and cells[f"ND_{c}"]["preferred"]["guard_adjusted"]["full"]
        for c in ("day", "night")
    }
    primary = interactions["day"]["finetune"]["unadjusted"]["balanced"]
    primary_qualified = (primary["ci95_pp"][0] > 0 or primary["ci95_pp"][1] < 0) and abs(primary["mean_pp"]) >= 0.5
    flips = {"day": bool(primary_qualified and flips["day"]), "night": False}
    result = {
        "cells": cells,
        "ND_minus_DN_action_response": interactions,
        "same_footage_state_preference_flip": flips,
        "primary_history_effect_qualified": primary_qualified,
        "decision": "matched_age_preference_flip_qualified"
        if flips["day"]
        else "history_changes_action_response"
        if primary_qualified
        else "no_matched_age_history_effect_qualified",
        "scientific_ssl_updates": 4848,
        "primary_fits": 324,
        "new_prediction_sets": 972,
        "source_blocks_per_context": BLOCKS,
        "uncertainty": "paired t over twelve new primary day blocks (df11) and five reused secondary night blocks (df4), optimization seeds averaged within block; capture-session independence unknown",
        "primary_interaction_tests": 1,
        "joint_guard_family_scalar_contrasts": 6,
        "controller_fit": False,
    }
    atomic_json(out / "summary.json", result)
    return result
