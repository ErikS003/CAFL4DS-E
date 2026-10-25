"""Crossed fixed parent states and incoming clip blocks; paired source-block inference."""

import statistics

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json

PARENTS = {"early": 4224, "late": 8448}
CASES = {
    f"{p}_{c}_b{k}": {"parent": p, "context": c, "block": k}
    for p in PARENTS
    for c in ("day", "night")
    for k in range(5)
}
STATES = {n: PARENTS[v["parent"]] for n, v in CASES.items()}
CHECKPOINTS = (1024,)
ACTIONS = ("hold", "full")
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
METRICS = ("balanced", "A", "B")
READERS = ("finetune", "linear", "knn")


def paired_interval(values, adjusted=False):
    if len(values) != 5:
        raise ValueError("Five source-block means required")
    mean = statistics.fmean(values)
    half = float(t.ppf(1 - (0.05 / 18 if adjusted else 0.05) / 2, 4)) * statistics.stdev(values) / 5**0.5
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


def report_cross(out, data):
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
                        * sum(counts[f"{parent}_{context}_b{k}", s, a, r, m] for k in range(5) for s in SEEDS)
                        / (15 * (1200 if m == "balanced" else 600))
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
                            [block_delta(parent, context, k, r, m) for k in range(5)], level == "simultaneous"
                        )
                        for m in METRICS
                    }
                    for level in ("unadjusted", "simultaneous")
                }
                for r in READERS
            }
            cells[key] = {
                "means": means,
                "contrasts": contrasts,
                "preferred": {
                    level: preference(contrasts["finetune"][level]) for level in ("unadjusted", "simultaneous")
                },
            }
    interactions = {
        context: {
            r: {
                level: {
                    m: paired_interval(
                        [
                            block_delta("late", context, k, r, m) - block_delta("early", context, k, r, m)
                            for k in range(5)
                        ],
                        level == "simultaneous",
                    )
                    for m in METRICS
                }
                for level in ("unadjusted", "simultaneous")
            }
            for r in READERS
        }
        for context in ("day", "night")
    }
    flips = {
        c: cells[f"early_{c}"]["preferred"]["simultaneous"]["full"]
        and cells[f"late_{c}"]["preferred"]["simultaneous"]["hold"]
        or cells[f"early_{c}"]["preferred"]["simultaneous"]["hold"]
        and cells[f"late_{c}"]["preferred"]["simultaneous"]["full"]
        for c in ("day", "night")
    }
    context_flips = {
        p: (
            cells[f"{p}_day"]["preferred"]["simultaneous"]["full"]
            and cells[f"{p}_night"]["preferred"]["simultaneous"]["hold"]
        )
        or (
            cells[f"{p}_day"]["preferred"]["simultaneous"]["hold"]
            and cells[f"{p}_night"]["preferred"]["simultaneous"]["full"]
        )
        for p in PARENTS
    }
    result = {
        "cells": cells,
        "late_minus_early_action_response": interactions,
        "same_footage_state_preference_flip": flips,
        "incoming_context_preference_flip": context_flips,
        "decision": "state_preference_flip_qualified" if any(flips.values()) else "no_state_preference_flip_qualified",
        "scientific_ssl_updates": 1920,
        "primary_fits": 198,
        "new_prediction_sets": 594,
        "source_blocks_per_context": 5,
        "uncertainty": "paired t df4 over five disjoint clip-block means, optimization seeds averaged within block; capture-session independence unknown",
        "simultaneous_primary_scalar_contrasts": 18,
        "controller_fit": False,
    }
    atomic_json(out / "summary.json", result)
    return result
