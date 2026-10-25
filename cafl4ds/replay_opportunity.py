"""Cached action opportunity with disjoint query and support selection."""

from __future__ import annotations

import hashlib

import numpy as np

ARMS = ("freeze", "balanced", "history", "current")
READERS = ("finetune", "knn", "linear")
TOL = 1e-10


def query_halves(clips, labels, domains):
    """Hash clip identities within each class/domain; keep a clip in one half."""
    clips, labels, domains = np.asarray(clips), np.asarray(labels), np.asarray(domains)
    if not (len(clips) == len(labels) == len(domains)):
        raise ValueError("Unaligned query metadata")
    cells = {}
    for clip in np.unique(clips):
        mask = clips == clip
        cell = set(zip(labels[mask].tolist(), domains[mask].tolist(), strict=True))
        if len(cell) != 1:
            raise ValueError("A query clip crosses strata")
        cells[str(clip)] = next(iter(cell))
    a = np.zeros(len(clips), bool)
    for d in (0, 1):
        for l in (0, 1):
            ids = [c for c, cell in cells.items() if cell == (l, d)]
            ids.sort(key=lambda c: hashlib.sha256(f"replay-opportunity:20261019:{c}".encode()).digest())
            if len(ids) < 2:
                raise ValueError("Need two query clips in each stratum")
            a |= np.isin(clips, ids[: len(ids) // 2])
    return a, ~a


def metrics(correct, labels, domains, mask):
    """Equal class weights within domains, then equal day/night weights."""
    values = []
    for d in (0, 1):
        means = []
        for l in (0, 1):
            selected = mask & (domains == d) & (labels == l)
            if not selected.any():
                raise ValueError("Empty query cell")
            means.append(np.asarray(correct)[..., selected].mean(-1) * 100)
        values.append((means[0] + means[1]) / 2)
    return np.stack([(values[0] + values[1]) / 2, *values], axis=-1)


def choices(scores, retention=True):
    """Pick using selection scores only; ties prefer freeze, then fixed arm order."""
    scores = np.asarray(scores, float)
    feasible = (
        np.all(scores[:, :, 1:] - scores[:, 0:1, 1:] >= -1 - TOL, axis=-1)
        if retention
        else np.ones(scores.shape[:2], bool)
    )
    objective = np.where(feasible, scores[:, :, 0], -np.inf)
    return np.argmax(objective >= objective.max(1, keepdims=True) - TOL, axis=1)


def fixed(scores):
    means = np.asarray(scores)[:, :, 0].mean(0)
    return int(np.flatnonzero(means >= means.max() - TOL)[0])


def transfer(selection, evaluation, retention=True):
    """Transfer frozen case-specific and pooled choices to separate observations."""
    selected, common = choices(selection, retention), fixed(selection)
    picked = evaluation[np.arange(len(selected)), selected]
    comparisons = {"freeze": picked - evaluation[:, 0], "fixed": picked - evaluation[:, common]}
    return {
        "choices": selected,
        "fixed_choice": common,
        "selected_scores": picked,
        "fixed_scores": evaluation[:, common],
        "effects": comparisons,
        "mean_effects": {k: v.mean(0) for k, v in comparisons.items()},
        "mean_SSL_updates": float(np.mean(selected != 0) * 1024),
        "domain_harm_cases_vs_freeze": np.flatnonzero(np.any(comparisons["freeze"][:, 1:] < -1 - TOL, axis=1)),
    }


def bootstrap_counts(a, labels, domains, draws, seed):
    """Paired panel/query resamples conditional on existing unique query clips."""
    rng = np.random.default_rng(seed)
    panels = rng.multinomial(12, np.full(12, 1 / 12), size=draws).astype(np.uint8)
    counts = np.zeros((draws, len(labels)), np.uint16)
    for mask in (a, ~a):
        for d in (0, 1):
            for l in (0, 1):
                ix = np.flatnonzero(mask & (domains == d) & (labels == l))
                counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=draws)
    return panels, counts


def bootstrap(correct, labels, domains, a, panel_counts, query_counts):
    """Recompute action/fixed selections inside every conditional resample."""
    # correct: case, support draw, action, query; SSL draws already equally averaged.
    correct = np.asarray(correct, float)
    draws = len(panel_counts)
    gains = np.empty((draws, 2, 2, 3))
    selected = np.empty((draws, 2, 6), np.uint8)
    for start in range(0, draws, 50):
        stop = min(start + 50, draws)
        weighted = np.einsum("br,crkq->bckq", panel_counts[start:stop] / 12, correct, optimize=True)
        partitions = []
        for mask in (a, ~a):
            cells = []
            for d in (0, 1):
                values = []
                for l in (0, 1):
                    ix = np.flatnonzero(mask & (domains == d) & (labels == l))
                    values.append(
                        np.einsum(
                            "bckq,bq->bck", weighted[..., ix], query_counts[start:stop][:, ix] / len(ix), optimize=True
                        )
                        * 100
                    )
                cells.append((values[0] + values[1]) / 2)
            partitions.append(np.stack([(cells[0] + cells[1]) / 2, *cells], axis=-1))
        for j in range(stop - start):
            for direction, (left, right) in enumerate(((0, 1), (1, 0))):
                result = transfer(partitions[left][j], partitions[right][j])
                selected[start + j, direction] = result["choices"]
                gains[start + j, direction] = np.stack([result["mean_effects"][k] for k in ("freeze", "fixed")])
    return gains, selected


def analysis(tables, bootstrap_gains, support_tables):
    """Screen a new mechanism experiment; never qualify a health controller."""
    # tables: reader, partition(all/A/B), case, action, metric.
    primary = np.asarray(tables)[0]
    full = {
        name: transfer(primary[0], primary[0], retention)
        for name, retention in (("unconstrained", False), ("retention_aware", True))
    }
    folds = {name: transfer(primary[l], primary[r]) for name, l, r in (("A_to_B", 1, 2), ("B_to_A", 2, 1))}
    crossed = {}
    for left, right, label in ((0, 1, "S0_to_S1"), (1, 0, "S1_to_S0")):
        for name, l, r in (("A_to_B", 1, 2), ("B_to_A", 2, 1)):
            crossed[f"{name}_{label}"] = transfer(support_tables[left, l], support_tables[right, r])
    # Four primary balanced comparisons: two query directions x two comparators.
    bounds = np.quantile(bootstrap_gains, [0.05 / (2 * 4), 1 - 0.05 / (2 * 4)], axis=0)
    gates = {}
    for direction, (name, result) in enumerate(folds.items()):
        for comp_index, comp in enumerate(("freeze", "fixed")):
            effects = result["effects"][comp]
            prefix = f"{name}_vs_{comp}"
            gates[prefix + "_at_least_half_pp"] = bool(effects[:, 0].mean() >= 0.5 - TOL)
            gates[prefix + "_conditional_gain_lower_positive"] = bool(bounds[0, direction, comp_index, 0] > 0)
            gates[prefix + "_domain_means_within_one_pp"] = bool(np.all(effects[:, 1:].mean(0) >= -1 - TOL))
            gates[prefix + "_four_positive_cases"] = bool((effects[:, 0] > TOL).sum() >= 4)
        gates[name + "_no_case_domain_loss_over_one_pp"] = len(result["domain_harm_cases_vs_freeze"]) == 0
    stable = sum(all(r["mean_effects"][comp][0] > TOL for comp in ("freeze", "fixed")) for r in crossed.values())
    gates["three_of_four_disjoint_support_checks_positive"] = bool(stable >= 3)
    return {
        "full_query_optimistic": full,
        "query_transfer": folds,
        "crossed_support_transfer": crossed,
        "conditional_gain_bounds": bounds,
        "gates": gates,
        "screen_passed": all(gates.values()),
        "decision": "prepare_fresh_mechanism_experiment"
        if all(gates.values())
        else "stop_this_cached_adaptive_opportunity",
        "controller_qualified": False,
        "population_confirmation": False,
        "primary_reader": "finetune",
        "known_future_contexts": 2,
        "historical_source_blocks": 3,
    }
