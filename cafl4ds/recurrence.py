"""Controlled recurrence diets and fixed-pool evaluation; no labels enter SSL updates."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import torch

from cafl4ds.data.attributes import AttributedImages


def restrict_scenes(data: AttributedImages, names: list[str]) -> tuple[AttributedImages, torch.Tensor]:
    """Select the declared shared scene classes and preserve source-index provenance."""
    lookup = {name: label for label, name in data.canary_names.items()}
    if any(name not in lookup for name in names) or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate declared scene class")
    selected = torch.isin(data.canary, torch.tensor([lookup[name] for name in names])).nonzero().flatten()
    remapped = torch.empty(len(selected), dtype=torch.long)
    for label, name in enumerate(names):
        remapped[data.canary[selected] == lookup[name]] = label
    return AttributedImages(
        data.images[selected],
        data.era_key[selected],
        remapped,
        data.regime_order,
        data.regime_names,
        dict(enumerate(names)),
    ), selected


def domain_ids(data: AttributedImages, names: list[str]) -> list[int]:
    """Resolve domains by stable names, not split-dependent numeric IDs."""
    lookup = {name: key for key, name in data.regime_names.items()}
    if len(names) != 2 or len(set(names)) != 2 or any(name not in lookup for name in names):
        raise ValueError("Exactly two distinct, present domains required")
    return [lookup[name] for name in names]


def visit_quotas(data: AttributedImages, names: list[str], maximum: int, minimum: int) -> dict[int, int]:
    """Match per-scene arrivals across both domains and both visits using counts only."""
    domains = domain_ids(data, names)
    quotas = {}
    for label in sorted(data.canary_names):
        counts = [int(((data.era_key == d) & (data.canary == label)).sum()) for d in domains]
        quota = min(maximum // 32 * 32, min(counts) // 64 * 32)
        if quota < minimum:
            raise ValueError(f"Class {label} has domain counts {counts}; visit quota {quota} < {minimum}")
        quotas[label] = quota
    return quotas


def arrival_orders(
    data: AttributedImages, names: list[str], quotas: dict[int, int], seed: int
) -> dict[str, torch.Tensor]:
    """Pair policies to A1-B1-A2-B2 and a global permutation of the identical image set."""
    generator = torch.Generator().manual_seed(seed)
    blocks: list[list[torch.Tensor]] = [[], [], [], []]
    for domain, actual in enumerate(domain_ids(data, names)):
        for label, quota in sorted(quotas.items()):
            indices = ((data.era_key == actual) & (data.canary == label)).nonzero().flatten()
            if len(indices) < 2 * quota:
                raise ValueError("Insufficient unique arrivals")
            chosen = indices[torch.randperm(len(indices), generator=generator)[: 2 * quota]]
            blocks[domain].append(chosen[:quota])
            blocks[domain + 2].append(chosen[quota:])
    visits = [torch.cat(parts) for parts in blocks]
    visits = [v[torch.randperm(len(v), generator=generator)] for v in visits]
    recurring = torch.cat(visits)
    if len(set(recurring.tolist())) != len(recurring):
        raise ValueError("Training image repeated across visits")
    return {"recurrence": recurring, "shuffled": recurring[torch.randperm(len(recurring), generator=generator)]}


@dataclass
class Evaluation:
    """Only fixed validation images, support draws, and query indices."""

    images: torch.Tensor
    labels: torch.Tensor
    domains: torch.Tensor
    query: torch.Tensor
    supports: dict[int, torch.Tensor]
    manifest: dict[str, Any]


def evaluation_pool(data: AttributedImages, names: list[str], settings: dict[str, Any]) -> Evaluation:
    """Reserve disjoint support/query cells and share them across time and methods."""
    generator = torch.Generator().manual_seed(int(settings["pool_seed"]))
    selected: list[int] = []
    domain_tags: list[int] = []
    query_positions: list[int] = []
    pools: dict[tuple[int, int], torch.Tensor] = {}
    offset = 0
    counts: dict[str, Any] = {}
    shots = int(settings["shots_per_class"])
    for domain, actual in enumerate(domain_ids(data, names)):
        for label in sorted(data.canary_names):
            candidates = ((data.era_key == actual) & (data.canary == label)).nonzero().flatten()
            candidates = candidates[torch.randperm(len(candidates), generator=generator)]
            support_count = int(settings["support_per_domain_class"])
            query_count = min(int(settings["query_per_domain_class"]), len(candidates) - support_count)
            if query_count < int(settings["min_query_per_domain_class"]) or support_count < (shots + 1) // 2:
                raise ValueError(f"Inadequate validation cell {names[domain]}/{data.canary_names[label]}")
            chosen = candidates[: support_count + query_count]
            selected.extend(chosen.tolist())
            domain_tags.extend([domain] * len(chosen))
            pools[domain, label] = torch.arange(offset, offset + support_count)
            query_positions.extend(range(offset + support_count, offset + len(chosen)))
            counts[f"{domain}_{label}"] = {"support_pool": support_count, "query": query_count}
            offset += len(chosen)
    supports = {}
    for position, draw in enumerate(settings["draw_seeds"]):
        generator = torch.Generator().manual_seed(int(draw))
        indices = []
        for label in sorted(data.canary_names):
            for domain in range(2):
                take = shots // 2 + int(shots % 2 == 1 and domain == (position + 1) % 2)
                candidates = pools[domain, label]
                indices.extend(candidates[torch.randperm(len(candidates), generator=generator)[:take]].tolist())
        supports[int(draw)] = torch.tensor(indices)
    original = torch.tensor(selected)
    manifest = {
        "original_indices": selected,
        "query_positions": query_positions,
        "cell_counts": counts,
        "supports": {str(k): v.tolist() for k, v in supports.items()},
        "domain_names": names,
        "class_names": data.canary_names,
    }
    return Evaluation(
        data.images[original],
        data.canary[original],
        torch.tensor(domain_tags),
        torch.tensor(query_positions),
        supports,
        manifest,
    )


def prediction_scores(predicted: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> dict[str, float]:
    """Use class-balanced domain accuracy and equal weighting of domains."""
    correct = (predicted == labels).float()
    result = {
        name: statistics.fmean(
            float(correct[(domains == d) & (labels == c)].mean()) for c in sorted(set(labels.tolist()))
        )
        for d, name in enumerate(("A", "B"))
    }
    result["balanced"] = (result["A"] + result["B"]) / 2
    result["micro"] = float(correct.mean())
    return result


def observation_grid(steps_per_visit: int) -> list[int]:
    """Observe quarter visits and exact endpoints on the optimizer-update clock."""
    if steps_per_visit < 4:
        raise ValueError("At least four updates per visit required")
    return sorted({v * steps_per_visit + round(q * steps_per_visit / 4) for v in range(4) for q in range(5)})


def retention_readout(trace: list[dict[str, Any]], steps_per_visit: int, readout: str) -> dict[str, float]:
    """Separate loss during absence, return gains, BWT and signed final forgetting."""
    endpoints = {
        int(r["step"]) // steps_per_visit: r["scores"][readout] for r in trace if int(r["step"]) % steps_per_visit == 0
    }
    if set(endpoints) != set(range(5)):
        raise ValueError("Incomplete boundary trajectory")
    r = endpoints
    return {
        "interference": ((r[1]["A"] - r[2]["A"]) + (r[2]["B"] - r[3]["B"])) / 2,
        "absence_A": r[1]["A"] - r[2]["A"],
        "absence_B": r[2]["B"] - r[3]["B"],
        "bwt": ((r[4]["A"] - r[1]["A"]) + (r[4]["B"] - r[2]["B"])) / 2,
        "forgetting": ((max(r[t]["A"] for t in (1, 2, 3)) - r[4]["A"]) + (max(r[t]["B"] for t in (2, 3)) - r[4]["B"]))
        / 2,
        "return_gain_A": r[3]["A"] - r[2]["A"],
        "return_gain_B": r[4]["B"] - r[3]["B"],
        "return_vs_first_A": r[3]["A"] - r[1]["A"],
        "return_vs_first_B": r[4]["B"] - r[2]["B"],
        "first_gain_A": r[1]["A"] - r[0]["A"],
        "first_gain_B": r[2]["B"] - r[0]["B"],
        "final_balanced": r[4]["balanced"],
        "final_vs_warm": r[4]["balanced"] - r[0]["balanced"],
    }


def recovery_latency(trace: list[dict[str, Any]], steps_per_visit: int, tolerance: float) -> dict[str, Any]:
    """Report sustained observed recovery, including zero and explicit right censoring."""
    by_step = {r["step"]: r["scores"]["linear"] for r in trace}
    result = {}
    for domain, first, start in (("A", 1, 2), ("B", 2, 3)):
        threshold = by_step[first * steps_per_visit][domain] - tolerance
        points = [
            (step - start * steps_per_visit, scores[domain])
            for step, scores in sorted(by_step.items())
            if start * steps_per_visit <= step <= (start + 1) * steps_per_visit
        ]
        latency = next(
            (
                points[i][0]
                for i in range(len(points) - 1)
                if points[i][1] >= threshold and points[i + 1][1] >= threshold
            ),
            None,
        )
        result[domain] = {
            "updates": latency,
            "censored": latency is None,
            "threshold": threshold,
            "observed_horizon_updates": steps_per_visit,
        }
    return result
