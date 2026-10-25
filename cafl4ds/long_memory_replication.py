"""Fixed-anchor path summaries and source-level historical-memory decisions."""

from __future__ import annotations

import numpy as np


def reader_assignments(jobs, workers):
    """Balance expensive endpoint panels rather than equally counting checkpoints."""
    if workers < 1:
        raise ValueError("Require a positive worker count")
    assigned, loads = [[] for _ in range(workers)], [0] * workers
    for job in sorted(jobs, key=lambda j: -len(j["draws"])):
        worker = min(range(workers), key=lambda i: (loads[i], i))
        assigned[worker].append(job)
        loads[worker] += len(job["draws"])
    return assigned


def path_summary(curve, points):
    """Keep warm shortfall distinct from peak-relative drawdown on a fixed grid."""
    curve, points = np.asarray(curve, float), np.asarray(points, float)
    if curve.shape[-2] != len(points) or len(points) < 2 or not np.all(np.diff(points) > 0):
        raise ValueError("Require ordered checkpoints matching the trajectory axis")
    if not np.isfinite(curve).all():
        raise ValueError("Nonfinite capability")
    shortfall = np.maximum(curve[..., :1, :] - curve, 0)
    drawdown = np.maximum.accumulate(curve, axis=-2) - curve
    span = points[-1] - points[0]
    return {
        "endpoint": curve[..., -1, :],
        "warm_gain": curve[..., -1, :] - curve[..., 0, :],
        "max_warm_shortfall": shortfall.max(axis=-2),
        "mean_warm_shortfall": np.trapezoid(shortfall, points, axis=-2) / span,
        "max_drawdown": drawdown.max(axis=-2),
        "mean_drawdown": np.trapezoid(drawdown, points, axis=-2) / span,
    }


def endpoint_screen(current, fifo, reservoir, warm, margin=1.0, domain_budget=1.0):
    """Average panels before SSL draws before equally weighted source blocks."""
    current, fifo, reservoir = [np.asarray(x, float) for x in (current, fifo, reservoir)]
    if current.shape != fifo.shape or current.shape != reservoir.shape or current.shape[:2] != (3, 3):
        raise ValueError("Require matched 3-source x 3-training-draw x panels x components arrays")
    if current.ndim != 4 or current.shape[-1] != 3 or not all(np.isfinite(x).all() for x in (current, fifo, reservoir)):
        raise ValueError("Require finite balanced/day/night scores")
    run = (reservoir - current).mean(2)
    blocks = run.mean(1)
    fifo_blocks = (reservoir - fifo).mean(axis=(1, 2))
    warm_gain = float(reservoir[..., 0].mean() - np.asarray(warm)[..., 0].mean())
    gates = {
        "balanced_gain_at_least_one_pp": bool(blocks[:, 0].mean() >= margin),
        "positive_each_source": bool((blocks[:, 0] > 0).all()),
        "two_positive_training_draws_per_source": bool(((run[..., 0] > 0).sum(1) >= 2).all()),
        "positive_vs_fifo_each_source": bool((fifo_blocks[:, 0] > 0).all()),
        "each_source_domain_budget": bool((blocks[:, 1:] >= -domain_budget).all()),
        "positive_mean_gain_over_warm": warm_gain > 0,
    }
    return {
        "point_screen_passed": all(gates.values()),
        "gates": gates,
        "overall_vs_current": blocks.mean(0),
        "per_source_vs_current": blocks,
        "per_training_draw_vs_current": run,
        "per_source_vs_fifo": fifo_blocks,
        "overall_vs_fifo": fifo_blocks.mean(0),
        "balanced_warm_gain": warm_gain,
        "worst_individual_domain_effect": float((reservoir - current)[..., 1:].min()),
        "controller_qualified": False,
    }


def conditional_interval(difference, labels, domains, panel_counts, query_counts):
    """Cross identical panel/query resamples; never resample sources as images."""
    difference = np.asarray(difference, float)
    labels, domains = np.asarray(labels), np.asarray(domains)
    if difference.ndim != 2 or difference.shape[1] != len(labels):
        raise ValueError("Require paired panel x query correctness differences")
    if panel_counts.shape[1] != len(difference) or query_counts.shape[1] != len(labels):
        raise ValueError("Resampling population mismatch")
    weighted = (panel_counts / len(difference)) @ difference
    samples = []
    for domain in (0, 1):
        values = []
        for label in (0, 1):
            selected = (domains == domain) & (labels == label)
            values.append((weighted[:, selected] * query_counts[:, selected]).sum(1) / selected.sum())
        samples.append((values[0] + values[1]) / 2)
    samples = 100 * np.stack([(samples[0] + samples[1]) / 2, *samples], axis=-1)
    lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "lower": lower,
        "upper": upper,
        "half_width": (upper - lower) / 2,
        "scope": "paired support/fitting/query sensitivity conditional on reused evaluation and sampled sources",
    }
