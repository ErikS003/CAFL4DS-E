"""Conditional opportunity screen for fixed earlier-state adaptation algorithms."""

from __future__ import annotations

import numpy as np

from cafl4ds.domain_health_calibration import schedule
from cafl4ds.target_alignment_measurement import metric_means

ARMS = ("current", "balanced")
FIELDS = ("rank_drop", "variance_drop", "covariance_rise", "cka_drift", "cosine_drift")


def diet(pool, arm, steps):
    if arm not in ARMS:
        raise ValueError("Only the two prespecified algorithms are eligible")
    return schedule(pool, arm, steps)


def crossed_samples(difference, labels, domains, panels, queries):
    """Common support/query multiplicities; sources/futures stay fixed."""
    difference = np.asarray(difference, dtype=float)
    labels, domains = np.asarray(labels), np.asarray(domains)
    if difference.ndim != 4 or difference.shape[-1] != len(labels):
        raise ValueError("Expected source by future by support by query differences")
    if not np.isfinite(difference).all():
        raise ValueError("Nonfinite effects")
    draws = difference.shape[-2]
    if not np.all(panels.sum(1) == draws):
        raise ValueError("Invalid support multiplicities")
    mixed = panels.astype(float) @ difference.mean((0, 1)) / draws
    domain_results = []
    for d in (0, 1):
        result = np.zeros(len(panels))
        for c in (0, 1):
            selected = (domains == d) & (labels == c)
            if not selected.any() or not np.all(queries[:, selected].sum(1) == selected.sum()):
                raise ValueError("Invalid query cell multiplicities")
            result += np.einsum("rq,rq->r", mixed[:, selected], queries[:, selected]) / (2 * selected.sum())
        domain_results.append(result)
    return np.stack([(domain_results[0] + domain_results[1]) / 2, *domain_results], axis=-1)


def screen(difference, labels, domains, panels, queries):
    values = metric_means(difference, labels, domains)
    mean = values.mean((0, 1, 2))
    sources = values.mean((1, 2))
    futures = values.mean((0, 2))
    samples = crossed_samples(difference, labels, domains, panels, queries)
    bounds = np.quantile(samples, [0.0125, 0.9875], axis=0).T
    requirements = {
        "balanced_gain_at_least_half_pp": bool(mean[0] >= 0.5),
        "positive_three_of_four_futures": bool((futures[:, 0] > 0).sum() >= 3),
        "positive_two_of_three_sources": bool((sources[:, 0] > 0).sum() >= 2),
        "pooled_domains_above_minus_one": bool(np.all(mean[1:] >= -1)),
        "every_future_domain_above_minus_one": bool(np.all(futures[:, 1:] >= -1)),
        "conditional_balanced_lower_positive": bool(bounds[0, 0] > 0),
    }
    return {
        "mean_pp": mean,
        "source_means_pp": sources,
        "future_means_pp": futures,
        "case_means_pp": values.mean(2),
        "support_means_pp": values.mean((0, 1)),
        "family_two_conditional_bounds_pp": bounds,
        "requirements": requirements,
        "passes": all(requirements.values()),
    }
