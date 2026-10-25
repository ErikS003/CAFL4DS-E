"""Paired uncertainty for a specified few-label downstream learning procedure."""

from __future__ import annotations

import numpy as np


def support_draws(labels, domains, pool, seeds):
    """Sample 25 labels per class using the inherited 12/13 domain allocation."""
    import torch

    result = {}
    labels, domains, pool = map(np.asarray, (labels, domains, pool))
    for position, seed in enumerate(seeds):
        generator = torch.Generator().manual_seed(int(seed))
        indices = []
        for label in (0, 1):
            for domain in (0, 1):
                cell = pool[(labels[pool] == label) & (domains[pool] == domain)]
                take = 12 + int(domain == (position + 1) % 2)
                if len(cell) < take:
                    raise ValueError("Insufficient support cell")
                indices.extend(cell[torch.randperm(len(cell), generator=generator)[:take].numpy()].tolist())
        result[int(seed)] = np.array(indices, dtype=np.int64)
    return result


def resampling_counts(labels, domains, panels, repetitions, seed):
    """Paired panel bootstrap and class-stratified query bootstrap, conditional on pools."""
    labels, domains = np.asarray(labels), np.asarray(domains)
    rng = np.random.default_rng(seed)
    panel = rng.multinomial(panels, np.full(panels, 1 / panels), size=repetitions).astype(np.uint16)
    query = np.zeros((repetitions, len(labels)), dtype=np.uint16)
    for domain in (0, 1):
        for label in (0, 1):
            indices = np.flatnonzero((domains == domain) & (labels == label))
            if not len(indices):
                raise ValueError("Empty query cell")
            query[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions)
    return panel, query


def metric_means(values, labels, domains):
    """Return balanced/day/night means, keeping all leading axes."""
    values, labels, domains = map(np.asarray, (values, labels, domains))
    by_domain = [
        np.mean([values[..., (domains == d) & (labels == c)].mean(axis=-1) for c in (0, 1)], axis=0) for d in (0, 1)
    ]
    return np.stack([(by_domain[0] + by_domain[1]) / 2, *by_domain], axis=-1)


class TargetAlignmentMeasurement:
    """Estimate paired effects and separate support/query/crossed sensitivity."""

    def __init__(self, support_half_width_pp=0.5, crossed_half_width_pp=1.0, required_fraction=0.8):
        self.support_half_width_pp = support_half_width_pp
        self.crossed_half_width_pp = crossed_half_width_pp
        self.required_fraction = required_fraction

    def compare(self, before, after, labels, domains, panel_counts, query_counts):
        """Use identical support/seed and query resamples on both encoder states."""
        before, after = np.asarray(before), np.asarray(after)
        labels, domains = np.asarray(labels), np.asarray(domains)
        if before.shape != after.shape or before.ndim != 3 or before.shape[1:] != (len(labels), 2):
            raise ValueError("Expected panels by queries by two probabilities")
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("Nonfinite predictions")
        difference = 100.0 * ((after.argmax(-1) == labels).astype(float) - (before.argmax(-1) == labels))
        panel_effect = metric_means(difference, labels, domains)
        weights = panel_counts.astype(float) / before.shape[0]
        if not np.all(panel_counts.sum(1) == before.shape[0]):
            raise ValueError("Invalid panel multiplicities")
        support_samples = weights @ panel_effect
        weighted_difference = weights @ difference
        query_samples, crossed_samples = [], []
        for d in (0, 1):
            qweight = np.zeros_like(query_counts, dtype=float)
            for c in (0, 1):
                selected = (domains == d) & (labels == c)
                if not np.all(query_counts[:, selected].sum(1) == selected.sum()):
                    raise ValueError("Invalid query cell multiplicities")
                qweight[:, selected] = query_counts[:, selected] / (2 * selected.sum())
            query_samples.append(qweight @ difference.mean(0))
            crossed_samples.append(np.einsum("rq,rq->r", weighted_difference, qweight))

        def combine(samples):
            return np.stack([(samples[0] + samples[1]) / 2, *samples], axis=-1)

        def interval(samples):
            bounds = np.quantile(samples, [0.025, 0.975], axis=0).T
            return {"lower": bounds[:, 0], "upper": bounds[:, 1], "half_width": np.diff(bounds, axis=1)[:, 0] / 2}

        return {
            "mean_gain_pp": panel_effect.mean(0),
            "panel_gain_pp": panel_effect,
            "positive_panel_fraction": (panel_effect > 1e-8).mean(0),
            "negative_panel_fraction": (panel_effect < -1e-8).mean(0),
            "support": interval(support_samples),
            "query": interval(combine(query_samples)),
            "crossed": interval(combine(crossed_samples)),
        }

    def summarize(self, records):
        """A bounded measurement/opportunity gate, not filter qualification."""
        support = np.array([r["fresh"]["support"]["half_width"][1:] for r in records])
        crossed = np.array([r["fresh"]["crossed"]["half_width"][1:] for r in records])
        support_fraction = float((support <= self.support_half_width_pp + 1e-8).mean())
        crossed_fraction = float((crossed <= self.crossed_half_width_pp + 1e-8).mean())
        useful = sorted(
            {
                r["source"]
                for r in records
                if r["fresh"]["mean_gain_pp"][0] >= 0.25 and r["fresh"]["crossed"]["lower"][0] > 0
            }
        )
        harmful = sorted(
            {r["source"] for r in records if np.any(np.asarray(r["fresh"]["crossed"]["upper"])[1:] < -1.0)}
        )
        gates = {
            "support_precision": support_fraction >= self.required_fraction,
            "crossed_precision": crossed_fraction >= self.required_fraction,
            "useful_sources": len(useful) >= 2,
            "harmful_sources": len(harmful) >= 2,
        }
        return {
            "gates": gates,
            "measurement_qualified": all(gates.values()),
            "support_precision_fraction": support_fraction,
            "crossed_precision_fraction": crossed_fraction,
            "median_support_half_width_pp": np.median(support, axis=0),
            "median_query_half_width_pp": np.median([r["fresh"]["query"]["half_width"][1:] for r in records], axis=0),
            "median_crossed_half_width_pp": np.median(crossed, axis=0),
            "useful_sources": useful,
            "harmful_sources": harmful,
            "decision": "prepare_separate_health_bridge_protocol"
            if all(gates.values())
            else "stop_before_health_fitting",
        }
