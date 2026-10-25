"""Pilot-based budget planning; projections are not additional observations."""

from __future__ import annotations

import numpy as np


class PrecisionBudgetPlanner:
    """Project paired-bootstrap precision at candidate support/query budgets."""

    def __init__(self, candidates=(48, 96, 144, 192, 256), repetitions=1000, seed=62501):
        self.candidates = list(candidates)
        self.repetitions = repetitions
        self.seed = seed

    def project(self, responses, labels, domains, query_counts):
        """Resample empirical pilot panels/images at proposed larger sample sizes."""
        responses, labels, domains = map(np.asarray, (responses, labels, domains))
        assert responses.ndim == 3 and responses.shape[-1] == len(labels)
        results = []
        for panels in self.candidates:
            rng = np.random.default_rng(self.seed)
            panel_weights = (
                rng.multinomial(panels, np.full(responses.shape[1], 1 / responses.shape[1]), size=self.repetitions)
                / panels
            )
            query_weights = []
            for domain in (0, 1):
                weight = np.zeros((self.repetitions, len(labels)))
                for label in (0, 1):
                    selected = np.flatnonzero((domains == domain) & (labels == label))
                    count = int(query_counts[f"{domain}/{label}"])
                    if count <= 0 or not len(selected):
                        raise ValueError("Query cells must be nonempty")
                    weight[:, selected] = rng.multinomial(
                        count, np.full(len(selected), 1 / len(selected)), size=self.repetitions
                    ) / (2 * count)
                query_weights.append(weight)
            support_widths, crossed_widths = [], []
            for response in responses:
                panel_effects = np.stack(
                    [
                        np.mean([response[:, (domains == d) & (labels == c)].mean(1) for c in (0, 1)], axis=0)
                        for d in (0, 1)
                    ],
                    axis=1,
                )
                support = panel_weights @ panel_effects
                weighted_response = panel_weights @ response
                crossed = np.stack(
                    [np.einsum("rq,rq->r", weighted_response, weight) for weight in query_weights], axis=1
                )
                support_widths.append(np.diff(np.quantile(support, [0.025, 0.975], axis=0), axis=0)[0] / 2)
                crossed_widths.append(np.diff(np.quantile(crossed, [0.025, 0.975], axis=0), axis=0)[0] / 2)
            support_widths, crossed_widths = np.array(support_widths), np.array(crossed_widths)
            results.append(
                {
                    "panels": panels,
                    "support_precision_fraction": float((support_widths <= 0.5 + 1e-8).mean()),
                    "crossed_precision_fraction": float((crossed_widths <= 1 + 1e-8).mean()),
                    "median_support_half_width_pp": np.median(support_widths, axis=0),
                    "median_crossed_half_width_pp": np.median(crossed_widths, axis=0),
                }
            )
        qualifying = [
            r["panels"]
            for r in results
            if r["support_precision_fraction"] >= 0.8 and r["crossed_precision_fraction"] >= 0.8
        ]
        return {
            "candidates": results,
            "selected_panels": min(qualifying) if qualifying else None,
            "scope": "empirical_pilot_projection_not_new_evidence",
        }


def planned_seeds(original, total=256, start=62001):
    """Preserve the old draw prefix and add a fixed, disjoint seed sequence."""
    result = [*original, *range(start, start + total - len(original))]
    if len(set(result)) != total:
        raise ValueError("Support seeds must be unique")
    return result


def query_precision_gate(records, required_fraction=0.8):
    """A prespecified interim resource gate; it cannot promote the final study."""
    widths = np.array([r["fresh"]["query"]["half_width"][1:] for r in records])
    fraction = float((widths <= 1 + 1e-8).mean())
    return {
        "query_precision_fraction": fraction,
        "continue_to_final_budget": fraction >= required_fraction,
        "required_fraction": required_fraction,
    }
