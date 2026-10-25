"""Read-only recognition outcomes and exact ideal alpha ranking geometry."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import torch

from cafl4ds.filters.steerable import midranks
from cafl4ds.image_utility import FixedProbe, balanced_losses, embed, outcome_losses
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from scripts.evaluate_checkpoint_robustness import probe_predictions


def diagnostic_outcomes(
    method: SSLMethod,
    evaluation: Evaluation,
    roles: dict[str, torch.Tensor],
    fixed: FixedProbe,
    draws: int = 3,
) -> dict[str, dict[str, float]]:
    """Refit labeled support only; outcome query labels never enter SSL or probe fitting."""
    result = outcome_losses(method, evaluation, roles, fixed, draws, reconstruction=True)
    refitted = FixedProbe.fit(method, evaluation)
    support = next(iter(evaluation.supports.values()))
    support_z = embed(method, evaluation.images[support])
    query_z = {role: embed(method, evaluation.images[roles[role]]) for role in ("development", "holdout")}
    predictions = probe_predictions("knn", support_z, evaluation.labels[support], torch.cat(list(query_z.values())))
    start = 0
    for role, z in query_z.items():
        indices = roles[role]
        labels, domains = evaluation.labels[indices], evaluation.domains[indices]
        probabilities = refitted.classifier.predict_proba((z.numpy() - refitted.mean) / refitted.scale)
        linear = torch.from_numpy(refitted.classifier.classes_[probabilities.argmax(1)])
        metrics = {
            "refit_ce": refitted.losses(z, labels),
            "linear_error": (linear != labels).double(),
            "knn_error": (predictions[start : start + len(indices)] != labels).double(),
        }
        for name, values in metrics.items():
            result[role].update(
                {f"{name}_{key}": value for key, value in balanced_losses(values, labels, domains).items()}
            )
        start += len(indices)
    return result


def implemented_keep(coverage: torch.Tensor, novelty: torch.Tensor, alpha: float, k: int) -> list[int]:
    """Use the actual float32 rank blend and stable original-arrival tie order."""
    scores = alpha * midranks(coverage) + (1 - alpha) * midranks(novelty)
    return torch.argsort(scores, descending=True, stable=True)[:k].sort().values.tolist()


def alpha_geometry(coverage: torch.Tensor, novelty: torch.Tensor, k: int) -> dict[str, Any]:  # noqa: C901 - rational and machine-score enumeration
    """Enumerate exact rational line crossings, with float32 implementation checks nearby."""
    if coverage.shape != novelty.shape or not 1 <= k <= len(coverage):
        raise ValueError("Need matching finite score vectors and a valid selection size")
    midranks(coverage)
    midranks(novelty)

    def numerators(values: torch.Tensor) -> list[int]:
        below = (values[:, None] > values[None, :]).sum(1)
        ties = (values[:, None] == values[None, :]).sum(1)
        return (2 * below + ties - 1).tolist()

    cov, loss = numerators(coverage), numerators(novelty)
    slopes = [c - n for c, n in zip(cov, loss, strict=True)]
    boundaries = {Fraction(0), Fraction(1)}
    for i in range(len(loss)):
        for j in range(i):
            if slopes[i] != slopes[j]:
                crossing = Fraction(loss[j] - loss[i], slopes[i] - slopes[j])
                if 0 <= crossing <= 1:
                    boundaries.add(crossing)
    ordered = sorted(boundaries)
    points = set(ordered)
    points.update((a + b) / 2 for a, b in zip(ordered[:-1], ordered[1:], strict=True))
    ideal: set[tuple[int, ...]] = set()
    for alpha in sorted(points):
        ranking = sorted(range(len(loss)), key=lambda i: (-(loss[i] + alpha * slopes[i]), i))
        ideal.add(tuple(sorted(ranking[:k])))
    checked = {float(alpha) for alpha in points}
    for alpha in ordered:
        value = np.float32(float(alpha))
        for direction in (np.float32(0), np.float32(1)):
            nearby = float(np.nextafter(value, direction))
            if 0 <= nearby <= 1:
                checked.add(nearby)
    actual: dict[tuple[int, ...], float] = {}
    mismatches = 0
    for checked_alpha in sorted(checked):
        keep = tuple(implemented_keep(coverage, novelty, checked_alpha, k))
        actual.setdefault(keep, checked_alpha)
    for alpha in sorted(points):
        ranking = sorted(range(len(loss)), key=lambda i: (-(loss[i] + alpha * slopes[i]), i))
        mismatches += tuple(sorted(ranking[:k])) != tuple(implemented_keep(coverage, novelty, float(alpha), k))
    return {
        "ideal_reachable_sets": [list(keep) for keep in sorted(ideal)],
        "implemented_checked_sets": [{"keep": list(keep), "alpha": alpha} for keep, alpha in actual.items()],
        "crossing_points": len(ordered),
        "checked_alpha_values": len(checked),
        "ideal_vs_float32_point_mismatches": mismatches,
        "limits": (
            "Ideal rank geometry is exhaustive; float32 checks use crossings/interiors/neighbors, "
            "not every machine alpha"
        ),
    }
