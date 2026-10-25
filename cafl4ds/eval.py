"""Downstream evaluation — probe-on-past, forgetting, and the B5 adaptation floor.

The *validating* axis of the study (the payoff, `[STD]`), kept separate from the *diagnostic*
health signals in :mod:`cafl4ds.monitor`. Three pieces, all label-in-the-probe-only:

* **Probe-on-past** — the current encoder, evaluated on every past era's held-out set, builds
  an accuracy matrix ``R[i][j]`` = accuracy on era ``j`` after training through era ``i``
  (:class:`PerEraProbe`, one row recorded per era as the correlated stream advances).
* **Forgetting metrics** — from that matrix, :func:`backward_transfer` (Lopez-Paz & Ranzato
  2017) and :func:`forgetting_measure` (Chaudhry et al. 2018): how much learning later eras
  eroded accuracy on earlier ones. The direct readout of *degradation present* (question **b**).
* **Adaptation vs. B5** — :func:`adaptation_report` compares the adapted encoder's downstream
  probe accuracy against the init-matched **frozen** backbone (baseline B5,
  :func:`cafl4ds.measurements.frozen_baseline`): does adapting beat doing nothing (question
  **a**)?

Forgetting is a *correlated-stream* phenomenon: with a single era (IID) there is no past to
forget, so the matrix has one row and the forgetting metrics are ``None`` (undefined).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch

from cafl4ds import measurements
from cafl4ds.data.streams import EvalSet, EvalSets
from cafl4ds.measurements import Encoder

# An accuracy matrix: ``matrix[i][j]`` = accuracy on era ``j`` after training through era ``i``.
AccuracyMatrix = dict[int, dict[int, float]]


class EraSummary(TypedDict):
    """The reduced probe-on-past report (final per-era accuracies + forgetting metrics)."""

    per_era_final: dict[int, float]
    backward_transfer: float | None
    forgetting_measure: float | None
    num_eras: int


def _probe(encode: Encoder, support: EvalSet, query: EvalSet, probe: str, knn_k: int) -> float:
    """Run one downstream probe (kNN or linear) on frozen features.

    Args:
        encode: Frozen encoder callable ``inputs -> embeddings``.
        support: The probe's support/database split.
        query: The split to score.
        probe: ``"knn"`` or ``"linear"``.
        knn_k: Neighbour count for the kNN probe (ignored by the linear probe).

    Returns:
        Top-1 query accuracy in ``[0, 1]``.

    Raises:
        ValueError: If ``probe`` is not ``"knn"`` or ``"linear"``.
    """
    if probe == "knn":
        return measurements.knn_probe(encode, (support.images, support.labels), (query.images, query.labels), k=knn_k)
    if probe == "linear":
        return measurements.linear_probe(encode, (support.images, support.labels), (query.images, query.labels))
    raise ValueError(f"probe must be 'knn' or 'linear'; got {probe!r}.")


def few_shot_probe_curve(
    encode: Encoder,
    eval_sets: EvalSets,
    shots: Sequence[int] = (1, 2, 5, 10, 20),
    *,
    probe: str = "linear",
    knn_k: int = 20,
) -> dict[int, float]:
    """Evaluate a frozen encoder with balanced, nested support subsets.

    Each point uses the first ``n`` examples of every class in the fixed probe-support split and
    scores on the unchanged probe-query split. The subsets are deterministic, so an adapted/frozen
    pair differs only through encoder features. Labels train only the readout, never the encoder.
    """
    support = eval_sets.probe_support
    classes = sorted({int(label) for label in support.labels.tolist()})
    curve: dict[int, float] = {}
    for shot in shots:
        if shot < 1:
            raise ValueError(f"few-shot counts must be positive; got {shot}.")
        selected: list[torch.Tensor] = []
        for cls in classes:
            indices = (support.labels == cls).nonzero(as_tuple=True)[0]
            if indices.numel() < shot:
                raise ValueError(
                    f"class {cls} has {indices.numel()} support examples, fewer than requested {shot}-shot."
                )
            selected.append(indices[:shot])
        idx = torch.cat(selected)
        subset = EvalSet(support.images[idx], support.labels[idx])
        curve[int(shot)] = _probe(encode, subset, eval_sets.probe_query, probe, knn_k)
    return curve


class PerEraProbe:
    """Accumulates the per-era accuracy matrix by probing the current encoder over past eras.

    Uses the stream's fixed :attr:`~cafl4ds.data.streams.EvalSets.probe_support` as the probe
    database and each era's held-out :attr:`~cafl4ds.data.streams.EvalSets.per_era` set as the
    query. The loop calls :meth:`record` once per era (as it completes) with the live encoder,
    which appends one matrix row; :meth:`summary` reduces the matrix to the reportable metrics.
    """

    def __init__(self, eval_sets: EvalSets, probe: str = "knn", knn_k: int = 20) -> None:
        """Configure the per-era probe.

        Args:
            eval_sets: The stream's held-out eval sets (fixed support + per-era queries).
            probe: Downstream probe to use, ``"knn"`` or ``"linear"`` (labels used HERE ONLY).
            knn_k: Neighbour count for the kNN probe.
        """
        self.eval_sets = eval_sets
        self.probe = probe
        self.knn_k = knn_k
        self.matrix: AccuracyMatrix = {}

    def record(self, encode: Encoder, era_completed: int) -> dict[int, float]:
        """Probe the current encoder on every era seen so far and store the row.

        Args:
            encode: The live (frozen for the probe) encoder callable.
            era_completed: Index of the era just finished; the encoder is scored on eras
                ``0 .. era_completed`` (past eras + the one just learned).

        Returns:
            The recorded row ``{era: accuracy}`` (also stored in :attr:`matrix`).
        """
        row = {
            era: _probe(encode, self.eval_sets.probe_support, eval_set, self.probe, self.knn_k)
            for era, eval_set in sorted(self.eval_sets.per_era.items())
            if era <= era_completed
        }
        self.matrix[era_completed] = row
        return row

    def summary(self) -> EraSummary:
        """Reduce the accumulated matrix to the reportable per-era / forgetting metrics.

        Returns:
            An :class:`EraSummary`. The forgetting metrics are ``None`` when fewer than two eras
            were recorded (no past to forget — e.g. an IID stream).
        """
        final_era = max(self.matrix) if self.matrix else None
        return {
            "per_era_final": self.matrix.get(final_era, {}) if final_era is not None else {},
            "backward_transfer": backward_transfer(self.matrix),
            "forgetting_measure": forgetting_measure(self.matrix),
            "num_eras": len(self.matrix),
        }


def backward_transfer(matrix: AccuracyMatrix) -> float | None:
    """Backward Transfer: mean change in past-era accuracy from first-learned to final.

    ``BWT = mean_{j < T-1} (R[T-1][j] - R[j][j])`` (Lopez-Paz & Ranzato 2017), where ``R[j][j]``
    is accuracy on era ``j`` right after learning it and ``R[T-1][j]`` is its accuracy at the
    end. **Negative** BWT means later eras eroded earlier ones (forgetting); positive means
    later learning helped past eras.

    Args:
        matrix: The per-era accuracy matrix (one row per completed era).

    Returns:
        The BWT, or ``None`` if fewer than two eras were recorded.
    """
    eras = sorted(matrix)
    if len(eras) < 2:
        return None
    final = eras[-1]
    row_final = matrix[final]
    diffs = [row_final[j] - matrix[j][j] for j in eras[:-1] if j in row_final and j in matrix[j]]
    return sum(diffs) / len(diffs) if diffs else None


def forgetting_measure(matrix: AccuracyMatrix) -> float | None:
    """Forgetting Measure: mean drop from an era's *best-ever* accuracy to its final accuracy.

    ``FM = mean_{j < T-1} ( max_{j <= l < T-1} R[l][j] - R[T-1][j] )`` (Chaudhry et al. 2018) —
    for each past era, how far its accuracy fell from the highest it reached at any earlier
    checkpoint to its value at the end. Higher = more forgetting.

    Args:
        matrix: The per-era accuracy matrix (one row per completed era).

    Returns:
        The forgetting measure, or ``None`` if fewer than two eras were recorded.
    """
    eras = sorted(matrix)
    if len(eras) < 2:
        return None
    final = eras[-1]
    row_final = matrix[final]
    forgets = []
    for j in eras[:-1]:
        prior = [matrix[k][j] for k in eras if j <= k < final and j in matrix[k]]
        if prior and j in row_final:
            forgets.append(max(prior) - row_final[j])
    return sum(forgets) / len(forgets) if forgets else None


def adaptation_report(
    adapted_encode: Encoder,
    frozen_encode: Encoder,
    eval_sets: EvalSets,
    probe: str = "knn",
    knn_k: int = 20,
) -> dict[str, float]:
    """Compare the adapted encoder against the init-matched frozen backbone (B5).

    Answers the existential question (a): does adapting the backbone beat the frozen model?
    Both are scored with the same downstream probe on the stream's fixed support/query split.

    Args:
        adapted_encode: The encoder after streaming adaptation.
        frozen_encode: The init-matched never-updated encoder (B5: frozen-pretrained in the
            pretrained regime, frozen-random from scratch).
        eval_sets: The stream's held-out eval sets (fixed support + query).
        probe: Downstream probe, ``"knn"`` or ``"linear"``.
        knn_k: Neighbour count for the kNN probe.

    Returns:
        ``{"adapted_acc", "b5_acc", "gain"}`` — the two accuracies and ``adapted - b5``.
    """
    support, query = eval_sets.probe_support, eval_sets.probe_query
    adapted_acc = _probe(adapted_encode, support, query, probe, knn_k)
    kwargs = {"k": knn_k} if probe == "knn" else {}
    b5_acc = measurements.frozen_baseline(
        frozen_encode, (support.images, support.labels), (query.images, query.labels), probe=probe, **kwargs
    )
    return {"adapted_acc": adapted_acc, "b5_acc": b5_acc, "gain": adapted_acc - b5_acc}
