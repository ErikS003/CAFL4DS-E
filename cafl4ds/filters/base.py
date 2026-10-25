"""The selection-filter interface (the ``A`` factor / independent variable).

The filter is the knob the whole study turns. The ``A`` factor has **two composable roles**,
because the cheap Phase-0 knobs are not all the same kind of thing:

* An **admission** :class:`Filter` *narrows* an incoming batch to a subset — B-floor
  (:class:`~cafl4ds.filters.accept_all.AcceptAll`, accept everything), SemDeDup
  (near-duplicate removal), the loss-gate (keep high-loss frames). Admission filters are pure
  batch reducers, so they **compose freely as a list** (apply one after another).
* A **replay** :class:`ReplayBuffer` decides what is *stored and replayed* — it may emit past
  items, so it can *grow* the training batch, and it carries state across steps. Reservoir
  sampling (:class:`~cafl4ds.filters.reservoir.ReservoirReplay`) is the Phase-0 instance. A
  buffer is a single-choice role and always runs **last** (after admission).

:class:`~cafl4ds.filters.composite.CompositeSelector` combines an admission list with an
optional buffer behind the single :class:`Filter` interface the loop calls, which is what lets
the plan's baselines be selected by config — B-floor, B1 (reservoir), dedup, loss-gate, and
B1.5 (dedup ∘ reservoir). Later phases add novelty (F-a), coverage (F-b), and the
health-steerable filter (F-c) behind these same interfaces.

The :class:`FilterContext` deliberately carries the live model and step so that
model-in-the-loop filters (the *fast edge*) and, later, health state (the *slow edge*) have
what they need without changing the loop's call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.ssl.base import SSLMethod


@dataclass
class FilterContext:
    """Everything a filter may consult when selecting from a batch."""

    method: SSLMethod
    """The live SSL method — its encoder scores informativeness against the *current* model
    (the fast edge). B-floor ignores it."""
    step: int
    """The current global stream step."""


class Filter(ABC):
    """Selects which images from an incoming stream batch enter the SSL update."""

    @abstractmethod
    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Return the images accepted from ``batch`` for this update.

        Args:
            batch: The incoming label-free stream batch.
            ctx: The live filter context (model, step).

        Returns:
            The accepted images ``[K, C, H, W]`` with ``0 <= K <= B`` (possibly the whole
            batch, possibly empty).
        """

    def auxiliary_loss(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor | None:
        """Return an optional training regularizer associated with the selected batch.

        Selection filters normally return ``None``. Replay implementations may use this hook
        to constrain replayed examples while keeping the primary SSL objective and optimizer
        step in the streaming loop unchanged.
        """
        del images, ctx
        return None


class ReplayBuffer(ABC):
    """Stores incoming images and emits the batch the SSL update trains on (the replay role).

    Unlike an admission :class:`Filter`, a buffer holds state across steps and may return
    *more* items than it was given (the incoming batch mixed with replayed past items), so it
    is a distinct role that runs after admission. Phase 0 has one instance,
    :class:`~cafl4ds.filters.reservoir.ReservoirReplay`.
    """

    @abstractmethod
    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Ingest the admitted images and return the batch to train on this step.

        Args:
            images: The admitted images ``[K, C, H, W]`` (post-admission) for this step.
            ctx: The live filter context (model, step).

        Returns:
            The training batch ``[M, C, H, W]`` — typically the incoming images plus a replay
            sample drawn from the buffer's stored history (``M >= K``).
        """

    def auxiliary_loss(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor | None:
        """Return an optional replay regularizer for the most recent :meth:`observe` call."""
        del images, ctx
        return None
