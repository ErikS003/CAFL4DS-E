"""The regime stream — the P1.0.2 deployment diet's attribute-driven era ordering.

Where :class:`~cafl4ds.data.streams.EraStream` orders a *single* class axis into eras (and labels its
eval sets on that same axis), the deployment diet needs the two-axis structure an
:class:`~cafl4ds.data.attributes.AttributeSource` provides: it orders **eras on the era-key axis**
(the driving regime) while reserving its held-out probe set on the **orthogonal canary axis** (so the
labelled canary reads something the diet did not sort by). It is otherwise a faithful sibling of
``EraStream`` — single-pass, a batch is never split across eras, and ``block_size`` is the same
correlation-strength knob — so it drops straight into the Phase-1 harness's ``run_stream_arm``.

Kept in its own module (importing only the stable :class:`~cafl4ds.data.streams.StreamBatch` /
:class:`~cafl4ds.data.streams.EvalSet` / :class:`~cafl4ds.data.streams.EvalSets` dataclasses) so the
Phase-0 stream is untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from cafl4ds.data.attributes import AttributedImages, AttributeSource
from cafl4ds.data.streams import EvalSet, EvalSets, StreamBatch


def iter_ordered_batches(
    order: list[tuple[int, int]], images: torch.Tensor, batch_size: int, drop_last: bool
) -> Iterator[StreamBatch]:
    """Yield label-free batches from an ordered ``(era, image_index)`` stream.

    A batch is never split across eras: each batch carries the era of its first sample, and a new
    batch starts at every era boundary (mirroring ``EraStream``'s contract exactly).

    Args:
        order: The ordered ``(era, image_index)`` delivery stream.
        images: The image tensor to index ``[N, C, H, W]``.
        batch_size: Images per batch.
        drop_last: Whether to drop the final short batch of the whole stream.

    Yields:
        Successive :class:`StreamBatch` batches.
    """
    step = 0
    buffer: list[int] = []
    buffer_era: int | None = None
    for era, image_index in order:
        if buffer_era is not None and era != buffer_era and buffer:
            yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=buffer_era, step=step)
            step += 1
            buffer = []
        buffer_era = era
        buffer.append(image_index)
        if len(buffer) == batch_size:
            yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=era, step=step)
            step += 1
            buffer = []
    if buffer and not drop_last and buffer_era is not None:
        yield StreamBatch(images=images[torch.tensor(buffer, dtype=torch.long)], era=buffer_era, step=step)


def count_ordered_batches(order: list[tuple[int, int]], batch_size: int, drop_last: bool) -> int:
    """Count the batches :func:`iter_ordered_batches` will yield (era-boundary flush included).

    Args:
        order: The ordered ``(era, image_index)`` stream.
        batch_size: Images per batch.
        drop_last: Whether the final short batch is dropped.

    Returns:
        The exact batch count.
    """
    count = 0
    buffer = 0
    buffer_era: int | None = None
    for era, _ in order:
        if buffer_era is not None and era != buffer_era and buffer:
            count += 1
            buffer = 0
        buffer_era = era
        buffer += 1
        if buffer == batch_size:
            count += 1
            buffer = 0
    if buffer and not drop_last and buffer_era is not None:
        count += 1
    return count


def _reserve_global_eval(
    attributed: AttributedImages,
    support_per_canary: int,
    query_per_canary: int,
    generator: torch.Generator,
    require_training_remainder: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Reserve balanced global support/query indices from one evaluation corpus."""
    canary = attributed.canary
    support_idx, query_idx = [], []
    available = torch.ones(attributed.images.shape[0], dtype=torch.bool)
    for cls in sorted(set(canary.tolist())):
        cls_idx = (canary == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
        need = support_per_canary + query_per_canary
        too_small = perm.numel() < need or (require_training_remainder and perm.numel() == need)
        if too_small:
            raise ValueError(
                f"canary class {cls} has {perm.numel()} images but {need} are reserved for probes; "
                "reduce support_per_canary / query_per_canary or use more data."
            )
        support_idx.append(perm[:support_per_canary])
        query_idx.append(perm[support_per_canary:need])
        available[perm[:need]] = False
    return support_idx, query_idx, available


def _reserve_per_era_eval(
    train_attributed: AttributedImages,
    eval_attributed: AttributedImages,
    requested: list[int],
    available: torch.Tensor,
    era_query_per_cell: int,
    generator: torch.Generator,
    require_training_remainder: bool,
) -> dict[int, EvalSet]:
    """Reserve condition queries, matching train and evaluation regimes by stable names."""
    if era_query_per_cell == 0:
        return {}
    eval_regime_by_name = {name: regime for regime, name in eval_attributed.regime_names.items()}
    per_era: dict[int, EvalSet] = {}
    for era, regime in enumerate(requested):
        eval_regime = eval_regime_by_name.get(train_attributed.regime_names[regime])
        if eval_regime is None:
            continue
        era_indices: list[torch.Tensor] = []
        for cls in sorted(set(eval_attributed.canary.tolist())):
            candidates = (
                (eval_attributed.era_key == eval_regime) & (eval_attributed.canary == cls) & available
            ).nonzero(as_tuple=True)[0]
            remainder = 1 if require_training_remainder else 0
            take = min(era_query_per_cell, max(0, candidates.numel() - remainder))
            if take:
                perm = candidates[torch.randperm(candidates.numel(), generator=generator)]
                chosen = perm[:take]
                era_indices.append(chosen)
                available[chosen] = False
        if era_indices:
            idx = torch.cat(era_indices)
            per_era[era] = EvalSet(eval_attributed.images[idx], eval_attributed.canary[idx])
    return per_era


class RegimeStream:
    """A single-pass diet that orders images by driving regime and probes an orthogonal canary axis.

    Reserves a balanced held-out probe support/query on the **canary** axis, then orders the remaining
    images into eras that walk the regimes in ``regime_order`` — contiguous by default (maximal
    correlation), interleaved by ``block_size`` (the correlation-strength knob). Delivers label-free
    :class:`StreamBatch` batches whose ``era`` is the regime's position in the walk.
    """

    def __init__(
        self,
        source: AttributeSource,
        evaluation_source: AttributeSource | None = None,
        batch_size: int = 32,
        regime_order: list[int] | None = None,
        block_size: int | None = None,
        support_per_canary: int = 10,
        query_per_canary: int = 10,
        era_query_per_cell: int = 0,
        max_train_per_regime: int | None = None,
        drop_last: bool = False,
        seed: int = 0,
        canary_seed: int = 0,
    ) -> None:
        """Build the stream (loads the source and constructs the splits eagerly).

        Args:
            source: The attributed data source (two axes: era-key regime + canary).
            evaluation_source: Optional independent source used only for probe support/query and
                condition queries. When supplied, every image in ``source`` remains eligible for
                online training and no evaluation image can enter the update stream.
            batch_size: Images per delivered batch.
            regime_order: Explicit regime walk order; defaults to the source's shift-walk order.
            block_size: Correlation knob — ``None`` delivers each regime contiguously (maximal
                correlation, following the walk); an integer ``b`` round-robins the regimes in chunks
                of ``b`` (interleaving them). Pass a multiple of ``batch_size`` to avoid confounding
                correlation with a shrinking effective batch.
            support_per_canary: Images per canary class reserved for the probe support set.
            query_per_canary: Images per canary class reserved for the probe query / drift set.
            era_query_per_cell: Maximum additional images reserved from every ``(regime, canary)`` cell
                for condition-specific probe queries. A positive value enables probe-on-past:
                after each regime, the same global probe support is scored on the current and all
                earlier regime-specific queries. These images are disjoint from training and the
                global support/query sets. Sparse or absent cells contribute fewer images, while at
                least one available image remains for training. ``0`` preserves the deployment-only
                reservation.
            max_train_per_regime: If set, cap the training images per regime (bounds run length).
            drop_last: Whether to drop the final short batch.
            seed: RNG seed for the **within-regime shuffle** (frame order inside each era, and — with
                ``max_train_per_regime`` — which frames survive). Varies per drive across a seed
                ensemble; does **not** touch the held-out reservation (see ``canary_seed``).
            canary_seed: RNG seed for the **held-out canary reservation** — which support/query
                frames each scene sets aside. Deliberately *decoupled* from ``seed`` and fixed by
                default, so every drive in a seed ensemble (and the warm well, which reserves the
                same way) holds out the **identical** probe set, disjoint from all SSL training. A
                per-drive reservation would leave each drive probed on images an earlier drive's well
                had already seen unsupervised — a soft transductive leak (audit P1.0 A1).

        Raises:
            ValueError: If ``block_size`` is non-positive, or a canary class has too few images to
                satisfy the requested held-out reservations.
        """
        if block_size is not None and block_size < 1:
            raise ValueError(f"block_size must be a positive integer; got {block_size}.")
        if era_query_per_cell < 0:
            raise ValueError(f"era_query_per_cell must be non-negative; got {era_query_per_cell}.")
        self.batch_size = batch_size
        self.block_size = block_size
        self.drop_last = drop_last
        self._generator = torch.Generator().manual_seed(seed)
        reserve_generator = torch.Generator().manual_seed(canary_seed)

        attributed = source.load()
        self._images = attributed.images
        self.regime_names = attributed.regime_names
        self.canary_names = attributed.canary_names
        self._object_categories = attributed.object_categories
        era_key, canary = attributed.era_key, attributed.canary
        self._canary = canary

        eval_attributed = evaluation_source.load() if evaluation_source is not None else attributed
        eval_images = eval_attributed.images
        eval_canary = eval_attributed.canary
        self.eval_num_canary_classes = len(set(eval_canary.tolist()))

        # Reserve a balanced held-out probe set on the canary axis; the rest is the training pool.
        # The reservation draws from `reserve_generator` (seeded by `canary_seed`, not the drive
        # `seed`), so the probe set is identical across every drive and the warm well — never a
        # per-drive split that would let a later drive train on an earlier drive's probe images.
        requested = regime_order if regime_order is not None else attributed.regime_order
        internal_eval = evaluation_source is None
        support_idx, query_idx, eval_available = _reserve_global_eval(
            eval_attributed,
            support_per_canary,
            query_per_canary,
            reserve_generator,
            require_training_remainder=internal_eval,
        )
        per_era_eval = _reserve_per_era_eval(
            attributed,
            eval_attributed,
            requested,
            eval_available,
            era_query_per_cell,
            reserve_generator,
            require_training_remainder=internal_eval,
        )
        train_mask = eval_available if internal_eval else torch.ones(self._images.shape[0], dtype=torch.bool)

        self._eval_sets = EvalSets(
            probe_support=EvalSet(eval_images[torch.cat(support_idx)], eval_canary[torch.cat(support_idx)]),
            probe_query=EvalSet(eval_images[torch.cat(query_idx)], eval_canary[torch.cat(query_idx)]),
            per_era=per_era_eval,
        )
        self._regime_order = requested
        self._order_stream = self._build_order(era_key, train_mask, requested, max_train_per_regime)

    def _build_order(
        self, era_key: torch.Tensor, train_mask: torch.Tensor, regime_order: list[int], max_train_per_regime: int | None
    ) -> list[tuple[int, int]]:
        """Order the training pool into eras walking ``regime_order`` (era = walk position).

        Args:
            era_key: Per-image regime id ``[N]``.
            train_mask: Boolean mask of training (non-reserved) images ``[N]``.
            regime_order: The regime ids to walk, in order (regimes with no training data are skipped).
            max_train_per_regime: Optional per-regime training cap.

        Returns:
            The ordered ``(era, image_index)`` stream (contiguous eras, or round-robin if
            ``block_size`` is set).
        """
        train_by_era: list[tuple[int, list[int]]] = []
        for era, regime in enumerate(regime_order):
            regime_idx = ((era_key == regime) & train_mask).nonzero(as_tuple=True)[0]
            if regime_idx.numel() == 0:
                continue
            shuffled = regime_idx[torch.randperm(regime_idx.numel(), generator=self._generator)].tolist()
            if max_train_per_regime is not None:
                shuffled = shuffled[:max_train_per_regime]
            train_by_era.append((era, shuffled))
        if self.block_size is not None:
            return self._round_robin(train_by_era, self.block_size)
        return [(era, i) for era, items in train_by_era for i in items]

    @staticmethod
    def _round_robin(train_by_era: list[tuple[int, list[int]]], block_size: int) -> list[tuple[int, int]]:
        """Round-robin the eras in chunks of ``block_size`` (the correlation-strength knob).

        Emits ``block_size`` images of the first era, then ``block_size`` of the next, … and repeats
        until every era is exhausted; the same-regime run length is ``block_size`` (a large value
        recovers contiguous regimes). The ``era`` tag stays the regime's walk position across
        recurrences.

        Args:
            train_by_era: ``(era, indices)`` pairs of training images per regime, in walk order.
            block_size: Number of consecutive same-era images per chunk.

        Returns:
            The round-robin ``(era, image_index)`` stream.
        """
        pointers = [0] * len(train_by_era)
        ordered: list[tuple[int, int]] = []
        while any(pointers[k] < len(train_by_era[k][1]) for k in range(len(train_by_era))):
            for k, (era, items) in enumerate(train_by_era):
                start = pointers[k]
                for i in items[start : start + block_size]:
                    ordered.append((era, i))
                pointers[k] = start + block_size
        return ordered

    def stationary_batches(self) -> Iterator[StreamBatch]:
        """Yield the training pool in fully shuffled, regime-agnostic order — the warmup diet.

        Where :meth:`__iter__` delivers the *nonstationary* regime walk, this reshuffles the same
        reserved training pool with no regime ordering (every batch tagged ``era=0``), so a warm-up
        pass sees a stationary all-regime mix. The reserved canary probe set is excluded exactly as in
        the drive, so a well warmed here is competent on the same probes the drive reads.

        Yields:
            Stationary :class:`StreamBatch` batches over the training pool.
        """
        indices = [idx for _, idx in self._order_stream]
        perm = torch.randperm(len(indices), generator=self._generator)
        order = [(0, indices[p]) for p in perm.tolist()]
        return iter_ordered_batches(order, self._images, self.batch_size, self.drop_last)

    @property
    def eval_sets(self) -> EvalSets:
        """The held-out eval sets (probe support/query on the canary axis)."""
        return self._eval_sets

    @property
    def era_names(self) -> dict[int, str]:
        """Map each era (walk position) present in the stream to its human regime name.

        Resolves the era tag carried on every batch — its position in the regime walk — back to the
        source's ``(timeofday·weather)`` name, so the deploy corpus can label each leg. Only eras with
        training data (those actually delivered) appear.
        """
        present = {era for era, _ in self._order_stream}
        return {era: self.regime_names[self._regime_order[era]] for era in sorted(present)}

    def era_composition(self) -> dict[int, dict[str, Any]]:
        """Per-era label composition of the delivered training pool — the leg's self-description.

        Aggregates, for each era actually delivered, the canary (scene) histogram and — when the
        source carries detection labels (BDD) — the object-category histogram, over the images that
        era streams. This is the label-derived tag set the deploy corpus attaches to each leg, so a
        consumer can read what each ``sunny → rain → …`` block *contained* without the raw frames.

        Returns:
            ``era -> {"n_images", "scene_hist", "category_hist"}`` (histograms sorted for
            determinism; ``category_hist`` is empty when the source has no detection labels).
        """
        composition: dict[int, dict[str, Any]] = {}
        for era, idx in self._order_stream:
            entry = composition.setdefault(era, {"n_images": 0, "scene_hist": {}, "category_hist": {}})
            entry["n_images"] += 1
            scene = self.canary_names[int(self._canary[idx])]
            entry["scene_hist"][scene] = entry["scene_hist"].get(scene, 0) + 1
            if self._object_categories is not None:
                for category, count in self._object_categories[idx].items():
                    entry["category_hist"][category] = entry["category_hist"].get(category, 0) + count
        for entry in composition.values():
            entry["scene_hist"] = dict(sorted(entry["scene_hist"].items()))
            entry["category_hist"] = dict(sorted(entry["category_hist"].items()))
        return composition

    @property
    def num_eras(self) -> int:
        """Number of distinct eras (regimes with training data) in the walk."""
        return len({era for era, _ in self._order_stream})

    def __len__(self) -> int:
        """Number of batches the stream will deliver (era-boundary flushes included)."""
        return count_ordered_batches(self._order_stream, self.batch_size, self.drop_last)

    def __iter__(self) -> Iterator[StreamBatch]:
        """Yield label-free :class:`StreamBatch` batches in walk order."""
        return iter_ordered_batches(self._order_stream, self._images, self.batch_size, self.drop_last)
