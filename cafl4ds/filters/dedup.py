"""SemDeDup-style near-duplicate removal — an admission knob (Abbas et al. 2023).

SemDeDup deduplicates a dataset by embedding it, clustering, and within each cluster dropping
points whose cosine similarity to a kept neighbour exceeds a threshold — keeping one
representative per near-duplicate group. Here we specialise the *within-cluster* step to a
single streaming batch: the batch is small, so no k-means is needed — we encode it with the
**current** model, then greedily keep points and drop any later point too similar (cosine) to
an already-kept one. This trims the redundancy a correlated stream pours in (many near-identical
consecutive frames), so the SSL update sees a more diverse diet.

This is a *label-free* selection heuristic over the live embedding (the fast edge); it never
touches labels or the SSL loss. Composes with the other admission knobs and, upstream of a
:class:`~cafl4ds.filters.reservoir.ReservoirReplay`, gives baseline **B1.5** (reservoir + dedup).
"""

from __future__ import annotations

import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext


def semantic_dedup_keep(embeddings: torch.Tensor, threshold: float) -> torch.Tensor:
    """Greedy near-duplicate removal in embedding space (the SemDeDup within-cluster step).

    Iterates the rows in order, keeping a point unless its cosine similarity to some
    already-kept point exceeds ``threshold`` (in which case it is a near-duplicate and
    dropped). Keeping the *earliest* of each near-duplicate group preserves stream order.

    Args:
        embeddings: Row-wise embeddings ``[B, d]`` (need not be normalized).
        threshold: Cosine-similarity above which two points are treated as near-duplicates,
            in ``[-1, 1]``. Higher = more permissive (only near-identical points are dropped);
            ``>= 1.0`` keeps everything.

    Returns:
        A 1-D ``long`` tensor of kept row indices, in ascending (stream) order.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return torch.arange(n, dtype=torch.long, device=embeddings.device)
    normed = torch.nn.functional.normalize(embeddings, dim=1)
    kept: list[int] = []
    for i in range(n):
        if not kept:
            kept.append(i)
            continue
        sims = normed[i] @ normed[torch.tensor(kept, device=embeddings.device)].T  # [len(kept)]
        if float(sims.max()) <= threshold:
            kept.append(i)
    return torch.tensor(kept, dtype=torch.long, device=embeddings.device)


class SemDeDup(Filter):
    """Admission knob: drop near-duplicate frames by cosine similarity in embedding space."""

    def __init__(self, threshold: float = 0.9) -> None:
        """Configure the deduplicator.

        Args:
            threshold: Cosine-similarity above which a frame is dropped as a near-duplicate of
                an already-kept frame. ``0.9`` keeps clearly-distinct frames and removes only
                the near-identical ones; raise toward ``1.0`` to dedup less aggressively.
        """
        self.threshold = threshold
        self.seen_images = 0
        self.kept_images = 0

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Encode the batch with the live model and return its near-duplicate-free subset.

        Args:
            batch: The incoming label-free stream batch.
            ctx: The live filter context; its ``method`` supplies the embedding (fast edge).

        Returns:
            The kept images ``[K, C, H, W]`` (``K <= B``), earliest of each duplicate group.
        """
        embeddings = ctx.method.encode(batch.images)  # [B, d], no grad
        keep = semantic_dedup_keep(embeddings, self.threshold)
        self.seen_images += batch.images.shape[0]
        self.kept_images += keep.numel()
        return batch.images[keep]

    def stats(self) -> dict[str, float | int]:
        """Return cumulative admission counts for experiment provenance."""
        return {
            "seen_images": self.seen_images,
            "kept_images": self.kept_images,
            "keep_fraction": self.kept_images / self.seen_images if self.seen_images else 0.0,
        }
