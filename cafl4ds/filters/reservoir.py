"""Reservoir sampling + experience replay — the replay knob (Vitter 1985; baseline B1).

A fixed-capacity buffer maintained by **Vitter's Algorithm R**: after ``N`` frames have passed,
each is present with equal probability ``capacity / N``, so the buffer is a uniform random
sample of the whole (single-pass, unbounded) stream — no matter how correlated the arrival
order. Each step the model trains on the incoming frames **plus** a random replay draw from the
buffer, mixing past eras back into a correlated stream. That interleaving is the "dumb"
protective bar (B1): replay alone, no informativeness, partly counters forgetting.

The buffer is the **replay** role of the ``A`` factor (see :mod:`cafl4ds.filters.base`): it
runs after any admission filters, so what it stores and replays is the *admitted* stream —
upstream :class:`~cafl4ds.filters.dedup.SemDeDup` gives reservoir + dedup (baseline B1.5).
Sampling uses a private, seeded generator so a run is reproducible independently of whatever
else consumes the global RNG.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cafl4ds.filters.base import FilterContext, ReplayBuffer


class ReservoirReplay(ReplayBuffer):
    """Uniform reservoir buffer (Algorithm R) with per-step experience replay."""

    def __init__(
        self,
        capacity: int = 256,
        replay_batch: int = 32,
        seed: int = 0,
        max_train_batch: int | None = None,
    ) -> None:
        """Configure the reservoir.

        Args:
            capacity: Maximum number of frames held in the buffer.
            replay_batch: Number of past frames replayed alongside each incoming batch (a
                uniform draw, without replacement, from the current buffer; clamped to the
                buffer's size, so early steps replay fewer).
            seed: Seed for the buffer's private RNG (reservoir replacement + replay sampling).
            max_train_batch: Optional cap on the returned optimizer batch. When set, replay
                displaces a uniformly sampled subset of incoming images while every incoming
                image is still admitted to the reservoir. ``None`` preserves incoming-plus-replay.

        Raises:
            ValueError: If ``capacity`` or ``replay_batch`` is not positive.
        """
        if capacity < 1 or replay_batch < 1:
            raise ValueError(f"capacity and replay_batch must be >= 1; got {capacity}, {replay_batch}.")
        if max_train_batch is not None and (max_train_batch < 1 or replay_batch > max_train_batch):
            raise ValueError(
                "max_train_batch must be >= replay_batch >= 1; "
                f"got max_train_batch={max_train_batch}, replay_batch={replay_batch}."
            )
        self.capacity = capacity
        self.replay_batch = replay_batch
        self.max_train_batch = max_train_batch
        self._buffer: list[torch.Tensor] = []
        self._seen = 0  # total frames observed so far (Algorithm R's running count N)
        self._gen = torch.Generator().manual_seed(seed)  # CPU generator; indices only

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Replay from the buffer, then ingest the incoming frames (Algorithm R).

        Replay is sampled from the buffer *before* the incoming frames are ingested, so it is
        drawn from genuine history (never the just-arrived frames), then each incoming frame
        updates the reservoir.

        Args:
            images: The admitted images ``[K, C, H, W]`` for this step.
            ctx: Unused (reservoir replay is model-agnostic and random).

        Returns:
            The training batch ``[K + R, C, H, W]``: the incoming frames followed by ``R``
            replayed frames (``R = min(replay_batch, buffer size before this step)``).
        """
        del ctx  # reservoir selection is model-agnostic
        replay = self._sample_replay()
        current = images
        if self.max_train_batch is not None:
            replay_size = 0 if replay is None else replay.shape[0]
            current_size = min(images.shape[0], self.max_train_batch - replay_size)
            if current_size < images.shape[0]:
                idx = torch.randperm(images.shape[0], generator=self._gen)[:current_size]
                current = images[idx.to(images.device)]
        for i in range(images.shape[0]):
            self._ingest(images[i])
        return current if replay is None else torch.cat([current, replay], dim=0)

    def _sample_replay(self) -> torch.Tensor | None:
        """Draw a uniform replay batch from the current buffer (without replacement).

        Returns:
            A ``[R, C, H, W]`` stack of replayed frames, or ``None`` if the buffer is empty.
        """
        if not self._buffer:
            return None
        r = min(self.replay_batch, len(self._buffer))
        idx = torch.randperm(len(self._buffer), generator=self._gen)[:r].tolist()
        return torch.stack([self._buffer[i] for i in idx], dim=0)

    def _ingest(self, image: torch.Tensor) -> None:
        """Update the reservoir with one frame via Vitter's Algorithm R.

        While the buffer is not full every frame is stored; once full, the ``N``-th frame
        (0-based index ``N``) replaces a uniformly-random slot with probability
        ``capacity / (N + 1)`` — the invariant that keeps the buffer a uniform sample.

        Args:
            image: A single frame ``[C, H, W]`` to (maybe) store; a detached clone is kept.
        """
        item = image.detach().clone()
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            j = int(torch.randint(0, self._seen + 1, (1,), generator=self._gen).item())
            if j < self.capacity:
                self._buffer[j] = item
        self._seen += 1


class FeatureDistillationReservoirReplay(ReservoirReplay):
    """Compute-matched reservoir replay with cached backbone-feature targets.

    Each image is stored with the normalized encoder embedding it had on arrival. When that
    image is replayed, cosine distance from its cached embedding penalizes representation drift.
    The regularizer is label-free and applies only to replayed history, leaving current examples
    free to adapt under the primary SSL objective.
    """

    def __init__(
        self,
        capacity: int = 256,
        replay_batch: int = 16,
        seed: int = 0,
        max_train_batch: int | None = None,
        distill_weight: float = 1.0,
    ) -> None:
        """Configure replay and the weight on cached-feature cosine distance."""
        super().__init__(capacity, replay_batch, seed, max_train_batch)
        if distill_weight < 0.0:
            raise ValueError(f"distill_weight must be >= 0; got {distill_weight}.")
        self.distill_weight = distill_weight
        self._targets: list[torch.Tensor] = []
        self._last_targets: torch.Tensor | None = None
        self._loss_sum = 0.0
        self._weighted_loss_sum = 0.0
        self._loss_steps = 0
        self._replayed_images = 0

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Replay paired images/targets, then admit every arrival with its current embedding."""
        replay_pair = self._sample_replay_pair()
        replay = None if replay_pair is None else replay_pair[0]
        self._last_targets = None if replay_pair is None else replay_pair[1]
        current = images
        if self.max_train_batch is not None:
            replay_size = 0 if replay is None else replay.shape[0]
            current_size = min(images.shape[0], self.max_train_batch - replay_size)
            if current_size < images.shape[0]:
                idx = torch.randperm(images.shape[0], generator=self._gen)[:current_size]
                current = images[idx.to(images.device)]

        targets = self._encode_targets(images, ctx)
        for image, target in zip(images, targets, strict=True):
            self._ingest_pair(image, target)
        return current if replay is None else torch.cat([current, replay], dim=0)

    def auxiliary_loss(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor | None:
        """Penalize drift of replayed images from their cached arrival-time embeddings."""
        if self._last_targets is None or self._last_targets.shape[0] == 0:
            return None
        count = self._last_targets.shape[0]
        current = F.normalize(ctx.method.encoder.embed(images[-count:]), dim=1)
        targets = self._last_targets.to(device=current.device, dtype=current.dtype)
        raw = (1.0 - (current * targets).sum(dim=1)).mean()
        weighted = self.distill_weight * raw
        self._loss_sum += float(raw.detach().item())
        self._weighted_loss_sum += float(weighted.detach().item())
        self._loss_steps += 1
        self._replayed_images += count
        return weighted

    def stats(self) -> dict[str, float | int]:
        """Return cumulative runtime diagnostics for experiment provenance."""
        denom = max(self._loss_steps, 1)
        return {
            "distill_weight": self.distill_weight,
            "distill_steps": self._loss_steps,
            "replayed_images": self._replayed_images,
            "mean_feature_cosine_distance": self._loss_sum / denom,
            "mean_weighted_distill_loss": self._weighted_loss_sum / denom,
        }

    def _encode_targets(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Snapshot deterministic normalized backbone features before the current update."""
        encoder = ctx.method.encoder
        was_training = encoder.training
        encoder.eval()
        try:
            with torch.no_grad():
                return F.normalize(encoder.embed(images), dim=1).detach()
        finally:
            encoder.train(was_training)

    def _sample_replay_pair(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draw aligned replay images and cached feature targets from the reservoir."""
        if not self._buffer:
            return None
        count = min(self.replay_batch, len(self._buffer))
        indices = torch.randperm(len(self._buffer), generator=self._gen)[:count].tolist()
        return (
            torch.stack([self._buffer[index] for index in indices]),
            torch.stack([self._targets[index] for index in indices]),
        )

    def _ingest_pair(self, image: torch.Tensor, target: torch.Tensor) -> None:
        """Apply Algorithm R while keeping image and cached target slots aligned."""
        item = image.detach().clone()
        feature = target.detach().clone()
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
            self._targets.append(feature)
        else:
            index = int(torch.randint(0, self._seen + 1, (1,), generator=self._gen).item())
            if index < self.capacity:
                self._buffer[index] = item
                self._targets[index] = feature
        self._seen += 1
