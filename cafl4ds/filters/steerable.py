"""Coverage/novelty admission with an action-independent all-arrival reservoir."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.filters.reservoir import ReservoirReplay
from cafl4ds.models.vit import TinyViTEncoder


def midranks(values: torch.Tensor) -> torch.Tensor:
    """Return normalized midranks, with .5 for constant components and singletons."""
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("Scores must be a nonempty finite vector")
    if len(values) == 1:
        return torch.full_like(values, 0.5)
    below = (values[:, None] > values[None, :]).sum(1)
    tied = (values[:, None] == values[None, :]).sum(1)
    return (below + (tied - 1) / 2) / (len(values) - 1)


def fit_anchors(embeddings: torch.Tensor, count: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit fixed coverage coordinates on warm TRAIN features, without labels."""
    if count < 2 or embeddings.ndim != 2 or len(embeddings) < count or not torch.isfinite(embeddings).all():
        raise ValueError("Need finite warm embeddings and at least two anchors")
    normalized = F.normalize(embeddings.detach().cpu(), dim=1)
    fit = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(normalized.numpy())
    centers = F.normalize(torch.tensor(fit.cluster_centers_, dtype=normalized.dtype), dim=1)
    # Targets use precisely the cosine assignment subsequently used by the actuator.
    assignments = (normalized @ centers.T).argmax(1)
    counts = assignments.bincount(minlength=count).float()
    if (counts == 0).any():
        raise ValueError("Degenerate anchors: an empty cosine-assignment cluster")
    return centers, counts / counts.sum()


class SteerableReplay(Filter):
    """Select current images by alpha; store ALL arrivals and replay uniform history."""

    def __init__(
        self,
        reference_encoder: TinyViTEncoder,
        centroids: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.5,
        capacity: int = 256,
        replay_batch: int = 16,
        batch_size: int = 32,
        half_life: float = 128,
        seed: int = 0,
    ) -> None:
        """Freeze coverage coordinates and initialize independent replay/scoring state."""
        if batch_size <= replay_batch or capacity < replay_batch or replay_batch < 1:
            raise ValueError("Require batch_size > replay_batch >= 1 and capacity >= replay_batch")
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be positive and finite")
        if (
            centroids.ndim != 2
            or target.shape != (len(centroids),)
            or not torch.isfinite(centroids).all()
            or not torch.isfinite(target).all()
            or (centroids.norm(dim=1) == 0).any()
            or (target <= 0).any()
            or not torch.isclose(target.sum(), torch.tensor(1.0, device=target.device))
        ):
            raise ValueError("Require finite nonzero centroids and positive normalized target")
        self.reference_encoder = copy.deepcopy(reference_encoder).eval()
        self.reference_encoder.requires_grad_(False)
        self.centroids = F.normalize(centroids.detach().cpu().clone(), dim=1)
        self.target = target.detach().cpu().clone()
        self.occupancy = self.target.clone()
        self.decay = 2 ** (-1 / half_life)
        self.batch_size = batch_size
        self.seed = seed
        self.reservoir = ReservoirReplay(capacity, replay_batch, seed)
        self.optimizer_images = 0
        self.updates = 0
        self.last: dict[str, Any] = {}
        self.set_action(alpha)

    def set_action(self, alpha: float) -> None:
        """Set a bounded action for subsequent selections only."""
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")
        self.alpha = float(alpha)

    def _assign(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = F.normalize(self.reference_encoder.embed(images), dim=1)
        return (z @ self.centroids.to(z.device).T).argmax(1).cpu()

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Score before training, replay before ingest, and keep exact optimizer exposures."""
        if len(batch.images) != self.batch_size:
            raise ValueError("This actuator requires full incoming batches")
        self.reference_encoder.to(batch.images.device)
        assignments = self._assign(batch.images)
        coverage = (self.target - self.occupancy).clamp_min(0)[assignments].to(batch.images.device)
        modes = {module: module.training for module in ctx.method.modules()}
        try:
            ctx.method.eval()
            # Scoring uses paired private draws and cannot perturb training RNG.
            devices = [] if batch.images.device.type == "cpu" else [batch.images.device.index or 0]
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed((self.seed + 104729 * (ctx.step + 1)) % (2**63 - 1))
                novelty = ctx.method.per_sample_loss(batch.images)
        finally:
            for module, training in modes.items():
                module.training = training
        scores = self.alpha * midranks(coverage) + (1 - self.alpha) * midranks(novelty)
        replay = self.reservoir._sample_replay()
        count = 0 if replay is None else len(replay)
        keep = torch.argsort(scores, descending=True, stable=True)[: self.batch_size - count]
        # Original order controls augmentation pairing and deterministic total-score ties.
        keep = keep.sort().values
        chosen = batch.images[keep]
        selected = chosen if replay is None else torch.cat((chosen, replay.to(chosen.device)))
        selected_clusters = self._assign(selected)
        exposure = selected_clusters.bincount(minlength=len(self.target)).float() / len(selected)
        self.occupancy = self.decay * self.occupancy + (1 - self.decay) * exposure
        for image in batch.images:
            self.reservoir._ingest(image.cpu())
        self.optimizer_images += len(selected)
        self.updates += 1
        self.last = {
            "step": ctx.step,
            "alpha": self.alpha,
            "current_indices": keep.cpu().tolist(),
            "current_count": len(chosen),
            "replay_count": count,
            "available_clusters": assignments.bincount(minlength=len(self.target)).tolist(),
            "selected_clusters": selected_clusters.bincount(minlength=len(self.target)).tolist(),
            "coverage_span": float(coverage.max() - coverage.min()),
            "novelty_span": float(novelty.max() - novelty.min()),
            "reservoir_seen": self.reservoir._seen,
            "occupancy": self.occupancy.tolist(),
        }
        return selected

    def state_dict(self) -> dict[str, Any]:
        """Capture mutable actuator state for exact branches or same-spec resume."""
        return copy.deepcopy(
            {
                "specification": {
                    "centroids": self.centroids,
                    "target": self.target,
                    "batch_size": self.batch_size,
                    "capacity": self.reservoir.capacity,
                    "replay_batch": self.reservoir.replay_batch,
                    "decay": self.decay,
                    "seed": self.seed,
                    "reference": self.reference_encoder.state_dict(),
                },
                "alpha": self.alpha,
                "occupancy": self.occupancy,
                "buffer": self.reservoir._buffer,
                "seen": self.reservoir._seen,
                "rng": self.reservoir._gen.get_state(),
                "optimizer_images": self.optimizer_images,
                "updates": self.updates,
                "last": self.last,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore mutable state into an actuator constructed with the identical specification."""
        state = copy.deepcopy(state)
        expected = self.state_dict()["specification"]
        received = state["specification"]
        for key in ("batch_size", "capacity", "replay_batch", "decay", "seed"):
            if expected[key] != received[key]:
                raise ValueError(f"Actuator specification mismatch: {key}")
        for key in ("centroids", "target"):
            if not torch.equal(expected[key], received[key]):
                raise ValueError(f"Actuator specification mismatch: {key}")
        if expected["reference"].keys() != received["reference"].keys() or any(
            not torch.equal(value.cpu(), received["reference"][key].cpu())
            for key, value in expected["reference"].items()
        ):
            raise ValueError("Actuator reference encoder mismatch")
        if (
            state["occupancy"].shape != self.target.shape
            or not torch.isfinite(state["occupancy"]).all()
            or len(state["buffer"]) > self.reservoir.capacity
            or state["seen"] < len(state["buffer"])
        ):
            raise ValueError("Invalid mutable actuator state")
        self.set_action(state["alpha"])
        self.occupancy = state["occupancy"]
        self.reservoir._buffer = state["buffer"]
        self.reservoir._seen = state["seen"]
        self.reservoir._gen.set_state(state["rng"])
        self.optimizer_images = state["optimizer_images"]
        self.updates = state["updates"]
        self.last = state["last"]
