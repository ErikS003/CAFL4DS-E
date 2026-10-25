"""Causal traced reservoir/minimum-redundancy memory with matched retrieval RNG schedules."""

from __future__ import annotations

import time
from typing import Any

import torch

from cafl4ds.filters.base import FilterContext, ReplayBuffer
from cafl4ds.models.vit import TinyViTEncoder


class MemoryAdmissionReplay(ReplayBuffer):
    """Change retained history while preserving the original 32-image replay recipe."""

    def __init__(
        self,
        policy: str,
        seed: int,
        reference: TinyViTEncoder | None = None,
        capacity: int = 256,
    ) -> None:
        """Configure one fixed policy and its independent frozen reference, when required."""
        if policy not in {"reservoir", "minred"} or capacity < 16:
            raise ValueError("Expected reservoir/minred and capacity >=16")
        if policy == "minred" and reference is None:
            raise ValueError("Minimum redundancy requires a frozen reference")
        self.policy = policy
        self.capacity = capacity
        self.reference = reference
        if reference is not None:
            reference.eval().requires_grad_(False)
        self.generator = torch.Generator().manual_seed(seed)
        self.images: list[torch.Tensor] = []
        self.identities: list[int] = []
        self.descriptors = torch.empty((0, 0), dtype=torch.float64)
        self.similarities = torch.empty((0, 0), dtype=torch.float64)
        self.seen = 0
        self.encoding_seconds = 0.0
        self.admission_seconds = 0.0
        self.last_trace: dict[str, Any] = {}
        self.last_features = torch.empty(0)

    def observe(self, images: torch.Tensor, ctx: FilterContext) -> torch.Tensor:
        """Use arrival ordinals as label-free identifiers in a generic streaming loop."""
        del ctx
        return self.observe_with_ids(images, torch.arange(self.seen, self.seen + len(images)))

    def observe_with_ids(self, images: torch.Tensor, identities: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Retrieve genuine history before offering all current arrivals to memory."""
        if len(images) != 32 or len(identities) != 32 or len(set(identities.tolist())) != 32:
            raise ValueError("Expected32 unique current identities")
        replay_slots = (
            [] if not self.images else torch.randperm(len(self.images), generator=self.generator)[:16].tolist()
        )
        current_slots = (
            list(range(32)) if not replay_slots else torch.randperm(32, generator=self.generator)[:16].tolist()
        )
        replay_ids = [self.identities[i] for i in replay_slots]
        current_ids = [int(identities[i]) for i in current_slots]
        selected = images[current_slots]
        if replay_slots:
            selected = torch.cat([selected, torch.stack([self.images[i] for i in replay_slots])])
        features = None
        if self.reference is not None:
            tick = time.perf_counter()
            with torch.no_grad():
                raw = self.reference.embed(images).detach()
            if not torch.isfinite(raw).all() or torch.any(raw.double().norm(dim=1) <= 1e-12):
                raise ValueError("Invalid reference descriptors")
            self.last_features = raw.clone()
            features = torch.nn.functional.normalize(raw.double(), dim=1)
            self.encoding_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        events = []
        for offset, image in enumerate(images):
            proposed = None
            if self.seen >= self.capacity:
                proposed = int(torch.randint(0, self.seen + 1, (1,), generator=self.generator))
            identity = int(identities[offset])
            if self.policy == "reservoir":
                evicted = None
                if len(self.images) < self.capacity:
                    self.images.append(image.detach().clone())
                    self.identities.append(identity)
                elif proposed is not None and proposed < self.capacity:
                    evicted = self.identities[proposed]
                    self.images[proposed] = image.detach().clone()
                    self.identities[proposed] = identity
                else:
                    evicted = identity
            else:
                if features is None:
                    raise ValueError("Missing minimum-redundancy coordinates")
                evicted = self._admit(image, identity, features[offset])
            events.append({"arrival": identity, "evicted": evicted})
            self.seen += 1
        self.admission_seconds += time.perf_counter() - tick
        self.last_trace = {
            "current": identities.tolist(),
            "selected_current": current_ids,
            "replay": replay_ids,
            "events": events,
            "memory": self.identities.copy(),
        }
        return selected

    def _admit(self, image: torch.Tensor, identity: int, feature: torch.Tensor) -> int | None:
        """Append then discard the newest member of a maximally similar pair."""
        n = len(self.images)
        similarity = torch.empty((n + 1, n + 1), dtype=torch.float64)
        if n:
            similarity[:n, :n] = self.similarities
            pair = torch.round((self.descriptors @ feature) * 1e12) / 1e12
            similarity[n, :n] = pair
            similarity[:n, n] = pair
        similarity[n, n] = -torch.inf
        self.images.append(image.detach().clone())
        self.identities.append(identity)
        self.descriptors = feature[None] if not n else torch.cat([self.descriptors, feature[None]])
        evicted = None
        if len(self.images) > self.capacity:
            nearest = similarity.max(dim=1).values
            victim = int((nearest == nearest.max()).nonzero()[-1])
            evicted = self.identities.pop(victim)
            self.images.pop(victim)
            keep = torch.arange(n + 1) != victim
            self.descriptors = self.descriptors[keep]
            similarity = similarity[keep][:, keep]
        self.similarities = similarity
        return evicted

    def persistent_bytes(self) -> dict[str, int]:
        """Report image memory separately from policy/reference overhead."""
        return {
            "images": sum(x.numel() * x.element_size() for x in self.images),
            "descriptors": self.descriptors.numel() * self.descriptors.element_size(),
            "similarities": self.similarities.numel() * self.similarities.element_size(),
            "reference_parameters": 0
            if self.reference is None
            else sum(x.numel() * x.element_size() for x in self.reference.parameters()),
            "reference_buffers": 0
            if self.reference is None
            else sum(x.numel() * x.element_size() for x in self.reference.buffers()),
        }
