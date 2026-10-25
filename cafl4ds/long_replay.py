"""Long paired replay interventions with explicit repeated-arrival provenance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio
from cafl4ds.structured_replay import structured_choices

RULES = ("uniform", "recent", "cluster_balanced")
HORIZONS = (1, 8, 16, 32)
INTERVENTION_UPDATES = 8


def continuation(order: list[int], prefix: int, updates: int = 32) -> list[int]:
    """Use the original tail, then cycle its original second-domain block."""
    if len(order) != 512 or len(set(order)) != 512 or not 10 <= prefix < 16 or updates < 1:
        raise ValueError("Require the unique512-image two-domain order and a second-domain parent")
    tail = order[prefix * 32 :].copy()
    block = order[320:]
    while len(tail) < updates * 32:
        tail.extend(block)
    return tail[: updates * 32]


class ReplayIntervention(ReplayRatio):
    """Change replay for eight updates while preserving reservoir draws and ingestion."""

    def __init__(
        self,
        rule: str,
        first_step: int,
        seed: int,
        observe: Callable[[Any, torch.Tensor], torch.Tensor],
    ) -> None:
        """Inject an isolated observation, without giving the selector any labels."""
        super().__init__(16, seed=seed)
        if rule not in RULES:
            raise ValueError("Unknown replay rule")
        self.rule, self.first_step, self.observe = rule, first_step, observe
        self.arrivals: list[int] = []
        self.observation_embeddings: torch.Tensor | None = None

    def restore_arrivals(self, arrivals: list[int]) -> None:
        """Arrival occurrences remain unique even when decoded images recur."""
        if len(arrivals) != len(self.reservoir._buffer) or len(set(arrivals)) != len(arrivals):
            raise ValueError("Incorrect initial memory arrival mapping")
        if min(arrivals) < 0 or max(arrivals) >= self.reservoir._seen:
            raise ValueError("Future memory arrival")
        self.arrivals = arrivals.copy()

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Observe before optimization; replay selection never changes memory admission."""
        if len(self.arrivals) != len(self.reservoir._buffer):
            raise ValueError("Missing arrival mapping")
        generator = torch.Generator().set_state(self.reservoir._gen.get_state())
        candidates = torch.randperm(len(self.arrivals), generator=generator)[:64].tolist()
        observed_arrivals = [self.arrivals[i] for i in candidates]
        seen = self.reservoir._seen
        active = 0 <= ctx.step - self.first_step < INTERVENTION_UPDATES
        replacement = None
        observation: dict[str, Any] = {}
        self.observation_embeddings = None
        if active:
            old = torch.stack([self.reservoir._buffer[i] for i in candidates])
            z = self.observe(ctx.method, old)
            self.observation_embeddings = z
            if self.rule == "cluster_balanced":
                choices, clusters = structured_choices(observed_arrivals, z)
                positions = choices[self.rule]
            elif self.rule == "recent":
                positions = sorted(sorted(range(64), key=lambda i: observed_arrivals[i])[-16:])
                clusters = []
            else:
                positions, clusters = list(range(16)), []
            replacement = old[positions].to(batch.images.device)
            normalized = F.normalize(z.double(), dim=1)
            singular = torch.linalg.svdvals(z.double() - z.double().mean(0))
            mass = singular / singular.sum().clamp_min(1e-12)
            rank = float(torch.exp(-(mass * mass.clamp_min(1e-12).log()).sum()))
            observation = {
                "candidate_indices": candidates,
                "candidate_arrivals": observed_arrivals,
                "selected_positions": positions,
                "cluster_labels": clusters,
                "overlap_uniform": len(set(positions) & set(range(16))),
                "mean_age": sum(seen - 1 - observed_arrivals[i] for i in positions) / 16,
                "coverage": float((1 - normalized @ normalized[positions].T).clamp_min(0).min(1).values.mean()),
                "selected_variance": float(normalized[positions].var(0).mean()),
                "candidate_variance": float(normalized.var(0).mean()),
                "candidate_rankme": rank,
            }
        # Replay the admission RNG separately to track occurrence age exactly.
        for position in range(seen, seen + len(batch.images)):
            if len(self.arrivals) < self.reservoir.capacity:
                self.arrivals.append(position)
            else:
                slot = int(torch.randint(0, position + 1, (1,), generator=generator).item())
                if slot < self.reservoir.capacity:
                    self.arrivals[slot] = position
        selected = super().select(batch, ctx)
        if not torch.equal(generator.get_state(), self.reservoir._gen.get_state()):
            raise ValueError("Arrival tracking changed the admission RNG")
        if replacement is not None:
            selected = torch.cat((selected[:16], replacement))
            self.last["replay_indices"] = [candidates[i] for i in observation["selected_positions"]]
        self.last.update(intervention_active=active, observation=observation)
        return selected
