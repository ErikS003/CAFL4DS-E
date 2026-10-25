"""Fixed-budget replay selection and cross-draw opportunity diagnostics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio

RULES = ("uniform", "coverage", "task_gradient", "mixed")


def coverage_selection(z: torch.Tensor, current: torch.Tensor, count: int = 16) -> list[int]:
    """Greedily cover normalized replay features, initially anchored by current images."""
    if len(z) < count or not len(current) or not torch.isfinite(z).all():
        raise ValueError("Require finite candidates and nonempty current anchors")
    candidates = F.normalize(z.double(), dim=1)
    anchors = F.normalize(current.double(), dim=1)
    distances = (1 - candidates @ anchors.T).clamp_min(0).min(dim=1).values
    chosen = []
    for _ in range(count):
        index = int(distances.argmax())
        chosen.append(index)
        distances = torch.minimum(distances, (1 - candidates @ candidates[index]).clamp_min(0))
        distances[chosen] = -1
    return chosen


def ranked_blocks(scores: torch.Tensor, uniform: list[int]) -> dict[str, list[int]]:
    """Rank eight disjoint blocks of eight; mixed keeps eight baseline images."""
    if scores.shape != (8,) or not torch.isfinite(scores).all() or uniform != list(range(16)):
        raise ValueError("Require eight finite block scores and the nested baseline")
    ranking = sorted(range(8), key=lambda i: (-float(scores[i]), i))
    targeted = [i for block in ranking[:2] for i in range(block * 8, (block + 1) * 8)]
    mixed_block = next(block for block in ranking if block != 0)
    return {"task_gradient": targeted, "mixed": list(range(8)) + list(range(mixed_block * 8, (mixed_block + 1) * 8))}


class FixedReplaySelection(ReplayRatio):
    """Replace only the first replay selection, retaining action-independent memory RNG."""

    def __init__(self, replay_indices: list[int], first_step: int, seed: int) -> None:
        """Use the ordinary16/32 selection before substituting the frozen replay set."""
        super().__init__(16, seed=seed)
        if len(replay_indices) != 16 or len(set(replay_indices)) != 16 or min(replay_indices) < 0:
            raise ValueError("Require16 distinct nonnegative reservoir positions")
        self.replay_indices = replay_indices
        self.first_step = first_step

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Capture old images before ingestion; later steps use ordinary uniform replay."""
        replacement = None
        if ctx.step == self.first_step:
            replacement = torch.stack([self.reservoir._buffer[i] for i in self.replay_indices])
        selected = super().select(batch, ctx)
        if replacement is not None:
            selected = torch.cat((selected[:16], replacement.to(selected.device)))
            self.last["replay_indices"] = self.replay_indices.copy()
        return selected


class SustainedReplayCoverage(ReplayRatio):
    """Recompute coverage on every step without changing reservoir or selection RNG."""

    def __init__(self, seed: int, observation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Use isolated encoder observations, a64-image shortlist and16 replay slots."""
        super().__init__(16, seed=seed)
        self.observation = observation
        self.features = torch.empty(0)

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Observe old memory before ingestion; canonicalize membership in shortlist order."""
        if len(batch.images) != 32 or len(self.reservoir._buffer) < 64:
            raise ValueError("Require32 arrivals and at least64 historical images")
        generator = torch.Generator()
        generator.set_state(self.reservoir._gen.get_state().clone())
        candidates = torch.randperm(len(self.reservoir._buffer), generator=generator)[:64]
        current_generator = torch.Generator().manual_seed(self.seed + 104729 * (ctx.step + 1))
        current_indices = torch.randperm(32, generator=current_generator)[:16].sort().values
        current = batch.images[current_indices.to(batch.images.device)]
        memory = torch.stack([self.reservoir._buffer[i] for i in candidates.tolist()]).to(current.device)
        before = time.perf_counter()
        self.features = self.observation(torch.cat((current, memory)))
        observation_seconds = time.perf_counter() - before
        positions = sorted(coverage_selection(self.features[16:], self.features[:16]))
        replacement = memory[positions]
        selected = super().select(batch, ctx)
        if self.last["replay_indices"] != candidates[:16].tolist():
            raise ValueError("Coverage preview differs from ordinary uniform draw")
        self.last.update(
            replay_indices=candidates[positions].tolist(),
            candidate_indices=candidates.tolist(),
            selected_positions=positions,
            observation_seconds=observation_seconds,
        )
        return torch.cat((selected[:16], replacement))


class CoveragePulse(SustainedReplayCoverage):
    """Apply the same coverage rule for a fixed duration, then ordinary uniform replay."""

    def __init__(
        self, seed: int, observation: Callable[[torch.Tensor], torch.Tensor], first_step: int, duration: int = 4
    ) -> None:
        """Pin the pulse before outcomes rather than choosing its length from a rollout."""
        super().__init__(seed, observation)
        if duration < 1:
            raise ValueError("Pulse duration must be positive")
        self.first_step, self.duration = first_step, duration

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Coverage is active only within the prespecified initial interval."""
        if self.first_step <= ctx.step < self.first_step + self.duration:
            return super().select(batch, ctx)
        return ReplayRatio.select(self, batch, ctx)


def cross_draw_opportunities(values: np.ndarray, minimum_gain: float = 0.001) -> dict[str, Any]:
    """Choose with other draws, score an untouched draw; diagnose future-label opportunity."""
    if values.ndim != 2 or values.shape[1] != len(RULES) or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Require at least two finite draws across all four rules")
    records = []
    for heldout in range(len(values)):
        training = np.delete(values, heldout, axis=0).mean(axis=0)
        gains = training - training[0]
        best = int(np.argmax(gains)) if gains.max() >= minimum_gain else 0
        records.append(
            {
                "heldout_draw": heldout,
                "selected_rule": RULES[best],
                "gain": float(values[heldout, best] - values[heldout, 0]),
            }
        )
    return {"folds": records, "mean_gain": float(np.mean([r["gain"] for r in records]))}
