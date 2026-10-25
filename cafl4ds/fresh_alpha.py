"""Disjoint data roles and isolated action-gradient observations for alpha research."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.image_utility import gradient_vector, utility_features
from cafl4ds.ssl.base import SSLMethod


def allocate_roles(cells: dict[str, dict[str, Any]], counts: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Sample balanced, globally disjoint roles without silently reducing requested counts."""
    if set(cells) != {f"{d}_{c}" for d in range(2) for c in range(3)}:
        raise ValueError("Require six canonical domain/scene cells")
    if not counts or any(n < 1 for n in counts.values()):
        raise ValueError("Role counts must be positive")
    available = [path for cell in cells.values() for path in cell["unused_paths"]]
    if len(set(available)) != len(available):
        raise ValueError("Duplicate image paths across cells")
    generator = torch.Generator().manual_seed(seed)
    result: dict[str, list[str]] = {role: [] for role in counts}
    for key in sorted(cells):
        paths = sorted(cells[key]["unused_paths"])
        if len(paths) < sum(counts.values()):
            raise ValueError(f"Cell {key} has {len(paths)} images; require {sum(counts.values())}")
        order = torch.randperm(len(paths), generator=generator).tolist()
        start = 0
        for role, count in counts.items():
            result[role].extend(paths[i] for i in order[start : start + count])
            start += count
    return result


def support_folds(
    labels: torch.Tensor, domains: torch.Tensor, support: torch.Tensor, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Four balanced folds for variable-sized support panels; private RNG only."""
    generator = torch.Generator().manual_seed(seed)
    cells = []
    for domain in (0, 1):
        for label in (0, 1, 2):
            cell = support[(domains[support] == domain) & (labels[support] == label)]
            if len(cell) < 4 or len(cell) % 4:
                raise ValueError("Support cells must contain a positive multiple of four images")
            cells.append(cell[torch.randperm(len(cell), generator=generator)].chunk(4))
    for fold in range(4):
        validation = torch.cat([cell[fold] for cell in cells])
        yield support[~torch.isin(support, validation)], validation


def action_observations(
    method: SSLMethod,
    selector: SteerableReplay,
    images: torch.Tensor,
    step: int,
    monitor: torch.Tensor,
    warm_monitor: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Observe hypothetical selected batches on private selectors, with no optimizer step."""
    rows, health = utility_features(method, selector, images, monitor, warm_monitor, gradient_signals=True)
    arrival = {
        f"arrival_{key}": sum(row[key] for row in rows) / len(rows) for key in rows[0] if not key.startswith("health_")
    }
    replay = torch.stack(selector.reservoir._buffer)
    replay_gradient = gradient_vector(method, replay[:64], 76001)
    actions = {}
    for alpha in (0.0, 0.5, 1.0):
        private = copy.deepcopy(selector)
        private.set_action(alpha)
        batch = private.select(StreamBatch(images, era=0, step=step), FilterContext(method, step))
        indices = private.last["current_indices"]
        gradient = gradient_vector(method, batch, 76002)
        actions[str(alpha)] = {
            "features": {
                "alpha": alpha,
                "step": float(step),
                "history_alpha": selector.alpha,
                **health,
                **arrival,
                **{
                    f"selected_{key}": sum(rows[i][key] for i in indices) / len(indices)
                    for key in rows[0]
                    if not key.startswith("health_")
                },
                "selected_batch_gradient_norm": float(gradient.norm()),
                "selected_batch_replay_cosine": float(F.cosine_similarity(gradient, replay_gradient, dim=0)),
            },
            "current_indices": indices,
        }
    return health, actions
