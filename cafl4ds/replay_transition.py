"""Matched batch-order interventions with fixed image membership and work."""

from __future__ import annotations

import torch


def schedules(order: torch.Tensor, domains: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reorder twelve pure-domain batches without changing any batch membership."""
    if order.shape != (384,) or len(set(order.tolist())) != 384:
        raise ValueError("Require384 distinct arrivals")
    batches = order.reshape(12, 32)
    groups = domains[batches]
    if not torch.equal(groups, groups[:, :1].expand_as(groups)):
        raise ValueError("Require domain-pure batches")
    day, night = batches[groups[:, 0] == 0], batches[groups[:, 0] == 1]
    if day.shape != (6, 32) or night.shape != (6, 32):
        raise ValueError("Require six day and six night batches")
    return {
        "day_then_night": torch.cat((day, night)).flatten(),
        "night_then_day": torch.cat((night, day)).flatten(),
        "alternating_day_first": torch.stack((day, night), dim=1).flatten(),
        "alternating_night_first": torch.stack((night, day), dim=1).flatten(),
    }
