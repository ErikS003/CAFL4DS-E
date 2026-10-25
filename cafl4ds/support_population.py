"""Disjoint support-population samples with canonical domain and scene labels."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def disjoint_panels(
    cells: Mapping[tuple[int, int], list[str]], excluded: set[str], seed: int = 73011
) -> list[list[str]]:
    """Sample two disjoint 44-image-per-cell panels without consuming training RNG."""
    if set(cells) != {(d, c) for d in (0, 1) for c in (0, 1, 2)}:
        raise ValueError("Require six canonical domain/class cells")
    generator = torch.Generator().manual_seed(seed)
    panels: list[list[str]] = [[], []]
    for key in sorted(cells):
        candidates = sorted(set(cells[key]) - excluded)
        if len(candidates) < 88:
            raise ValueError(f"Insufficient independent support images: {key}: {len(candidates)}")
        chosen = [candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:88].tolist()]
        for panel in range(2):
            panels[panel].extend(chosen[panel * 44 : (panel + 1) * 44])
    if len(set(panels[0] + panels[1])) != 528:
        raise ValueError("Support panels overlap or images have conflicting labels")
    return panels
