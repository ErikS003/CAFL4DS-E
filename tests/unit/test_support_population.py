"""Exercise support sampling before the expensive matched branch replay."""

import pytest
import torch

from cafl4ds.support_population import disjoint_panels


def test_disjoint_balanced_private_samples() -> None:
    """Neither panel can use excluded query/support paths, duplicates or training RNG."""
    cells = {(d, c): [f"{d}_{c}_{i:03}" for i in range(100)] for d in (0, 1) for c in (0, 1, 2)}
    excluded = {paths[0] for paths in cells.values()}
    before = torch.get_rng_state().clone()
    panels = disjoint_panels(cells, excluded)
    assert torch.equal(before, torch.get_rng_state())
    assert panels == disjoint_panels({k: list(reversed(v)) for k, v in cells.items()}, excluded)
    assert len(set(panels[0] + panels[1])) == 528
    for panel in panels:
        assert len(panel) == 264 and not set(panel) & excluded
        for d, c in cells:
            assert sum(p.startswith(f"{d}_{c}_") for p in panel) == 44
    with pytest.raises(ValueError, match="Insufficient"):
        disjoint_panels({k: v[:87] for k, v in cells.items()}, set())
