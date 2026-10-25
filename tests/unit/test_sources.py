"""Property tests for the P0.3.8 distribution-shift transforms (audit P0.3 §E, E3).

``_to_grayscale`` (cosmetic, structure-preserving) and ``_phase_scramble`` (structure-destroying,
power-spectrum-preserving) are load-bearing for P0.3.8's two conclusions — grayscale drives the recon
false positive, phase-scramble leaves the *genuine* forgetting pole unpopulated — yet neither had a
test. These pin the properties each conclusion rests on.
"""

from __future__ import annotations

import pytest
import torch

from cafl4ds.data.sources import _apply_pixel_transform, _phase_scramble, _to_grayscale


def _imgs(seed: int = 0) -> torch.Tensor:
    """A small batch of ``[N, 3, H, W]`` color images in ``[0, 1]``."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(4, 3, 16, 16, generator=g)


def _shape_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean per-image cosine similarity of the mean-centred flattened tensors (compares *shape*)."""
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean()


def test_grayscale_equalizes_channels_and_keeps_structure() -> None:
    """Grayscale is a cosmetic shift: all 3 channels equal, spatial structure intact, stays in range."""
    g = _to_grayscale(_imgs())
    assert torch.allclose(g[:, 0], g[:, 1]) and torch.allclose(g[:, 1], g[:, 2]), "channels not equal"
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert g[:, 0].var(dim=(1, 2)).mean() > 1e-3, "grayscale collapsed spatial structure"


def test_phase_scramble_is_reproducible_and_seed_dependent() -> None:
    """A fixed seed yields the identical shift (so both arms match); different seeds differ; output real."""
    imgs = _imgs()
    a = _phase_scramble(imgs, seed=0)
    assert torch.equal(a, _phase_scramble(imgs, seed=0)), "not reproducible at a fixed seed"
    assert not torch.equal(a, _phase_scramble(imgs, seed=1)), "seed had no effect"
    assert torch.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0, "output not real / out of range"


def test_phase_scramble_preserves_power_spectrum_and_destroys_structure() -> None:
    """The per-channel amplitude spectrum survives (shape corr high); pixel structure does not."""
    imgs = _imgs()
    scrambled = _phase_scramble(imgs, seed=0)
    amp_in = torch.fft.fft2(imgs).abs()
    amp_out = torch.fft.fft2(scrambled).abs()
    assert _shape_corr(amp_in, amp_out) > 0.9, "amplitude spectrum not preserved"
    assert _shape_corr(imgs, scrambled) < 0.5, "pixel structure not destroyed"


def test_apply_pixel_transform_dispatch_and_unknown() -> None:
    """The dispatcher routes to each transform and rejects an unknown kind."""
    imgs = _imgs()
    assert torch.equal(_apply_pixel_transform(imgs, "grayscale", 0), _to_grayscale(imgs))
    assert torch.equal(_apply_pixel_transform(imgs, "phase_scramble", 0), _phase_scramble(imgs, 0))
    with pytest.raises(ValueError, match="Unknown pixel transform"):
        _apply_pixel_transform(imgs, "sepia", 0)
