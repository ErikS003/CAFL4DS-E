"""Augmentation pipelines for the SSL methods.

Joint-embedding methods (SimSiam here, BYOL/SimCLR later) learn by pulling together two
*augmented views* of the same image, so :class:`TwoView` returns an independent pair. MAE
needs only light spatial augmentation (its learning signal comes from masking), so it uses a
single view.

Simplification for the Phase-0 harness: a transform draws its random parameters **once per
call** and applies them to the whole batch (fast on CPU). Two views therefore differ from
each other (two independent draws) but samples within a view share a crop/jitter. That is
enough to exercise the joint-embedding mechanics; per-sample augmentation can replace this
without touching the method code.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2


def make_ssl_augment(img_size: int, min_scale: float = 0.4, jitter_strength: float = 1.0) -> v2.Transform:
    """Build a standard joint-embedding augmentation pipeline.

    Args:
        img_size: Output image side length (square).
        min_scale: Lower bound of the random-resized-crop area fraction. A *smaller* value crops
            more aggressively (stronger views).
        jitter_strength: Multiplier on the ColorJitter magnitudes (brightness/contrast/saturation
            base 0.4, hue base 0.1). ``1.0`` reproduces the published SimSiam-style jitter; larger
            values are stronger colour distortion. The P0.2.4 heavy-augmentation stressor turns
            both this and a low ``min_scale`` up to make *over-strong* (but still healthy) views.

    Returns:
        A composed transform mapping a float image batch ``[B, C, H, W]`` in ``[0, 1]`` to an
        augmented batch of the same shape.
    """
    bcs = 0.4 * jitter_strength  # brightness / contrast / saturation
    hue = min(0.5, 0.1 * jitter_strength)  # ColorJitter caps hue at 0.5
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(bcs, bcs, bcs, hue)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ]
    )


def make_light_augment(img_size: int, min_scale: float = 0.6) -> v2.Transform:
    """Build a light spatial augmentation pipeline (for MAE).

    Args:
        img_size: Output image side length (square).
        min_scale: Lower bound of the random-resized-crop area fraction.

    Returns:
        A composed transform mapping a float image batch to an augmented batch.
    """
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )


class TwoView(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """Produce two independently augmented views of an image batch."""

    def __init__(self, augment: v2.Transform) -> None:
        """Wrap an augmentation transform to emit a positive-pair.

        Args:
            augment: The per-view augmentation to apply twice (independently).
        """
        super().__init__()
        self.augment = augment

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of ``imgs``.

        Args:
            imgs: Image batch ``[B, C, H, W]``.

        Returns:
            A ``(view_1, view_2)`` pair, each ``[B, C, H, W]``.
        """
        return self.augment(imgs), self.augment(imgs)
