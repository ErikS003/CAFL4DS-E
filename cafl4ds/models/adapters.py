"""Parameter-efficient residual adaptation for a frozen TinyViT backbone."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder


class ResidualAdapterEncoder(nn.Module):  # type: ignore[misc]
    """Wrap a TinyViT with a zero-initialized bottleneck residual on its pooled embedding."""

    def __init__(self, backbone: TinyViTEncoder, bottleneck_dim: int = 16) -> None:
        """Create the residual adapter and freeze the wrapped backbone."""
        super().__init__()
        if bottleneck_dim < 1:
            raise ValueError(f"bottleneck_dim must be positive; got {bottleneck_dim}.")
        self.backbone = backbone
        self.adapter = nn.Sequential(
            nn.Linear(backbone.embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, backbone.embed_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @property
    def embed_dim(self) -> int:
        """Return the unchanged embedding width."""
        return self.backbone.embed_dim

    @property
    def num_patches(self) -> int:
        """Return the wrapped transformer's patch count."""
        return self.backbone.num_patches

    @property
    def patch_size(self) -> int:
        """Return the wrapped transformer's patch size."""
        return self.backbone.patch_size

    @property
    def in_chans(self) -> int:
        """Return the wrapped transformer's input channel count."""
        return self.backbone.in_chans

    @property
    def grid_size(self) -> int:
        """Return the wrapped transformer's patch-grid side length."""
        return self.backbone.grid_size

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return the stable backbone feature plus its learned residual."""
        base = self.backbone.embed(imgs)
        return base + self.adapter(base)

    def attention_maps(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        """Expose frozen-backbone attention maps for optional monitoring."""
        return cast(list[torch.Tensor], self.backbone.attention_maps(imgs))
