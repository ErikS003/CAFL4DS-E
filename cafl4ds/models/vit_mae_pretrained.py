"""An MAE-pretrained ViT-B/16 backbone for the label-free forgetting fire (P0.3.6).

P0.3.4's fire is *supervised* (ImageNet-supervised weights + a supervised fine-tune); the
calibration cafl4ds actually needs is **instrument + MAE, label-free**. This encoder supplies
the deep phase-A well for that: the official MAE-pretrained ViT-B/16 (He et al. 2022) plugged in
as the backbone of the harness's MAE method, so continued **masked-reconstruction** training on a
new task can respecialize — and (the question P0.3.6 asks) *erode* — a genuinely rich SSL
representation, with no labels in the learning signal.

It subclasses :class:`~cafl4ds.models.vit.TinyViTEncoder` so the MAE method reads the exact
surface it expects (``forward_encoder(mask_ratio)``, ``num_patches``, ``patch_size``,
``embed_dim``, ``embed``). Built at ViT-B dims (224px / patch 16 / 768-d / depth 12 / 12 heads)
the official checkpoint loads **strict-clean** after a single key remap — its
``patch_embed.proj.*`` maps onto our bare-``Conv2d`` ``patch_embed.*`` (the only structural
difference; the timm-style block/qkv naming matches). The checkpoint is **encoder-only** (no
decoder / mask token), so the harness's :func:`~cafl4ds.ssl.factory.build_mae` supplies a fresh
decoder — warm it with a short phase A (``pretrained_encoder=true`` keeps these weights while
still training phase A) rather than ``skip_phase_a``.

Inputs are the harness's ``[0, 1]`` images at the source resolution; ``_embed_patches`` resizes
them to 224 and ImageNet-normalizes internally (matching the MAE pretraining distribution and the
host-validated ~0.80 CIFAR-100 task-A linear probe), so the rest of the harness stays
resolution-agnostic. Weights load offline from ``state_dict_path`` (a host-fetched checkpoint
bind-mounted into the Gaudi container).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from loguru import logger

from cafl4ds.models.vit import TinyViTEncoder

# ImageNet channel statistics the MAE ViT-B/16 was pretrained under.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MAEPretrainedViTEncoder(TinyViTEncoder):
    """MAE-pretrained ViT-B/16, exposed through the ``TinyViTEncoder`` MAE surface.

    Identical architecture to :class:`~cafl4ds.models.vit.TinyViTEncoder` at ViT-B dims, with the
    official MAE encoder weights loaded and an internal resize + ImageNet-normalize so the harness
    can keep feeding native-resolution ``[0, 1]`` images.
    """

    def __init__(
        self,
        state_dict_path: str,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        imagenet_norm: bool = True,
        norm_eps: float | None = None,
    ) -> None:
        """Build the ViT-B encoder and load the MAE-pretrained weights.

        Args:
            state_dict_path: Path to the official MAE ViT-B ``.pth`` (``{"model": state_dict}``),
                loaded offline. Built at ViT-B dims it loads strict-clean after the
                ``patch_embed.proj`` remap.
            img_size: The resolution the encoder operates at (224 for the ViT-B checkpoint — sets
                ``num_patches`` and the ``pos_embed`` length; inputs are resized to this).
            patch_size: Patch side (16).
            embed_dim: Token width (768).
            depth: Number of transformer blocks (12).
            num_heads: Attention heads per block (12).
            mlp_ratio: MLP hidden-width multiple (4.0).
            imagenet_norm: Whether to ImageNet-normalize inputs (matches the pretraining
                distribution; the host probe validated the well at 0.80 with normalization on).
            norm_eps: Optional explicit LayerNorm epsilon. None preserves the historical
                wrapper behavior; Stage B requests the official MAE value of 1e-6.

        Raises:
            FileNotFoundError: If ``state_dict_path`` does not exist.
            ValueError: If the checkpoint does not map cleanly onto the encoder parameters.
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self._input_size = img_size
        self._imagenet_norm = imagenet_norm
        if norm_eps is not None:
            for layer in self.modules():
                if isinstance(layer, torch.nn.LayerNorm):
                    layer.eps = norm_eps
        self._load_mae_weights(state_dict_path)  # load BEFORE registering norm buffers -> clean strict match
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _load_mae_weights(self, state_dict_path: str) -> None:
        """Load the official MAE encoder weights, remapping ``patch_embed.proj.* -> patch_embed.*``."""
        path = Path(state_dict_path)
        if not path.exists():
            raise FileNotFoundError(f"MAE ViT-B state_dict not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        remapped = {k.replace("patch_embed.proj.", "patch_embed."): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise ValueError(f"MAE ViT-B load mismatch: missing={list(missing)} unexpected={list(unexpected)}")
        logger.info(f"loaded MAE-pretrained ViT-B/16 encoder from {path}")

    def _embed_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """Resize to the ViT-B input size + ImageNet-normalize, then embed patches (base impl).

        Overrides the base so every path (``forward_encoder`` for masked training *and* ``embed``
        for the probes/drift) sees the pretraining input contract — the harness stays free to feed
        native-resolution ``[0, 1]`` images.
        """
        imgs = imgs.to(self._mean.device)
        if imgs.shape[-1] != self._input_size or imgs.shape[-2] != self._input_size:
            imgs = F.interpolate(imgs, size=self._input_size, mode="bilinear", align_corners=False)
        if self._imagenet_norm:
            imgs = (imgs - self._mean) / self._std
        return super()._embed_patches(imgs)
