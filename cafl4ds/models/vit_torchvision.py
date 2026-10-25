"""An ImageNet-pretrained ViT-B/16 backbone, exposed through the encoder ``embed`` surface.

Phase 0's forgetting calibration (P0.3.4) needs a *genuinely rich* phase-A representation so
that continued narrow training has something large to destroy. Training the hand-rolled
:class:`~cafl4ds.models.vit.TinyViTEncoder` from scratch never clears that bar — on
Split-CIFAR-100 the from-scratch supervised task-A probe tops out near 0.58 and the
forgetting contrast stays borderline (P0.3.4). A backbone pretrained on a large diverse
corpus does the "phase A" for us: a frozen ImageNet ViT-B/16 linear-probes ~0.94 on
CIFAR-100 task-A classes, leaving a deep well for the positive control to crater.

This wrapper adapts ``torchvision.models.vit_b_16`` to the two things the forgetting harness
asks of an encoder: an ``embed_dim`` attribute and a grad-enabled ``embed(imgs) -> [B, d]``
returning the pooled (class-token) representation. Inputs are the harness's native
``[0, 1]`` images at ``img_size``; this class resizes them to the 224 the ViT expects and
applies ImageNet normalization internally, so the rest of the harness stays resolution-
agnostic. Weights are loaded from a local ``state_dict_path`` (a checkpoint pre-fetched on the
host) so the run needs no network — the Gaudi container sees the file via a read-only bind
mount.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
import torchvision.models as tvm
from loguru import logger
from torch import nn

# ImageNet channel statistics the torchvision ViT-B/16 weights were trained under.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_VIT_INPUT = 224  # the resolution vit_b_16 was trained at (patch 16 -> 14x14 tokens)
_VIT_PATCH = 16  # vit_b_16 patch size (14x14 = 196 tokens over the 224 input)


class TorchvisionViTEncoder(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A pretrained torchvision ViT-B/16 as a pooled-embedding backbone for the harness.

    Exposes the surface the harness reads: ``embed_dim`` and a grad-enabled ``embed`` returning the
    class-token representation, plus the ``patch_size`` / ``num_patches`` / ``in_chans`` token-grid
    contract the SSL factory and MAE decoder size against (so a JE method — SimSiam / Barlow, the P0.6
    vehicle — can be built on this pretrained backbone). The classification head is discarded (replaced
    by identity) so ``embed`` yields the 768-d pre-head features; the whole backbone is trainable, so a
    phase B fine-tunes (and respecializes) it.
    """

    def __init__(self, state_dict_path: str | None = None, weights: str | None = None) -> None:
        """Build the ViT-B/16 and load pretrained weights.

        Args:
            state_dict_path: Path to a local ``vit_b_16`` ``state_dict`` (``.pth``) to load
                offline (preferred in the Gaudi container). Takes precedence over ``weights``.
            weights: Fallback torchvision weights enum name (e.g. ``"IMAGENET1K_V1"``) fetched
                via torchvision's download path — used only when ``state_dict_path`` is ``None``.

        Raises:
            FileNotFoundError: If ``state_dict_path`` is given but does not exist.
        """
        super().__init__()
        if state_dict_path is not None:
            path = Path(state_dict_path)
            if not path.exists():
                raise FileNotFoundError(f"pretrained ViT-B/16 state_dict not found: {path}")
            net = tvm.vit_b_16(weights=None)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            logger.info(f"loaded pretrained ViT-B/16 weights from {path}")
        else:
            net = tvm.vit_b_16(weights=weights)
            logger.info(f"built ViT-B/16 with torchvision weights={weights}")
        self.embed_dim: int = int(net.hidden_dim)  # 768
        # The 224/16 token-grid contract, matching the cafl4ds ViT's own attributes so the SSL factory
        # (`_encoder_img_size` → the SimSiam/Barlow augmentation crop) and the MAE decoder can size
        # against a torchvision backbone too. Inputs at any size are resized to 224 in `_prep`, so the
        # grid is fixed (P0.6 reads the JE projector off this backbone).
        self.patch_size: int = _VIT_PATCH
        self.in_chans: int = 3
        self.num_patches: int = (_VIT_INPUT // _VIT_PATCH) ** 2  # 196
        net.heads = nn.Identity()  # embed() returns the pooled class-token features
        self.vit = net
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, imgs: torch.Tensor) -> torch.Tensor:
        """Resize ``[0, 1]`` images to 224 and ImageNet-normalize (the ViT's input contract)."""
        x = F.interpolate(imgs, size=_VIT_INPUT, mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pooled class-token embedding ``[B, embed_dim]`` (gradient-enabled).

        Args:
            imgs: A batch of ``[0, 1]`` images ``[B, 3, H, W]`` at the harness's ``img_size``.

        Returns:
            The 768-d class-token representation ``[B, 768]``.
        """
        # Probe/measurement callers hold their fixed eval tensors on CPU; coerce to the backbone's
        # device (a no-op when both are CPU) so ``embed`` stays usable after ``.to(hpu)``, matching
        # TinyViTEncoder.embed. The output is moved back to CPU by the measurements' _as_tensor.
        imgs = imgs.to(self._mean.device)
        return self.vit(self._prep(imgs))
