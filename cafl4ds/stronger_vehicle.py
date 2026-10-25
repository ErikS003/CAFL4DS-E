"""A bounded, pretrained ViT-B MAE vehicle with explicit preprocessing contracts."""

# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torchvision.transforms import v2

from cafl4ds.models.vit_mae_pretrained import MAEPretrainedViTEncoder
from cafl4ds.ssl.factory import build_mae
from cafl4ds.ssl.mae import MAE

CHECKPOINT_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


class QualifiedInputMAE(MAE):
    """Reconstruct normalized input patches while accepting raw RGB input."""

    def _reconstruction_target(self, imgs: torch.Tensor) -> torch.Tensor:
        """Match the target to the encoder's ImageNet-normalized view exactly once."""
        encoder = self.encoder
        normalized = (imgs - cast(torch.Tensor, encoder._mean)) / cast(torch.Tensor, encoder._std)
        return super()._reconstruction_target(normalized)


def vehicle(checkpoint: Path) -> QualifiedInputMAE:
    """Build the single declared vehicle, preserving historical wrapper defaults."""
    encoder = MAEPretrainedViTEncoder(str(checkpoint), norm_eps=1e-6)
    encoder.pos_embed.requires_grad_(False)
    base = build_mae(encoder, decoder_dim=256, decoder_depth=4, decoder_heads=8, decoder_mlp_ratio=4)
    with torch.no_grad():
        base.decoder.decoder_pos_embed.copy_(position_codes(256, 14))
        torch.nn.init.normal_(base.decoder.mask_token, std=0.02)
        for layer in base.decoder.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    base.decoder.decoder_pos_embed.requires_grad_(False)
    # This is a fresh decoder, not claimed to be the pretrained paper decoder.
    for layer in base.decoder.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.eps = 1e-6
    return QualifiedInputMAE(encoder, base.decoder, mask_ratio=0.75, norm_pix_loss=True)


def position_codes(width: int, grid: int) -> torch.Tensor:
    """Fixed two-dimensional sine/cosine decoder positions, plus a zero CLS position."""
    if width % 4:
        raise ValueError("Positional width must be divisible by four")
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    frequency = 10000 ** (-torch.arange(width // 4, dtype=torch.float64) / (width // 4))
    components = []
    for axis in (x, y):
        phase = axis.flatten().double()[:, None] * frequency[None]
        components.extend([phase.sin(), phase.cos()])
    patches = torch.cat(components, dim=1).float()
    return torch.cat([torch.zeros(1, width), patches]).unsqueeze(0)


def read_images(paths: list[str], device: str = "cpu") -> torch.Tensor:
    """Decode at native task resolution without the historical 64-pixel bottleneck."""
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224), antialias=True)])
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images).to(device)


def ssl_update(
    method: MAE, optimizer: torch.optim.Optimizer, paths: list[str], seed: int, device: str = "cpu", microbatch: int = 4
) -> float:
    """Accumulate a mean loss over exactly 32 exposures, then perform one update."""
    if len(paths) != 32 or 32 % microbatch:
        raise ValueError("Require effective batch32 and a dividing microbatch")
    method.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    devices = [torch.device(device)] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, 32, microbatch):
            loss = method.training_step(read_images(paths[start : start + microbatch], device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            (loss * microbatch / 32).backward()
            total += float(loss.detach()) * microbatch / 32
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in method.parameters()):
        raise ValueError("Nonfinite gradients")
    optimizer.step()
    return total


def embed(method: MAE, paths: list[str], device: str = "cpu", batch: int = 8) -> torch.Tensor:
    """Read pooled patch features without changing the optimizer RNG or model mode."""
    previous = method.training
    method.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            result.append(method.encoder.embed(read_images(paths[start : start + batch], device)).cpu())
    method.train(previous)
    return torch.cat(result)
