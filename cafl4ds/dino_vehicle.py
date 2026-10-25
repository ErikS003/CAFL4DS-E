"""Pretrained DINO continuation with batch-correct centering and teacher updates."""

# mypy: disable-error-code=no-untyped-call
from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import InterpolationMode, v2

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.stronger_vehicle import read_images

CHECKPOINT_SHA256 = "9aba1fa8e6875e8413b413e8b8d7b843413280b33b972bd313d2ace8c4129943"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth"
)


class DINOEncoder(TinyViTEncoder):
    """ViT-S/16 with normalized CLS readout and official local-crop positions."""

    _mean: torch.Tensor
    _std: torch.Tensor

    def __init__(self) -> None:
        """Set architecture and epsilon without altering historical encoders."""
        super().__init__(224, 16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4)
        for layer in self.modules():
            if isinstance(layer, torch.nn.LayerNorm):
                layer.eps = 1e-6
        self.register_buffer("_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize raw RGB once, embed variable-resolution patches, then transform."""
        patches = self.patch_embed((images - self._mean) / self._std).flatten(2).transpose(1, 2)
        side = images.shape[-1] // 16
        positions: torch.Tensor = self.pos_embed
        if side != 14:
            spatial = positions[:, 1:].reshape(1, 14, 14, 384).permute(0, 3, 1, 2)
            spatial = F.interpolate(
                spatial, scale_factor=((side + 0.1) / 14, (side + 0.1) / 14), mode="bicubic", align_corners=False
            )
            positions = torch.cat([positions[:, :1], spatial.flatten(2).transpose(1, 2)], dim=1)
        x = torch.cat([self.cls_token.expand(len(images), -1, -1), patches], dim=1) + positions
        for block in self.blocks:
            x = block(x)
        return cast(torch.Tensor, self.norm(x))

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        """Primary readout: normalized student CLS, fixed before outcomes."""
        return self.tokens(imgs)[:, 0]


class ScaledDirection(torch.nn.Module):
    """Explicit row weight normalization matching the published head coefficients."""

    def __init__(self, outputs: int = 65536) -> None:
        """Allocate gain and direction with checkpoint-compatible names."""
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.ones(outputs, 1))
        self.weight_v = torch.nn.Parameter(torch.empty(outputs, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each output row, multiply by its learned gain, then project."""
        return F.linear(x, self.weight_g * F.normalize(self.weight_v, dim=1))


class DINOHead(torch.nn.Module):
    """Three-layer trained projection and normalized prototype head, without BN."""

    def __init__(self) -> None:
        """Use the checkpoint's complete head geometry."""
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(384, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 256),
        )
        self.last_layer = ScaledDirection()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the bottleneck before prototype logits."""
        return cast(torch.Tensor, self.last_layer(F.normalize(self.mlp(x), dim=-1)))


class DINOBranch(torch.nn.Module):
    """Encoder and pretrained projection head."""

    def __init__(self) -> None:
        """Build the architecture before strict checkpoint loading."""
        super().__init__()
        self.backbone = DINOEncoder()
        self.head = DINOHead()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the pretrained prototype logits."""
        return cast(torch.Tensor, self.head(self.backbone.embed(images)))


class DINOVehicle(torch.nn.Module):
    """Student and stop-gradient EMA teacher initialized from the published teacher."""

    center: torch.Tensor

    def __init__(self, checkpoint: Path) -> None:
        """Load full trained heads; initialize only the missing centering buffer."""
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.student = DINOBranch()
        self.teacher = DINOBranch()
        values = {k.replace("patch_embed.proj.", "patch_embed."): v for k, v in state["teacher"].items()}
        for branch in (self.student, self.teacher):
            buffers = {k: v for k, v in branch.state_dict().items() if k in ("backbone._mean", "backbone._std")}
            branch.load_state_dict({**values, **buffers}, strict=True)
        self.teacher.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, 65536))

    @property
    def encoder(self) -> DINOEncoder:
        """Expose the student backbone to the shared readout harness."""
        return self.student.backbone


class BoundedRGB(torch.nn.Module):
    """Preserve PIL-like RGB bounds after bicubic interpolation on float tensors."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Prevent bicubic undershoot from invalidating hue conversion."""
        return image.clamp(0, 1)


def views(images: torch.Tensor) -> list[torch.Tensor]:
    """Two224px global crops and two96px local crops; augment each image independently."""
    color = [
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
    ]
    transforms = [
        v2.Compose(
            [
                v2.RandomResizedCrop(224, scale=(0.25, 1), interpolation=InterpolationMode.BICUBIC),
                BoundedRGB(),
                *color,
                v2.GaussianBlur(9, sigma=(0.1, 2)),
            ]
        ),
        v2.Compose(
            [
                v2.RandomResizedCrop(224, scale=(0.25, 1), interpolation=InterpolationMode.BICUBIC),
                BoundedRGB(),
                *color,
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2))], p=0.1),
                v2.RandomSolarize(0.5, p=0.2),
            ]
        ),
        *[
            v2.Compose(
                [
                    v2.RandomResizedCrop(96, scale=(0.05, 0.25), interpolation=InterpolationMode.BICUBIC),
                    BoundedRGB(),
                    *color,
                    v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2))], p=0.5),
                ]
            )
            for _ in range(2)
        ],
    ]
    result = [torch.stack([transform(image) for image in images]).clamp(0, 1) for transform in transforms]
    if any(not torch.isfinite(view).all() for view in result):
        raise ValueError("Nonfinite augmented RGB view")
    return result


def distillation(student: list[torch.Tensor], teacher: list[torch.Tensor], center: torch.Tensor) -> torch.Tensor:
    """Cross-view CE with fixed temperatures; targets are detached and same views excluded."""
    targets = [F.softmax((t.detach() - center) / 0.07, dim=-1) for t in teacher]
    terms = [
        -(target * F.log_softmax(logits / 0.1, dim=-1)).sum(-1).mean()
        for i, target in enumerate(targets)
        for j, logits in enumerate(student)
        if i != j
    ]
    return torch.stack(terms).mean()


def update(
    method: DINOVehicle, optimizer: torch.optim.Optimizer, paths: list[str], seed: int, momentum: float
) -> float:
    """One effective batch32 update, one batch-center update, one EMA update."""
    if len(paths) != 32:
        raise ValueError("Require32 exposures")
    method.train()
    method.teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    center = method.center.detach().clone()
    if not torch.isfinite(center).all():
        raise ValueError("Nonfinite loss center before SSL update")
    teacher_sum = torch.zeros_like(center)
    total = 0.0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0, 32, 4):
            crops = views(read_images(paths[start : start + 4]))
            with torch.no_grad():
                target = [method.teacher(crop) for crop in crops[:2]]
                teacher_sum += torch.cat(target).sum(0, keepdim=True)
            prediction = [method.student(crop) for crop in crops]
            loss = distillation(prediction, target, center)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite DINO loss")
            (loss / 8).backward()
            total += float(loss.detach()) / 8
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in method.student.parameters()):
        raise ValueError("Nonfinite DINO gradients")
    torch.nn.utils.clip_grad_norm_(method.student.parameters(), 3.0)
    optimizer.step()
    with torch.no_grad():
        method.center.mul_(0.9).add_(teacher_sum / 64, alpha=0.1)
        for ema_parameter, online in zip(method.teacher.parameters(), method.student.parameters(), strict=True):
            ema_parameter.mul_(momentum).add_(online, alpha=1 - momentum)
    return total


def optimizer_for(method: DINOVehicle, lr: float) -> torch.optim.AdamW:
    """AdamW with no decay on biases, normalization scales or prototype gains."""
    decay: list[torch.nn.Parameter] = []
    regular: list[torch.nn.Parameter] = []
    for name, parameter in method.student.named_parameters():
        (regular if parameter.ndim == 1 or name.endswith(("bias", "weight_g")) else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.04}, {"params": regular, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999)
    )


def calibrate_center(method: DINOVehicle, paths: list[str], seed: int = 323031) -> None:
    """Mean teacher logits on both global views of fitting images, without optimization."""
    total = torch.zeros_like(method.center)
    method.teacher.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0, len(paths), 4):
            crops = views(read_images(paths[start : start + 4]))
            total += torch.cat([method.teacher(crop) for crop in crops[:2]]).sum(0, keepdim=True)
        method.center.copy_(total / (2 * len(paths)))
        if not torch.isfinite(method.center).all():
            raise ValueError("Nonfinite fitting-only center calibration")


def embed(method: DINOVehicle, paths: list[str]) -> torch.Tensor:
    """Extract student CLS features in eval mode without optimizer work."""
    method.encoder.eval()
    with torch.no_grad():
        return torch.cat([method.encoder.embed(read_images(paths[i : i + 8])).cpu() for i in range(0, len(paths), 8)])
