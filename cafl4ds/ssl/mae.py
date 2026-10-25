"""Masked Autoencoder (MAE) SSL method — the default ``C`` backbone (He et al. 2022).

MAE masks a large fraction of image patches, encodes the visible ones, and reconstructs the
missing pixels; the per-patch reconstruction error is a free label-free informativeness
signal (used by novelty filters in later phases). MAE is **collapse-resistant** — a constant
code cannot reconstruct diverse pixels — so its Phase-1 degradation mode is
forgetting/overspecialization, not representational collapse; the collapse demonstration uses
the joint-embedding method (:mod:`cafl4ds.ssl.simsiam`) instead.
"""

from __future__ import annotations

from typing import cast

import torch
from torchvision.transforms import v2

from cafl4ds.models.heads import MAEDecoder
from cafl4ds.models.vit import TinyViTEncoder, patchify
from cafl4ds.ssl.augment import make_light_augment, make_ssl_augment
from cafl4ds.ssl.base import SSLMethod


class MAE(SSLMethod):
    """Masked Autoencoder over the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`."""

    def __init__(
        self,
        encoder: TinyViTEncoder,
        decoder: MAEDecoder,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        augment: v2.Transform | None = None,
    ) -> None:
        """Build the MAE method.

        Args:
            encoder: The shared backbone encoder.
            decoder: The pixel-reconstruction decoder.
            mask_ratio: Fraction of patches masked each step.
            norm_pix_loss: Whether to normalize each target patch (per-patch mean/var) before
                the reconstruction loss, as in the MAE paper (usually more stable).
            augment: Light spatial augmentation applied before masking; defaults to
                :func:`~cafl4ds.ssl.augment.make_light_augment` sized to the encoder.
        """
        super().__init__(encoder)
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        img_size = int(round(encoder.num_patches**0.5)) * encoder.patch_size
        self.augment = augment if augment is not None else make_light_augment(img_size)
        # A heavier, still semantic-preserving augment used ONLY by the P0.5.2 read-only
        # alignment-under-stronger-aug instrument (via make_views_strong) — never in training.
        self.strong_augment = make_ssl_augment(img_size)

    @property
    def name(self) -> str:
        """Return the method identifier."""
        return "mae"

    def _reconstruction_target(self, imgs: torch.Tensor) -> torch.Tensor:
        """Patchify images into reconstruction targets, optionally per-patch normalized.

        Args:
            imgs: Images ``[B, C, H, W]``.

        Returns:
            Target patches ``[B, N, p * p * C]``.
        """
        target = patchify(imgs, self.encoder.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        return target

    def _masked_recon_loss(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment, mask, reconstruct, and return the per-patch loss and the mask.

        Shared by :meth:`training_step` (mean over masked patches) and
        :meth:`per_sample_loss` (per-image mean over its masked patches). Masking is random,
        so each call draws a fresh mask — fine for both a training step and a selection probe.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A ``(loss_per_patch, mask)`` pair, each ``[B, N]``; ``mask`` is 1 on masked
            (reconstructed) patches and 0 on visible ones.
        """
        views = self.augment(imgs)
        latent, mask, ids_restore = self.encoder.forward_encoder(views, mask_ratio=self.mask_ratio)
        assert mask is not None and ids_restore is not None  # noqa: S101 - guaranteed by mask_ratio > 0
        pred = self.decoder(latent, ids_restore)
        target = self._reconstruction_target(views)
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)  # [B, N]
        return loss_per_patch, mask

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the masked reconstruction loss on the masked patches only.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            The mean squared reconstruction error over masked patches (scalar).
        """
        loss_per_patch, mask = self._masked_recon_loss(imgs)
        return cast(torch.Tensor, (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0))

    def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        """Per-image masked reconstruction MSE ``[B]`` (no gradient).

        The free per-frame informativeness signal MAE exposes: high loss = the current model
        reconstructs this frame poorly. Averaged over each image's own masked patches.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A detached ``[B]`` tensor of per-image reconstruction errors.
        """
        with torch.no_grad():
            loss_per_patch, mask = self._masked_recon_loss(imgs)
            per_image = (loss_per_patch * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return per_image

    def make_views(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a positive pair via two independent draws of MAE's own augmentation.

        MAE has no joint-embedding positive pair (it never optimizes augmentation-invariance),
        so the base :class:`~cafl4ds.ssl.base.SSLMethod` returns ``None`` and the alignment
        instrument is skipped. P0.5 puts the collapse suite *on trial* on an MAE backbone (does
        any label-free reader track the frozen-probe crater of a shortcut-degraded rep?), and
        alignment is one candidate reader. So MAE exposes a pair here — two independent passes of
        its light spatial augmentation, the same transform :meth:`_masked_recon_loss` applies
        before masking. This makes ``alignment`` *measurable* on MAE; whether it reads quality is
        exactly the open question (a semantic rep should be more augmentation-invariant than a
        low-level shortcut rep). It is a read-only measurement path — nothing here enters
        :meth:`training_step`, so the MAE objective is unchanged.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A ``(view_a, view_b)`` pair, each an independently augmented view of ``imgs``.
        """
        return self.augment(imgs), self.augment(imgs)

    def make_views_strong(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a positive pair under a *strong* (SSL-grade) augment, for P0.5.2's reader.

        Two independent draws of the heavier :func:`~cafl4ds.ssl.augment.make_ssl_augment`
        pipeline (random-resized-crop + flip + colour jitter + grayscale) — a strong but
        semantic-preserving transform. The *alignment-under-stronger-aug* candidate reader tests
        the invariance a semantic MAE representation should have but a low-level shortcut should
        not. Read-only: this augment never enters :meth:`training_step`.

        Args:
            imgs: A batch of raw images ``[B, C, H, W]``.

        Returns:
            A ``(view_a, view_b)`` pair, each a strongly augmented view of ``imgs``.
        """
        return self.strong_augment(imgs), self.strong_augment(imgs)
