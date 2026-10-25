"""Task heads attached to the shared :class:`~cafl4ds.models.vit.TinyViTEncoder`.

These are *training* heads, discarded at measurement time (the health instruments read the
encoder's pooled representation, never these outputs):

* :class:`MLPHead` — a configurable BN-MLP; used as SimSiam's projector and predictor
  (BYOL later reuses the same shape).
* :class:`MAEDecoder` — a lightweight transformer decoder that reconstructs pixels from the
  encoder's visible-token latent (He et al. 2022).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from cafl4ds.models.vit import Block


class MLPHead(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A batch-norm MLP head (SimSiam/BYOL projector or predictor).

    With ``num_layers=3`` and ``last_bn=True`` this is the SimSiam projector; with
    ``num_layers=2`` and ``last_bn=False`` it is the SimSiam predictor (Chen & He 2021).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        last_bn: bool = True,
    ) -> None:
        """Build the MLP head.

        Args:
            in_dim: Input dimensionality.
            hidden_dim: Hidden-layer width (used for all but the output layer).
            out_dim: Output dimensionality.
            num_layers: Total number of linear layers (``>= 1``).
            last_bn: Whether to apply a (bias-free) BatchNorm after the output layer, with
                no final activation — the SimSiam projector does, the predictor does not.

        Raises:
            ValueError: If ``num_layers < 1``.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim, bias=not last_bn))
        if last_bn:
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP head (``[B, in_dim] -> [B, out_dim]``)."""
        return cast(torch.Tensor, self.net(x))


class MAEDecoder(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A lightweight MAE decoder: reconstruct pixels from the visible-token latent.

    Following He et al. (2022): project the encoder latent into the decoder width, scatter
    the visible tokens back into the full sequence (filling the holes with a shared learned
    mask token), add decoder positional embeddings, run a few transformer blocks, and
    linearly predict the per-patch pixels.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int,
        patch_size: int,
        in_chans: int = 3,
        decoder_dim: int = 64,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        """Build the decoder.

        Args:
            num_patches: Number of patches ``N`` in the full (unmasked) sequence.
            encoder_dim: Dimensionality of the encoder latent tokens.
            patch_size: Side length ``p`` of each square patch (sets the pixel output width).
            in_chans: Number of image channels ``C``.
            decoder_dim: Decoder token dimensionality (smaller than the encoder's).
            depth: Number of decoder transformer blocks.
            num_heads: Number of attention heads per decoder block.
            mlp_ratio: Decoder MLP hidden width as a multiple of ``decoder_dim``.
        """
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, decoder_dim))
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Reconstruct per-patch pixels from the encoder's visible-token latent.

        Args:
            latent: Encoder output ``[B, 1 + N_keep, encoder_dim]`` (``cls`` token first).
            ids_restore: The ``[B, N]`` permutation from
                :meth:`~cafl4ds.models.vit.TinyViTEncoder.random_masking` that undoes the
                visible-patch shuffle.

        Returns:
            Predicted patches ``[B, N, p * p * C]`` in row-major patch order (``cls`` token
            dropped).
        """
        x = self.decoder_embed(latent)
        b, n = ids_restore.shape
        n_keep = x.shape[1] - 1  # exclude cls
        mask_tokens = self.mask_token.expand(b, n - n_keep, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # drop cls, append mask tokens
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # restore cls
        x = x + self.decoder_pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return cast(torch.Tensor, self.pred(x[:, 1:, :]))  # drop cls; [B, N, p*p*C]
