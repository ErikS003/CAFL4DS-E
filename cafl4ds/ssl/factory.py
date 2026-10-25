"""Factories that build an :class:`~cafl4ds.ssl.base.SSLMethod` from an encoder + scalars.

The SSL methods take *pre-built* heads (decoder / projector / predictor) so they stay
decoupled from any particular sizing. These factories bridge the gap for Hydra: a config
supplies plain scalar hyper-parameters, the run script supplies the shared ``encoder``, and
the factory sizes the heads from the encoder's own dimensions. This keeps the ``ssl`` config
group a flat list of numbers while the head shapes stay consistent with the backbone.
"""

from __future__ import annotations

from cafl4ds.models.heads import MAEDecoder, MLPHead
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.augment import make_ssl_augment
from cafl4ds.ssl.barlow import BarlowTwins
from cafl4ds.ssl.mae import MAE
from cafl4ds.ssl.simsiam import SimSiam


def _encoder_img_size(encoder: TinyViTEncoder) -> int:
    """Reconstruct the square input side length the encoder was built for."""
    return int(round(encoder.num_patches**0.5)) * encoder.patch_size


def build_mae(
    encoder: TinyViTEncoder,
    mask_ratio: float = 0.75,
    decoder_dim: int = 64,
    decoder_depth: int = 2,
    decoder_heads: int = 4,
    decoder_mlp_ratio: float = 2.0,
    norm_pix_loss: bool = True,
) -> MAE:
    """Build an :class:`~cafl4ds.ssl.mae.MAE` with a decoder sized to the encoder.

    Args:
        encoder: The shared backbone encoder.
        mask_ratio: Fraction of patches masked each step.
        decoder_dim: Decoder token width.
        decoder_depth: Number of decoder transformer blocks.
        decoder_heads: Attention heads per decoder block (must divide ``decoder_dim``).
        decoder_mlp_ratio: Decoder MLP hidden width as a multiple of ``decoder_dim``.
        norm_pix_loss: Whether to per-patch normalize the reconstruction targets.

    Returns:
        The assembled MAE method.
    """
    decoder = MAEDecoder(
        num_patches=encoder.num_patches,
        encoder_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        in_chans=encoder.in_chans,
        decoder_dim=decoder_dim,
        depth=decoder_depth,
        num_heads=decoder_heads,
        mlp_ratio=decoder_mlp_ratio,
    )
    return MAE(encoder, decoder, mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss)


def build_simsiam(
    encoder: TinyViTEncoder,
    proj_hidden: int = 256,
    proj_dim: int = 128,
    pred_hidden: int = 64,
    anti_collapse: bool = True,
    collapse_alpha: float = 0.0,
    stopgrad_beta: float = 1.0,
    aug_min_scale: float = 0.4,
    aug_jitter_strength: float = 1.0,
) -> SimSiam:
    """Build a :class:`~cafl4ds.ssl.simsiam.SimSiam` with heads sized to the encoder.

    Args:
        encoder: The shared backbone encoder.
        proj_hidden: Hidden width of the 3-layer projector.
        proj_dim: Output width of the projector (and the predictor's input/output).
        pred_hidden: Hidden width of the 2-layer predictor bottleneck.
        anti_collapse: Keep SimSiam's predictor + stop-gradient (``True``, the healthy
            control) or disable both for the forced-collapse positive control (``False``).
        collapse_alpha: Partial-collapse loss-blend knob (P0.2.5); ``0.0`` = healthy, ``1.0`` =
            forced-collapse objective. See :class:`~cafl4ds.ssl.simsiam.SimSiam`.
        stopgrad_beta: Partial-collapse soft-stop-gradient knob (P0.2.5); ``1.0`` = full published
            stop-gradient (healthy), ``0.0`` = none. See :class:`~cafl4ds.ssl.simsiam.SimSiam`.
        aug_min_scale: Random-resized-crop lower area bound; the defaults reproduce the certified
            P0.2.x augmentation, and the P0.2.4 heavy-augmentation stressor lowers this.
        aug_jitter_strength: ColorJitter magnitude multiplier (``1.0`` = published strength); the
            P0.2.4 heavy-augmentation stressor raises this.

    Returns:
        The assembled SimSiam method.
    """
    projector = MLPHead(encoder.embed_dim, proj_hidden, proj_dim, num_layers=3, last_bn=True)
    predictor = MLPHead(proj_dim, pred_hidden, proj_dim, num_layers=2, last_bn=False)
    augment = make_ssl_augment(_encoder_img_size(encoder), min_scale=aug_min_scale, jitter_strength=aug_jitter_strength)
    return SimSiam(
        encoder,
        projector,
        predictor,
        augment=augment,
        anti_collapse=anti_collapse,
        collapse_alpha=collapse_alpha,
        stopgrad_beta=stopgrad_beta,
    )


def build_barlow(
    encoder: TinyViTEncoder,
    proj_hidden: int = 256,
    proj_dim: int = 128,
    lambd: float = 5e-3,
    anti_collapse: bool = True,
    aug_min_scale: float = 0.4,
    aug_jitter_strength: float = 1.0,
) -> BarlowTwins:
    """Build a :class:`~cafl4ds.ssl.barlow.BarlowTwins` with a projector sized to the encoder.

    Args:
        encoder: The shared backbone encoder.
        proj_hidden: Hidden width of the 3-layer projector.
        proj_dim: Output width of the projector — the dimension the cross-correlation matrix is
            computed over. Its terminal ``BatchNorm(affine=False)`` is Barlow's per-feature
            standardization.
        lambd: Weight of the redundancy-reduction (off-diagonal) term; ignored when
            ``anti_collapse=False``.
        anti_collapse: Keep Barlow's redundancy-reduction term (``True``, the healthy control)
            or drop it for the forced redundancy-collapse positive control (``False``, P0.2.3).
        aug_min_scale: Random-resized-crop lower area bound (defaults reproduce the certified
            P0.2.x augmentation).
        aug_jitter_strength: ColorJitter magnitude multiplier (``1.0`` = published strength).

    Returns:
        The assembled Barlow Twins method.
    """
    projector = MLPHead(encoder.embed_dim, proj_hidden, proj_dim, num_layers=3, last_bn=True)
    augment = make_ssl_augment(_encoder_img_size(encoder), min_scale=aug_min_scale, jitter_strength=aug_jitter_strength)
    return BarlowTwins(encoder, projector, augment=augment, lambd=lambd, anti_collapse=anti_collapse)
