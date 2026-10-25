"""Shape / invariance tests for the tiny ViT encoder and task heads.

No training here — just that the building blocks compose with correct shapes, that
patchify/unpatchify round-trip, and that MAE masking produces a consistent mask / restore.
"""

import torch

from cafl4ds.models.heads import MAEDecoder, MLPHead
from cafl4ds.models.vit import TinyViTEncoder, patchify, unpatchify


def _gen(seed: int = 0) -> torch.Generator:
    """Return a seeded CPU generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_patchify_unpatchify_roundtrip() -> None:
    """Unpatchify inverts patchify exactly."""
    g = _gen()
    imgs = torch.rand(3, 3, 16, 16, generator=g)
    patches = patchify(imgs, patch_size=8)
    assert patches.shape == (3, 4, 8 * 8 * 3)
    recon = unpatchify(patches, patch_size=8, channels=3)
    assert torch.allclose(recon, imgs, atol=1e-6)


def test_encoder_embed_and_token_shapes() -> None:
    """The encoder yields ``[B, 1+N, d]`` tokens and a pooled ``[B, d]`` embedding."""
    enc = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    x = torch.rand(5, 3, 16, 16, generator=_gen())
    tokens = enc(x)
    assert tokens.shape == (5, 1 + enc.num_patches, 32)
    assert enc.embed(x).shape == (5, 32)


def test_encoder_masking_keeps_expected_fraction() -> None:
    """With ``mask_ratio``, the encoder keeps ~``(1-ratio)`` patches and returns a valid mask."""
    enc = TinyViTEncoder(img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    x = torch.rand(4, 3, 32, 32, generator=_gen())
    tokens, mask, ids_restore = enc.forward_encoder(x, mask_ratio=0.75)
    n = enc.num_patches  # 16
    len_keep = max(1, round(n * 0.25))
    assert tokens.shape == (4, 1 + len_keep, 32)  # +1 for cls
    assert mask is not None and ids_restore is not None
    assert mask.shape == (4, n)
    # mask is binary and marks exactly the removed patches.
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.allclose(mask.sum(dim=1), torch.full((4,), float(n - len_keep)))


def test_encoder_rejects_indivisible_img_size() -> None:
    """img_size must be divisible by patch_size."""
    try:
        TinyViTEncoder(img_size=30, patch_size=8)
    except ValueError as e:
        assert "divisible" in str(e)
    else:  # pragma: no cover - the constructor must raise
        raise AssertionError("expected ValueError for indivisible img_size")


def test_mae_decoder_reconstructs_full_patch_grid() -> None:
    """The decoder maps the visible-token latent back to the full patch grid."""
    enc = TinyViTEncoder(img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    dec = MAEDecoder(num_patches=enc.num_patches, encoder_dim=32, patch_size=8, decoder_dim=16, depth=1, num_heads=2)
    x = torch.rand(4, 3, 32, 32, generator=_gen())
    latent, _, ids_restore = enc.forward_encoder(x, mask_ratio=0.5)
    pred = dec(latent, ids_restore)
    assert pred.shape == (4, enc.num_patches, 8 * 8 * 3)


def test_mlp_head_shapes_for_projector_and_predictor() -> None:
    """The projector (3-layer, last BN) and predictor (2-layer) produce the right shapes."""
    projector = MLPHead(32, 64, 16, num_layers=3, last_bn=True)
    predictor = MLPHead(16, 8, 16, num_layers=2, last_bn=False)
    x = torch.rand(6, 32, generator=_gen())
    z = projector(x)
    assert z.shape == (6, 16)
    assert predictor(z).shape == (6, 16)


def test_reset_parameters_from_scratch_wipes_loaded_weights() -> None:
    """reset_parameters_from_scratch re-randomizes *every* learned param, incl. the patch-embed conv.

    The P0.3.7/P0.3.8 savings "scratch" pole depends on this genuinely wiping a loaded checkpoint (it
    is the fix for the ``apply_encoder_init`` no-op bug P0.3.7 caught — audit P0.3 §E, E2). Simulate a
    warm-started checkpoint with a constant fill, then assert the wipe leaves nothing at that value —
    including ``patch_embed`` (the conv ``_init_weights`` alone skips, so a naive reset would miss it).
    """
    enc = TinyViTEncoder(img_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    with torch.no_grad():
        for p in enc.parameters():
            p.fill_(0.123)
    before = {name: p.clone() for name, p in enc.named_parameters()}
    enc.reset_parameters_from_scratch()
    for name, p in enc.named_parameters():
        assert not torch.equal(before[name], p), f"{name} was not re-randomized"
    assert "patch_embed.weight" in before  # the conv is present and (per the loop above) was wiped


def test_reset_parameters_from_scratch_preserves_buffers() -> None:
    """Registered buffers (e.g. a subclass's ImageNet-norm constants) survive the wipe untouched."""
    enc = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    enc.register_buffer("_norm_const", torch.tensor([1.0, 2.0, 3.0]))
    enc.reset_parameters_from_scratch()
    assert torch.equal(enc._norm_const, torch.tensor([1.0, 2.0, 3.0]))
