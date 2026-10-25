"""Protect pretrained input, target and historical normalization contracts."""

from pathlib import Path

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder, patchify
from cafl4ds.models.vit_mae_pretrained import MAEPretrainedViTEncoder
from cafl4ds.ssl.factory import build_mae
from cafl4ds.stronger_vehicle import QualifiedInputMAE, position_codes


def pretrained_fixture(path: Path) -> Path:
    """Write a small official-key checkpoint without downloading test weights."""
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2, mlp_ratio=4)
    state = {k.replace("patch_embed.", "patch_embed.proj."): v for k, v in encoder.state_dict().items()}
    torch.save({"model": state}, path)
    return path


def test_norm_epsilon_opt_in_and_strict_checkpoint(tmp_path: Path) -> None:
    """Stage B may change epsilon explicitly without changing historical defaults."""
    path = pretrained_fixture(tmp_path / "weights.pt")
    params = {
        "state_dict_path": str(path),
        "img_size": 16,
        "patch_size": 8,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 2,
    }
    historical = MAEPretrainedViTEncoder(**params)
    candidate = MAEPretrainedViTEncoder(**params, norm_eps=1e-6)
    assert historical.blocks[0].norm1.eps == 1e-5
    assert candidate.blocks[0].norm1.eps == 1e-6
    for key, value in historical.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    state = torch.load(path, weights_only=True)
    del state["model"]["norm.weight"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="load mismatch"):
        MAEPretrainedViTEncoder(**params)


def test_reconstruction_normalizes_channels_once(tmp_path: Path) -> None:
    """The target must follow normalized pretrained input, not raw/double-normalized RGB."""
    path = pretrained_fixture(tmp_path / "weights.pt")
    encoder = MAEPretrainedViTEncoder(str(path), img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    base = build_mae(encoder, decoder_dim=16, decoder_depth=1, decoder_heads=2)
    method = QualifiedInputMAE(encoder, base.decoder, norm_pix_loss=False)
    images = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(18))
    expected = patchify((images - encoder._mean) / encoder._std, 8)
    actual = method._reconstruction_target(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, patchify(images, 8))


def test_fixed_decoder_positions_are_spatial() -> None:
    """Two-dimensional positions distinguish neighboring horizontal and vertical patches."""
    codes = position_codes(8, 2)[0]
    assert not codes[0].any()
    torch.testing.assert_close(codes[1, 4:], codes[2, 4:])
    torch.testing.assert_close(codes[1, :4], codes[3, :4])
    assert not torch.equal(codes[1, :4], codes[2, :4])
    assert not torch.equal(codes[1, 4:], codes[3, 4:])
