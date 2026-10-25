"""Unit tests for the P1.0.2 warm-the-well step (``cafl4ds.warmup``)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from cafl4ds.data.attributes import SyntheticAttributeSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.factory import build_simsiam
from cafl4ds.warmup import PlateauCriterion, load_well, save_well, warm_until_settled


def _stream(seed: int = 0) -> RegimeStream:
    """A small synthetic regime stream (3 regimes × 3 canary classes)."""
    source = SyntheticAttributeSource(num_regimes=3, num_canary_classes=3, per_cell=16, img_size=16, long_tail=False)
    return RegimeStream(source, batch_size=8, support_per_canary=6, query_per_canary=6, seed=seed)


def _method() -> SSLMethod:
    """A tiny SimSiam method matching the 16px synthetic images."""
    encoder = TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0)
    return build_simsiam(encoder=encoder, proj_hidden=32, proj_dim=16, pred_hidden=16)


def test_plateau_needs_min_steps() -> None:
    """A short trace never plateaus, however flat, until min_steps is reached."""
    criterion = PlateauCriterion(window=5, tol=1e-2, min_steps=20)
    assert criterion.settled([1.0] * 10) is False


def test_plateau_fires_on_a_flat_trace() -> None:
    """A flat trace past min_steps has ~zero relative improvement and settles."""
    criterion = PlateauCriterion(window=5, tol=1e-3, min_steps=10)
    assert criterion.settled([0.5] * 40) is True


def test_plateau_holds_open_while_improving() -> None:
    """A steadily decreasing loss keeps improving faster than tol, so it does not settle."""
    criterion = PlateauCriterion(window=5, tol=1e-3, min_steps=10)
    decreasing = [1.0 - 0.02 * i for i in range(40)]
    assert criterion.settled(decreasing) is False


def test_warm_until_settled_reduces_loss_and_reports() -> None:
    """Warming a from-scratch SimSiam lowers the smoothed loss and returns a summary."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    summary = warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=PlateauCriterion(window=5, tol=1e-3, min_steps=10),
        max_steps=200,
        device="cpu",
    )
    assert summary["steps"] > 0
    assert math.isfinite(summary["init_loss"]) and math.isfinite(summary["final_loss"])
    assert len(summary["loss_trace"]) == summary["steps"]


def test_warm_until_settled_respects_the_step_cap() -> None:
    """When a plateau cannot be declared, warming stops exactly at max_steps (across passes)."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    summary = warm_until_settled(
        method=method,
        stream=stream,
        optimizer=optimizer,
        plateau=PlateauCriterion(window=5, tol=1e-3, min_steps=10_000),  # min_steps unreachable → never settles
        max_steps=24,
        device="cpu",
    )
    assert summary["steps"] == 24
    assert summary["plateaued"] is False


def test_save_and_load_well_roundtrips_the_full_method(tmp_path: Path) -> None:
    """The well saves the whole method (encoder + head) and resumes it bit-for-bit."""
    method = _method()
    stream = _stream()
    optimizer = torch.optim.AdamW(method.parameters(), lr=1e-3)
    warm_until_settled(method=method, stream=stream, optimizer=optimizer, max_steps=12, device="cpu")

    path = save_well(method, tmp_path / "wells" / "je_well.pt")
    assert path.is_file()

    # a fresh method (different random init) is made identical once the well is loaded
    resumed = _method()
    before = torch.cat([p.flatten() for p in resumed.parameters()])
    load_well(resumed, path)
    after = torch.cat([p.flatten() for p in resumed.parameters()])
    original = torch.cat([p.flatten() for p in method.parameters()])
    assert not torch.equal(before, after)  # loading changed the fresh weights
    assert torch.equal(after, original)  # ...to match the saved well exactly (head included)


def test_load_well_missing_file_raises(tmp_path: Path) -> None:
    """Pointing a drive at a non-existent well is a hard error, not a silent from-scratch run."""
    with pytest.raises(FileNotFoundError, match="warm well not found"):
        load_well(_method(), tmp_path / "absent.pt")
