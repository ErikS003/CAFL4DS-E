"""Tests for the Barlow Twins method (P0.2.3's redundancy-collapse vehicle).

Mirrors ``test_monitor.py``'s two-surface / alignment coverage — Barlow exposes the same
backbone + projector surfaces and a positive pair, so the monitor must emit the whole geometry
suite at both surfaces plus alignment — and adds method-level coverage of the ``anti_collapse``
toggle (the redundancy-reduction term is present iff anti-collapse is on) and the loss/gradient
path.
"""

import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.ssl.barlow import _off_diagonal
from cafl4ds.ssl.factory import build_barlow


def _encoder() -> TinyViTEncoder:
    """Build the tiny ViT encoder shared across these tests (fixed seed)."""
    torch.manual_seed(0)
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)


def _barlow_and_monitor() -> tuple[object, HealthMonitor]:
    """Build a tiny Barlow Twins (two surfaces + positive pair) and a monitor over its eval sets."""
    method = build_barlow(_encoder(), proj_hidden=64, proj_dim=32)
    stream = EraStream(
        SyntheticSource(num_classes=3, per_class=40, img_size=16),
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=5,
    )
    return method, HealthMonitor(stream.eval_sets, knn_k=5)


def test_off_diagonal_selects_off_diagonal_entries() -> None:
    """``_off_diagonal`` returns exactly the non-diagonal entries of a square matrix."""
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    off = _off_diagonal(x)
    assert set(off.tolist()) == {1.0, 2.0, 3.0, 5.0, 6.0, 7.0}  # everything but 0, 4, 8
    assert off.numel() == 3 * 2


def test_barlow_reports_both_surfaces_and_alignment() -> None:
    """Barlow exposes backbone + projector surfaces and a positive pair.

    So the monitor emits the geometry suite at both surfaces (``_proj`` keys) plus alignment,
    exactly like SimSiam.
    """
    method, monitor = _barlow_and_monitor()
    metrics = monitor.measure(method, step=0)  # type: ignore[arg-type]
    surfaced = {"rankme", "mean_feature_var", "offdiag_cov", "uniformity", "alignment"}
    expected = {"step", "cka_drift", "cosine_drift", "knn_acc", "linear_acc"}
    expected |= surfaced | {f"{m}_proj" for m in surfaced}
    assert set(metrics) == expected
    assert all(isinstance(v, float) and v == v for v in metrics.values())  # all finite


def test_name_reflects_ablation() -> None:
    """The method name flips to the ``_collapse`` variant when anti-collapse is disabled."""
    assert build_barlow(_encoder()).name == "barlow"
    assert build_barlow(_encoder(), anti_collapse=False).name == "barlow_collapse"


def test_anti_collapse_toggle_adds_redundancy_term() -> None:
    """The full loss exceeds the ablated (invariance-only) loss by the redundancy term.

    Both methods share bit-identical encoder/projector weights (same seed) and see the same
    views (the global RNG is reset before each ``training_step``), so the only difference is the
    ``lambd * off-diagonal`` redundancy-reduction term — non-zero at random init.
    """
    imgs = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]

    full = build_barlow(_encoder(), proj_hidden=64, proj_dim=32, anti_collapse=True)
    ablated = build_barlow(_encoder(), proj_hidden=64, proj_dim=32, anti_collapse=False)

    torch.manual_seed(1)
    full_loss = full.training_step(imgs)
    torch.manual_seed(1)
    ablated_loss = ablated.training_step(imgs)

    assert full_loss.item() > ablated_loss.item()  # redundancy term is strictly positive here


def test_training_step_is_scalar_with_gradient_to_encoder() -> None:
    """``training_step`` returns a differentiable scalar whose gradient reaches the encoder."""
    method = build_barlow(_encoder(), proj_hidden=64, proj_dim=32)
    imgs = SyntheticSource(num_classes=3, per_class=8, img_size=16).load()[0]
    loss = method.training_step(imgs)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in method.encoder.parameters())
