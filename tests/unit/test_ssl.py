"""Tests for the SSL methods and the init/checkpoint utilities.

Checks the two things the loop relies on: ``training_step`` returns a differentiable scalar
whose value can be driven down by optimization, and ``encode`` yields the pooled backbone
embedding. Also covers the ``I`` (init) factor: from-scratch, pretrained warm-start, and the
checkpoint round-trip.
"""

from pathlib import Path

import pytest
import torch

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod, apply_encoder_init, load_encoder_checkpoint, save_encoder_checkpoint
from cafl4ds.ssl.factory import build_mae, build_simsiam
from cafl4ds.ssl.simsiam import _neg_cosine


def _encoder() -> TinyViTEncoder:
    """Return a tiny encoder for fast tests."""
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)


def _batch(n: int = 8, size: int = 16) -> torch.Tensor:
    """Return a small random image batch."""
    g = torch.Generator()
    g.manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_training_step_returns_differentiable_scalar(build: object) -> None:
    """Each method's training step is a scalar tensor with a gradient path to the encoder."""
    method = build(_encoder())  # type: ignore[operator]
    loss = method.training_step(_batch())
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in method.encoder.parameters())


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_loss_decreases_when_overfitting_a_fixed_batch(build: object) -> None:
    """Optimizing a fixed batch drives the SSL loss down (the objective is learnable)."""
    torch.manual_seed(0)
    method = build(_encoder())  # type: ignore[operator]
    method.train()
    opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
    x = _batch()
    first = method.training_step(x).item()
    for _ in range(30):
        opt.zero_grad()
        loss = method.training_step(x)
        loss.backward()
        opt.step()
    assert method.training_step(x).item() < first


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_per_sample_loss_is_a_detached_per_frame_vector(build: object) -> None:
    """Each method exposes a ``[B]`` per-frame loss (no grad) — the loss-gate's score."""
    method = build(_encoder())  # type: ignore[operator]
    per_sample = method.per_sample_loss(_batch(n=6))
    assert per_sample.shape == (6,)
    assert not per_sample.requires_grad
    assert torch.isfinite(per_sample).all()


def test_per_sample_loss_default_raises() -> None:
    """A method that does not define a per-sample loss raises a clear error."""

    class _NoPerSample(SSLMethod):
        def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
            return imgs.mean()

        @property
        def name(self) -> str:
            return "no_per_sample"

    with pytest.raises(NotImplementedError, match="per-sample loss"):
        _NoPerSample(_encoder()).per_sample_loss(_batch())


@pytest.mark.parametrize("build", [build_mae, build_simsiam])
def test_encode_returns_pooled_embedding_without_grad(build: object) -> None:
    """Encode yields ``[B, embed_dim]`` detached embeddings for the instruments."""
    method = build(_encoder())  # type: ignore[operator]
    z = method.encode(_batch(n=5))
    assert z.shape == (5, method.encoder.embed_dim)
    assert not z.requires_grad


def test_anti_collapse_off_bypasses_predictor_and_renames() -> None:
    """The PC ablation removes the predictor from the graph (no grad) and renames the method."""
    healthy = build_simsiam(_encoder(), anti_collapse=True)
    pc = build_simsiam(_encoder(), anti_collapse=False)
    assert healthy.name == "simsiam" and pc.name == "simsiam_collapse"

    for method, predictor_used in ((healthy, True), (pc, False)):
        method.train()
        method.training_step(_batch()).backward()
        pred_has_grad = any(p.grad is not None for p in method.predictor.parameters())
        assert pred_has_grad is predictor_used, f"predictor grad={pred_has_grad}, expected {predictor_used}"
        # The encoder is always in the graph (it is what we measure).
        assert any(p.grad is not None for p in method.encoder.parameters())


def test_stop_gradient_toggle_controls_target_branch_gradient() -> None:
    """The other half of the ablation: ``stop_grad`` gates whether gradient flows to the target.

    With ``stop_grad=True`` (SimSiam as published) the target branch is detached — no gradient
    reaches it; with ``stop_grad=False`` (the PC ablation) the gradient flows through both
    branches. The behavioural collapse *contrast* is a streaming-scale phenomenon exercised by
    ``scripts/positive_control.py`` (a single tiny fixed batch collapses either way), so it is
    validated there, not here.
    """
    p = torch.randn(4, 8, requires_grad=True)
    z = torch.randn(4, 8, requires_grad=True)
    _neg_cosine(p, z, stop_grad=True).backward()
    assert z.grad is None, "stop-gradient should detach the target branch"

    p2, z2 = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    _neg_cosine(p2, z2, stop_grad=False).backward()
    assert z2.grad is not None, "without stop-gradient the target branch must receive gradient"


def test_default_simsiam_objective_unchanged_by_the_partial_collapse_refactor() -> None:
    """With both P0.2.5 knobs at defaults, ``training_step`` is the published stop-grad objective.

    The knob refactor restructured ``training_step``; this pins that the default path
    (``collapse_alpha=0``, ``stopgrad_beta=1``) still equals the hand-computed symmetric
    predictor + full-stop-gradient loss, byte-for-byte, so every pre-P0.2.5 run is unchanged.
    """
    torch.manual_seed(0)
    method = build_simsiam(_encoder(), anti_collapse=True)  # defaults: alpha=0, beta=1
    method.train()
    x = _batch()

    torch.manual_seed(1)
    v1, v2 = method.two_view(x)
    z1 = method.projector(method.encoder.embed(v1))
    z2 = method.projector(method.encoder.embed(v2))
    p1, p2 = method.predictor(z1), method.predictor(z2)
    expected = 0.5 * (_neg_cosine(p1, z2, stop_grad=True) + _neg_cosine(p2, z1, stop_grad=True))

    torch.manual_seed(1)
    got = method.training_step(x)
    assert torch.allclose(got, expected)


def test_collapse_alpha_blends_healthy_and_forced_collapse() -> None:
    """The loss-blend knob is the exact convex mix of the two calibrated objectives (P0.2.5).

    ``alpha=1`` must equal the forced-collapse objective (``anti_collapse=False``), and an
    interior ``alpha`` must be the literal convex combination ``(1-alpha)L_healthy + alpha
    L_collapse`` — so the ladder genuinely interpolates between the two poles.
    """
    torch.manual_seed(0)
    method = build_simsiam(_encoder(), anti_collapse=True)
    method.train()
    x = _batch()

    method.collapse_alpha = 0.0
    torch.manual_seed(1)
    loss_healthy = method.training_step(x)
    method.collapse_alpha = 1.0
    torch.manual_seed(1)
    loss_alpha1 = method.training_step(x)
    method.collapse_alpha = 0.5
    torch.manual_seed(1)
    loss_mid = method.training_step(x)

    # alpha=1 reproduces the forced-collapse objective exactly.
    method.collapse_alpha = 0.0
    method.anti_collapse = False
    torch.manual_seed(1)
    loss_forced = method.training_step(x)
    assert torch.allclose(loss_alpha1, loss_forced)
    # An interior blend is the convex combination of the two endpoints.
    assert torch.allclose(loss_mid, 0.5 * (loss_healthy + loss_alpha1), atol=1e-6)


def test_stopgrad_beta_is_wired_into_the_target_branch_gradient() -> None:
    """The soft-stop-gradient knob genuinely changes the graph, not just the forward value.

    Detaching only affects the backward pass, so ``beta`` leaves the loss *value* unchanged
    (identical views) but must alter the encoder gradient: at ``beta=0`` the target branch is
    fully attached and feeds an extra gradient path that ``beta=1`` (published stop-gradient)
    removes.
    """

    def encoder_grad_norm(method: object, beta: float) -> float:
        method.stopgrad_beta = beta  # type: ignore[attr-defined]
        method.zero_grad()  # type: ignore[attr-defined]
        torch.manual_seed(1)
        method.training_step(_batch()).backward()  # type: ignore[attr-defined]
        return float(sum(p.grad.pow(2).sum() for p in method.encoder.parameters() if p.grad is not None) ** 0.5)  # type: ignore[attr-defined]

    torch.manual_seed(0)
    method = build_simsiam(_encoder(), anti_collapse=True)
    method.train()
    assert not torch.isclose(
        torch.tensor(encoder_grad_norm(method, 1.0)), torch.tensor(encoder_grad_norm(method, 0.0))
    ), "stopgrad_beta must change the encoder gradient (the target branch path)"


@pytest.mark.parametrize(("alpha", "beta"), [(-0.1, 1.0), (1.5, 1.0), (0.0, -0.2), (0.0, 2.0)])
def test_partial_collapse_knobs_reject_out_of_range(alpha: float, beta: float) -> None:
    """Both P0.2.5 knobs are validated to ``[0, 1]`` at construction."""
    with pytest.raises(ValueError, match="must be in"):
        build_simsiam(_encoder(), collapse_alpha=alpha, stopgrad_beta=beta)


def test_checkpoint_roundtrip_and_pretrained_init(tmp_path: Path) -> None:
    """Saving then applying a pretrained init reproduces the encoder weights exactly."""
    src = _encoder()
    ckpt = tmp_path / "enc.pt"
    save_encoder_checkpoint(src, ckpt)

    dst = _encoder()
    # Sanity: a fresh encoder differs before loading.
    ref_param = next(iter(src.state_dict().values()))
    assert not torch.allclose(ref_param, next(iter(dst.state_dict().values())))

    apply_encoder_init(dst, mode="pretrained", checkpoint=ckpt)
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_from_scratch_init_is_a_noop() -> None:
    """from_scratch init leaves the encoder's random weights untouched."""
    enc = _encoder()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    apply_encoder_init(enc, mode="from_scratch")
    for k, v in enc.state_dict().items():
        assert torch.allclose(v, before[k])


def test_pretrained_without_checkpoint_raises() -> None:
    """Pretrained init requires a checkpoint path."""
    with pytest.raises(ValueError, match="requires a checkpoint"):
        apply_encoder_init(_encoder(), mode="pretrained", checkpoint=None)


def test_unknown_init_mode_raises() -> None:
    """An unknown init mode is rejected."""
    with pytest.raises(ValueError, match="unknown init mode"):
        apply_encoder_init(_encoder(), mode="warm")


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    """Loading a non-existent checkpoint raises a clear error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_encoder_checkpoint(_encoder(), tmp_path / "nope.pt")
