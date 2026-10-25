"""Synthetic known-answer tests for the representation-health instruments.

Phase 0 exit criterion (part 1): the *instruments* must be verified in isolation, on
minimal synthetic data with analytically known answers, before any stream, SSL model, or
positive control is built. No model is trained here; where an encoder is needed it is a
trivial closed-form callable (identity or a fixed random projection).

Reference answers used below:
    * rank-1 matrix                    -> effective_rank / rankme ~ 1
    * ``k`` equal nonzero singular vals -> ~ k
    * isotropic Gaussian (N >> d)      -> ~ d
    * separable clusters + identity enc -> knn / linear probe ~ 100%
    * identical vs. changed probe set   -> cka_drift 0 vs. > 0
    * identical vs. rotated probe set   -> cosine_drift 0 vs. > 0
"""

import numpy as np
import pytest
import torch

from cafl4ds.measurements import (
    alignment,
    cka_drift,
    clusterability,
    cosine_drift,
    effective_rank,
    feature_variance,
    frozen_baseline,
    knn_probe,
    linear_cka,
    linear_probe,
    mean_attention_distance,
    offdiag_covariance,
    rankme,
    uniformity,
)

SEED = 0


def _gen(seed: int = SEED) -> torch.Generator:
    """Return a seeded CPU generator for deterministic synthetic data."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _orthonormal(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Return an ``[n, k]`` matrix with orthonormal columns (``n >= k``)."""
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=g))
    return q[:, :k]


def _rank_k_matrix(n: int, d: int, k: int, scale: float, g: torch.Generator) -> torch.Tensor:
    """Build an ``[n, d]`` matrix with exactly ``k`` equal nonzero singular values.

    Args:
        n: Number of rows (samples).
        d: Number of columns (features).
        k: Number of nonzero singular values.
        scale: The common value of the ``k`` nonzero singular values.
        g: Seeded generator.

    Returns:
        A matrix whose singular spectrum is ``[scale] * k`` followed by zeros.
    """
    u = _orthonormal(n, k, g)
    v = _orthonormal(d, k, g)
    return scale * (u @ v.T)


def _blobs(n_per_class: int, d: int, sep: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Two linearly separable Gaussian blobs in ``d`` dimensions.

    Args:
        n_per_class: Samples per class.
        d: Feature dimensionality.
        sep: Separation of the two class means along the first axis.
        g: Seeded generator.

    Returns:
        ``(X, y)`` with ``X`` of shape ``[2 * n_per_class, d]`` and integer labels.
    """
    offset = torch.zeros(d)
    offset[0] = sep
    x0 = torch.randn(n_per_class, d, generator=g) - offset
    x1 = torch.randn(n_per_class, d, generator=g) + offset
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()
    return x, y


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Identity encoder: features are the inputs themselves."""
    return x


# --- Geometry / collapse ---------------------------------------------------------------


def test_rank1_matrix_scores_one() -> None:
    """A rank-1 (fully collapsed) matrix has effective rank / RankMe ~ 1."""
    g = _gen()
    z = _rank_k_matrix(n=200, d=32, k=1, scale=4.0, g=g)
    assert effective_rank(z) == pytest.approx(1.0, abs=1e-3)
    assert rankme(z) == pytest.approx(1.0, abs=1e-2)


def test_k_equal_singular_values_scores_k() -> None:
    """``k`` equal nonzero singular values give effective rank / RankMe ~ k."""
    g = _gen()
    for k in (3, 5, 8):
        z = _rank_k_matrix(n=200, d=32, k=k, scale=3.0, g=g)
        assert effective_rank(z) == pytest.approx(k, rel=1e-3)
        assert rankme(z) == pytest.approx(k, rel=2e-2)


def test_isotropic_gaussian_scores_near_d() -> None:
    """An isotropic Gaussian with N >> d uses ~all d dimensions, so rank ~ d."""
    g = _gen()
    d = 16
    z = torch.randn(4000, d, generator=g)
    er = effective_rank(z)
    rm = rankme(z)
    assert 0.8 * d < er <= d + 1e-3
    assert 0.8 * d < rm <= d + 1e-3


def test_effective_rank_accepts_numpy() -> None:
    """Instruments accept numpy arrays as well as tensors."""
    g = _gen()
    z = _rank_k_matrix(n=100, d=16, k=4, scale=2.0, g=g).numpy()
    assert effective_rank(z) == pytest.approx(4.0, rel=1e-3)


def test_feature_variance_flags_collapsed_dimension() -> None:
    """A constant embedding dimension reads ~0 variance; live dimensions read > 0."""
    g = _gen()
    z = torch.randn(500, 6, generator=g)
    z[:, 2] = 5.0  # collapse dimension 2 to a constant.
    var = feature_variance(z)
    assert var.shape == (6,)
    assert var[2] == pytest.approx(0.0, abs=1e-6)
    assert (var[[0, 1, 3, 4, 5]] > 0.5).all()


def test_offdiag_covariance_zero_for_independent_large_for_redundant() -> None:
    """Off-diagonal covariance is ~0 for decorrelated dims and large for redundant ones."""
    g = _gen()
    indep = torch.randn(5000, 8, generator=g)
    redundant = torch.randn(5000, 8, generator=g)
    redundant[:, 1] = redundant[:, 0] + 1e-3 * torch.randn(5000, generator=g)  # dim 1 ~ dim 0.
    c_indep = offdiag_covariance(indep)
    c_redundant = offdiag_covariance(redundant)
    assert c_indep < 0.05
    assert c_redundant > 0.1
    assert c_redundant > 10 * c_indep


def test_offdiag_covariance_requires_two_samples() -> None:
    """Covariance is undefined for a single sample."""
    with pytest.raises(ValueError, match="N >= 2"):
        offdiag_covariance(torch.randn(1, 4))


def test_alignment_zero_for_identical_pairs() -> None:
    """Identical positive pairs are perfectly aligned (distance 0)."""
    g = _gen()
    z = torch.randn(300, 16, generator=g)
    assert alignment(z, z.clone()) == pytest.approx(0.0, abs=1e-6)


def test_alignment_rises_for_mismatched_pairs() -> None:
    """Opposite / random positive pairs align worse than identical ones."""
    g = _gen()
    z = torch.randn(300, 16, generator=g)
    other = torch.randn(300, 16, generator=g)
    assert alignment(z, -z) > 1.0  # antipodal unit vectors: squared distance ~4.
    assert alignment(z, other) > alignment(z, z.clone())


def test_alignment_accepts_pair_sequence() -> None:
    """``alignment`` also accepts the pair packed as a single ``(z_a, z_b)`` argument."""
    g = _gen()
    z = torch.randn(50, 8, generator=g)
    packed = alignment((z, z.clone()))
    assert packed == pytest.approx(0.0, abs=1e-6)


def test_uniformity_lower_when_spread_than_clumped() -> None:
    """Spread-out embeddings have lower (more negative) uniformity than clumped ones."""
    g = _gen()
    spread = torch.randn(400, 32, generator=g)
    clumped = 0.01 * torch.randn(400, 32, generator=g)
    clumped[:, 0] += 1.0  # collapse toward a single direction.
    assert uniformity(spread) < uniformity(clumped)


# --- Dynamics / stability --------------------------------------------------------------


def test_cka_identity_and_self_drift_zero() -> None:
    """Linear CKA of a representation with itself is 1, so ``cka_drift`` is 0."""
    g = _gen()
    z = torch.randn(200, 24, generator=g)
    assert linear_cka(z, z.clone()) == pytest.approx(1.0, abs=1e-5)
    assert cka_drift(z, z.clone()) == pytest.approx(0.0, abs=1e-5)


def test_cka_drift_rotation_invariant() -> None:
    """Linear CKA is invariant to an orthogonal change of coordinate frame (drift ~0)."""
    g = _gen()
    z = torch.randn(200, 24, generator=g)
    q = _orthonormal(24, 24, g)  # orthogonal rotation of the embedding axes.
    z_rot = z @ q
    assert cka_drift(z, z_rot) == pytest.approx(0.0, abs=1e-4)


def test_cka_drift_positive_for_genuine_change() -> None:
    """A genuinely different representation of the same samples yields drift > 0."""
    g = _gen()
    z = torch.randn(200, 24, generator=g)
    z_changed = torch.randn(200, 24, generator=g)  # unrelated representation.
    assert cka_drift(z, z_changed) > 0.1


def test_cosine_drift_zero_for_identical_positive_for_rotated() -> None:
    """Cosine churn is 0 for an unchanged frame and > 0 once the frame rotates."""
    g = _gen()
    z = torch.randn(200, 24, generator=g)
    q = _orthonormal(24, 24, g)
    z_rot = z @ q
    assert cosine_drift(z, z.clone()) == pytest.approx(0.0, abs=1e-6)
    assert cosine_drift(z, z_rot) > 0.1


# --- Downstream probes (labels used HERE ONLY) -----------------------------------------


def test_knn_probe_perfect_on_separable_data() -> None:
    """A separable embedding under the identity encoder gives ~100% k-NN accuracy."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=100, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=50, d=8, sep=5.0, g=g)
    acc = knn_probe(_identity, (x_s, y_s), (x_q, y_q), k=5)
    assert acc == pytest.approx(1.0, abs=1e-9)


def test_knn_probe_clamps_k_to_support_size() -> None:
    """``k`` larger than the support set is clamped rather than erroring."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=10, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=10, d=8, sep=5.0, g=g)
    acc = knn_probe(_identity, (x_s, y_s), (x_q, y_q), k=1000)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_perfect_on_separable_data() -> None:
    """A separable embedding under the identity encoder is linearly separable (~100%)."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=100, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=50, d=8, sep=5.0, g=g)
    acc = linear_probe(_identity, (x_s, y_s), (x_q, y_q))
    assert acc == pytest.approx(1.0, abs=1e-9)


def test_probes_accept_numpy_inputs_and_labels() -> None:
    """Probes accept numpy inputs/labels (labels used HERE ONLY)."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=80, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=40, d=8, sep=5.0, g=g)
    acc = knn_probe(_identity, (x_s.numpy(), y_s.numpy()), (x_q.numpy(), y_q.numpy()), k=5)
    assert acc == pytest.approx(1.0, abs=1e-9)


# --- Frozen baseline B5 ----------------------------------------------------------------


def test_frozen_baseline_dispatches_to_probes() -> None:
    """B5 wraps the chosen probe on a never-updated encoder and matches it exactly."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=100, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=50, d=8, sep=5.0, g=g)
    knn_acc = frozen_baseline(_identity, (x_s, y_s), (x_q, y_q), probe="knn", k=5)
    lin_acc = frozen_baseline(_identity, (x_s, y_s), (x_q, y_q), probe="linear")
    assert knn_acc == pytest.approx(knn_probe(_identity, (x_s, y_s), (x_q, y_q), k=5))
    assert lin_acc == pytest.approx(linear_probe(_identity, (x_s, y_s), (x_q, y_q)))


def test_frozen_baseline_random_encoder_runs() -> None:
    """B5 with a frozen-random encoder (from-scratch regime) evaluates to a valid accuracy."""
    g = _gen()
    proj = torch.randn(8, 32, generator=g)  # fixed, never-updated random projection.

    def random_encoder(x: torch.Tensor) -> torch.Tensor:
        return x @ proj

    x_s, y_s = _blobs(n_per_class=100, d=8, sep=5.0, g=g)
    x_q, y_q = _blobs(n_per_class=50, d=8, sep=5.0, g=g)
    acc = frozen_baseline(random_encoder, (x_s, y_s), (x_q, y_q), probe="knn", k=5)
    assert 0.0 <= acc <= 1.0


def test_frozen_baseline_rejects_unknown_probe() -> None:
    """An unknown probe name is rejected."""
    g = _gen()
    x_s, y_s = _blobs(n_per_class=10, d=8, sep=5.0, g=g)
    with pytest.raises(ValueError, match="knn.*linear"):
        frozen_baseline(_identity, (x_s, y_s), (x_s, y_s), probe="mlp")


def test_measurements_do_not_import_numpy_random_state() -> None:
    """Sanity: instruments are deterministic given fixed inputs (no hidden RNG)."""
    g = _gen()
    z = torch.randn(100, 12, generator=g)
    assert effective_rank(z) == effective_rank(z.clone())
    assert offdiag_covariance(z) == offdiag_covariance(z.clone())
    assert np.isfinite(uniformity(z))


# --------------------------------------------------------------------------------------
# Quality readers on trial (P0.5.2): clusterability, mean attention distance
# --------------------------------------------------------------------------------------


def _blob_clusters(n_per: int, d: int, k: int, sep: float, g: torch.Generator) -> torch.Tensor:
    """Return ``[k * n_per, d]`` embeddings drawn as ``k`` well-separated isotropic blobs."""
    centers = sep * torch.eye(d)[:k]
    return torch.cat([centers[i] + torch.randn(n_per, d, generator=g) for i in range(k)], dim=0)


def test_clusterability_separates_structure_from_noise() -> None:
    """Well-separated blobs score high silhouette; isotropic noise scores near zero."""
    g = _gen()
    structured = _blob_clusters(n_per=40, d=8, k=4, sep=8.0, g=g)
    noise = torch.randn(160, 8, generator=g)
    assert clusterability(structured, n_clusters=4) > 0.5
    assert clusterability(noise, n_clusters=4) < 0.15
    # A cratered rep (less separable) must read strictly lower — the direction the reader relies on.
    assert clusterability(structured, n_clusters=4) > clusterability(noise, n_clusters=4)


def test_clusterability_degenerate_returns_zero() -> None:
    """Too few samples for the requested clusters returns the undefined-case floor (0.0)."""
    g = _gen()
    assert clusterability(torch.randn(2, 4, generator=g), n_clusters=4) == 0.0


def _attn_with_grid(grid: int) -> int:
    """Token count ``T = 1 + grid**2`` for a square patch grid."""
    return grid * grid + 1


def test_mean_attention_distance_uniform_equals_mean_pairwise() -> None:
    """Uniform attention reduces to the mean pairwise patch distance (a closed-form check)."""
    grid = 4
    t = _attn_with_grid(grid)
    uniform = torch.full((1, 1, t, t), 1.0 / t)
    rows = torch.arange(grid).repeat_interleave(grid)
    cols = torch.arange(grid).repeat(grid)
    coords = torch.stack([rows, cols], dim=1).to(torch.float32)
    expected = torch.cdist(coords, coords).mean().item()  # renormalized uniform over N keys = mean row
    assert mean_attention_distance(uniform, grid) == pytest.approx(expected, abs=1e-5)


def test_mean_attention_distance_local_is_smaller_than_global() -> None:
    """Self-only (local) attention reads ~0; attend-to-farthest reads large — the reader's axis."""
    grid = 4
    t = _attn_with_grid(grid)
    local = torch.eye(t).reshape(1, 1, t, t)  # each token attends only to itself -> distance 0
    assert mean_attention_distance(local, grid) == pytest.approx(0.0, abs=1e-6)
    # A shortcut (local) encoder must read strictly below a diffuse (uniform) one.
    uniform = torch.full((1, 1, t, t), 1.0 / t)
    assert mean_attention_distance(local, grid) < mean_attention_distance(uniform, grid)


def test_mean_attention_distance_rejects_wrong_shape() -> None:
    """The token axes must match ``1 + grid**2`` or the call errors (guards a wiring bug)."""
    with pytest.raises(ValueError, match="last two dims"):
        mean_attention_distance(torch.rand(1, 1, 10, 10), grid_size=4)  # 4x4 -> expects 17, not 10
