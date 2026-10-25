"""Representation-health instruments (Phase 0 — *Instrument*).

This is the **measurement apparatus** of the project: the "thermometer" the rest of the
study reads. Every function here is *standalone* — it depends on nothing but embedding
tensors (``Z``, shape ``[N, d]``) or a passed-in ``encoder`` callable. There is no stream,
no SSL model, and no training in this module, by design: per the plan's *Strategic
ordering*, the instruments are built and verified in isolation first, because everything
downstream depends on them being correct.

Signal catalogue implemented here (see ``docs/project-plan/metrics.md`` → *Signal catalogue*):

Geometry / collapse (label-free):
    * :func:`rankme` / :func:`effective_rank` — exponential entropy of the normalized
      singular-value spectrum; the core collapse readout (Roy & Vetterli 2007; Garrido
      et al. 2023).
    * :func:`feature_variance` — per-dimension variance; flags dimensions collapsing to a
      constant (VICReg variance term, Bardes et al. 2022).
    * :func:`offdiag_covariance` — sum of squared off-diagonal covariances; flags
      informational (redundancy) collapse (VICReg covariance term, Bardes et al. 2022).
    * :func:`alignment` / :func:`uniformity` — positive-pair closeness and spread on the
      hypersphere, for the joint-embedding collapse demo (Wang & Isola 2020).

Dynamics / stability (label-free):
    * :func:`linear_cka` / :func:`cka_drift` — representation drift of a fixed probe set
      across checkpoints (CKA, Kornblith et al. 2019). Rotation/isotropic-scale invariant.
    * :func:`cosine_drift` — rotation-*sensitive* per-sample churn; the direct readout of
      "how fast the coordinate frame moves" (the moving-reference problem).

Downstream / forgetting (labels used HERE ONLY — never in any SSL/training path):
    * :func:`knn_probe` — nearest-neighbour accuracy on frozen features (Wu et al. 2018;
      Caron et al. 2021).
    * :func:`linear_probe` — linear-head accuracy on frozen features (standard SSL eval).
    * :func:`frozen_baseline` — evaluate a never-updated encoder (baseline **B5**);
      init-matched (frozen-pretrained *or* frozen-random).

Conventions:
    * Embedding matrices are ``[N, d]`` (rows are samples). Inputs may be ``torch.Tensor``
      or ``numpy.ndarray``; everything is computed in ``float`` on CPU.
    * Scalar metrics return Python ``float``; per-dimension metrics return a 1-D
      ``torch.Tensor`` of length ``d``.
    * An ``encoder`` is any callable ``Tensor -> Tensor`` mapping a batch of inputs to a
      batch of embeddings.
"""

from collections.abc import Callable, Sequence
from typing import TypeAlias, cast

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

# Anything we accept as an embedding matrix / label vector.
TensorLike: TypeAlias = torch.Tensor | np.ndarray

# A callable mapping a batch of inputs to a batch of embeddings.
Encoder: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

# A ``(inputs, labels)`` pair for the probe instruments. Labels are used HERE ONLY.
Dataset: TypeAlias = tuple[TensorLike, TensorLike]


def _as_tensor(x: TensorLike) -> torch.Tensor:
    """Return ``x`` as a detached, contiguous float ``torch.Tensor`` on the CPU.

    Args:
        x: A ``torch.Tensor`` or ``numpy.ndarray``.

    Returns:
        A 2-D-preserving float tensor (dtype ``float32``) detached from any graph.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x.detach()
    else:  # pragma: no cover - defensive; callers pass tensors/arrays.
        t = torch.as_tensor(x)
    return t.to(dtype=torch.float32).cpu().contiguous()


def _to_float(x: torch.Tensor) -> float:
    """Extract a Python ``float`` from a scalar tensor (keeps mypy strict happy)."""
    return cast(float, x.item())


def _check_matrix(z: torch.Tensor, name: str = "Z") -> torch.Tensor:
    """Validate that ``z`` is a 2-D ``[N, d]`` embedding matrix.

    Args:
        z: Candidate embedding matrix.
        name: Name used in error messages.

    Returns:
        ``z`` unchanged, once validated.

    Raises:
        ValueError: If ``z`` is not 2-dimensional or is empty.
    """
    if z.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N, d]; got shape {tuple(z.shape)}.")
    if z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty; got shape {tuple(z.shape)}.")
    return z


# --------------------------------------------------------------------------------------
# Geometry / collapse (label-free)
# --------------------------------------------------------------------------------------


def _singular_values(z: TensorLike) -> torch.Tensor:
    """Return the singular values of an embedding matrix in descending order.

    Args:
        z: Embedding matrix ``[N, d]``.

    Returns:
        A 1-D tensor of ``min(N, d)`` singular values, sorted descending.
    """
    zt = _check_matrix(_as_tensor(z))
    return cast(torch.Tensor, torch.linalg.svdvals(zt))


def effective_rank(z: TensorLike) -> float:
    """Effective rank: exponential of the entropy of the normalized singular-value spectrum.

    Roy & Vetterli (2007). Given singular values ``s`` of ``Z``, form the probability
    distribution ``p = s / sum(s)`` and return ``exp(H(p))`` where ``H`` is the Shannon
    entropy in nats. Reads out how many embedding dimensions are *actually in use*: a
    rank-1 (fully collapsed) representation scores ~1; ``k`` equal directions score ~``k``.

    Args:
        z: Embedding matrix ``[N, d]``.

    Returns:
        The effective rank, a float in ``[1, min(N, d)]``.
    """
    s = _singular_values(z)
    total = s.sum()
    if total <= 0:  # pragma: no cover - all-zero embeddings; degenerate but well-defined.
        return 1.0
    # Drop exact-zero singular values (from rank-deficient matrices) before the log so they
    # contribute 0 to the entropy (the -p*log(p) -> 0 limit) rather than producing NaNs.
    p = s[s > 0] / total
    entropy = -(p * p.log()).sum()
    return _to_float(torch.exp(entropy))


def rankme(z: TensorLike, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023): effective rank with the paper's numerical smoothing.

    Identical in spirit to :func:`effective_rank`, using the RankMe formulation
    ``p_k = s_k / ||s||_1 + eps`` before taking the exponential entropy. The ``eps`` term
    keeps the log well-defined and matches the reference implementation.

    Args:
        z: Embedding matrix ``[N, d]``.
        eps: Numerical-stability constant added to each normalized singular value.

    Returns:
        The RankMe score, a float in ``[1, min(N, d)]``.
    """
    s = _singular_values(z)
    total = s.sum()
    if total <= 0:  # pragma: no cover - all-zero embeddings.
        return 1.0
    p = s / total + eps
    entropy = -(p * p.log()).sum()
    return _to_float(torch.exp(entropy))


def feature_variance(z: TensorLike, unbiased: bool = True) -> torch.Tensor:
    """Per-dimension variance of the embeddings (VICReg variance term, Bardes et al. 2022).

    Flags dimensions that have collapsed to a (near-)constant value: such a dimension
    reads ~0. Returned per-dimension rather than aggregated so callers can inspect *which*
    dimensions are collapsing.

    Args:
        z: Embedding matrix ``[N, d]``.
        unbiased: Whether to use Bessel's correction (``ddof=1``), as VICReg does.

    Returns:
        A 1-D tensor of length ``d`` giving the variance of each embedding dimension.
    """
    zt = _check_matrix(_as_tensor(z))
    return zt.var(dim=0, unbiased=unbiased)


def offdiag_covariance(z: TensorLike) -> float:
    """VICReg covariance term: mean squared off-diagonal covariance (Bardes et al. 2022).

    Computes the covariance matrix ``C`` of the (feature-centered) embeddings and returns
    ``sum_{i != j} C[i, j]^2 / d``. Flags *informational* (redundancy) collapse: when
    dimensions become linearly redundant the off-diagonal mass grows; a decorrelated
    representation scores ~0.

    Args:
        z: Embedding matrix ``[N, d]`` with ``N >= 2``.

    Returns:
        The mean squared off-diagonal covariance, a non-negative float.

    Raises:
        ValueError: If ``N < 2`` (covariance is undefined for a single sample).
    """
    zt = _check_matrix(_as_tensor(z))
    n, d = zt.shape
    if n < 2:
        raise ValueError(f"offdiag_covariance needs N >= 2 samples; got N={n}.")
    zc = zt - zt.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return _to_float(off_diag_sq / d)


def _l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2-normalize an embedding matrix onto the unit hypersphere.

    Args:
        z: Embedding matrix ``[N, d]``.
        eps: Lower bound on the norm to avoid division by zero.

    Returns:
        ``z`` with each row scaled to unit L2 norm.
    """
    return cast(torch.Tensor, z / z.norm(dim=1, keepdim=True).clamp_min(eps))


def alignment(
    pos_a: TensorLike | Sequence[TensorLike],
    pos_b: TensorLike | None = None,
    *,
    alpha: float = 2.0,
) -> float:
    """Alignment of positive pairs on the hypersphere (Wang & Isola 2020).

    ``E[ ||f(x) - f(y)||_2^alpha ]`` over positive pairs ``(x, y)``, both L2-normalized to
    the unit sphere. Lower is better (positives map close together); it *rises* as a
    joint-embedding model's positives drift apart.

    Args:
        pos_a: Either the first matrix of a positive pair ``[N, d]``, or — when ``pos_b`` is
            omitted — a ``(z_a, z_b)`` sequence of the two matrices.
        pos_b: The second matrix of the positive pair ``[N, d]``, aligned row-for-row with
            ``pos_a``. Omit when passing the pair as ``pos_a``.
        alpha: Exponent on the pairwise distance (Wang & Isola use ``alpha = 2``).

    Returns:
        The mean aligned distance, a non-negative float.

    Raises:
        ValueError: If the two matrices differ in shape.
    """
    if pos_b is None:
        pair = cast("Sequence[TensorLike]", pos_a)  # unpack the (z_a, z_b) sequence.
        a_mat, b_mat = pair[0], pair[1]
    else:
        a_mat, b_mat = cast(TensorLike, pos_a), pos_b
    za = _check_matrix(_as_tensor(a_mat), "pos_a")
    zb = _check_matrix(_as_tensor(b_mat), "pos_b")
    if za.shape != zb.shape:
        raise ValueError(f"positive pairs must share a shape; got {tuple(za.shape)} vs {tuple(zb.shape)}.")
    za, zb = _l2_normalize(za), _l2_normalize(zb)
    dist = (za - zb).norm(dim=1)
    return _to_float(dist.pow(alpha).mean())


def uniformity(z: TensorLike, t: float = 2.0) -> float:
    """Uniformity of embeddings on the hypersphere (Wang & Isola 2020).

    ``log E_{x,y}[ exp(-t * ||x - y||_2^2) ]`` over all distinct pairs, with rows
    L2-normalized to the unit sphere. *Lower* (more negative) means the embeddings are
    spread more uniformly; the value climbs toward 0 as embeddings clump together
    (a collapse signature).

    Args:
        z: Embedding matrix ``[N, d]`` with ``N >= 2``.
        t: Temperature of the Gaussian potential (Wang & Isola use ``t = 2``).

    Returns:
        The uniformity loss, a float (typically negative).

    Raises:
        ValueError: If ``N < 2`` (no pairs exist).
    """
    zt = _check_matrix(_as_tensor(z))
    if zt.shape[0] < 2:
        raise ValueError(f"uniformity needs N >= 2 samples; got N={zt.shape[0]}.")
    zt = _l2_normalize(zt)
    # pdist gives the condensed vector of pairwise L2 distances (upper triangle, i < j).
    sq_dist = torch.pdist(zt).pow(2)
    return _to_float(torch.log(torch.exp(-t * sq_dist).mean()))


# --------------------------------------------------------------------------------------
# Quality readers on trial (label-free) — candidate MAE-native quality signals (P0.5.2)
# --------------------------------------------------------------------------------------


def clusterability(z: TensorLike, n_clusters: int, *, seed: int = 0, n_init: int = 10) -> float:
    """Unsupervised clusterability: the silhouette of a k-means partition of the frozen rep.

    A **label-free** proxy for the class separability the frozen probe measures *with* labels
    (P0.5.2): partition the embeddings into ``n_clusters`` groups with k-means, then score how
    well-separated that partition is via the mean silhouette coefficient. A representation that has
    arranged samples into tight, well-separated clusters scores near ``+1``; a diffuse,
    structure-free representation scores near ``0``. ``n_clusters`` is the *count* of classes
    (dataset metadata), never the per-sample labels — nothing supervised enters here.

    Embeddings are L2-normalized first (onto the unit hypersphere, matching the cosine geometry the
    kNN probe and uniformity use), so the Euclidean silhouette on the normalized features tracks
    cosine separability.

    Args:
        z: Embedding matrix ``[N, d]``.
        n_clusters: Number of k-means clusters (the class *count* — metadata, not labels).
        seed: Random seed for the k-means initialization (deterministic across checkpoints).
        n_init: Number of k-means restarts (best inertia kept), as in scikit-learn.

    Returns:
        The mean silhouette coefficient in ``[-1, 1]`` (higher = more separable). Returns ``0.0``
        for degenerate cases (too few samples, or k-means yields a single non-empty cluster) where
        the silhouette is undefined.
    """
    zt = _check_matrix(_as_tensor(z))
    z_unit = _l2_normalize(zt).numpy()
    n = z_unit.shape[0]
    k = max(2, min(int(n_clusters), n - 1))
    if n <= k:
        return 0.0
    labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(z_unit)
    if len(np.unique(labels)) < 2:  # pragma: no cover - degenerate single cluster; silhouette undefined.
        return 0.0
    return float(silhouette_score(z_unit, labels, metric="euclidean"))


def mean_attention_distance(attn: TensorLike, grid_size: int) -> float:
    """Attention-weighted mean spatial distance between patches (ViT locality readout, P0.5.2).

    A **label-free**, MAE/ViT-native quality candidate. A *shortcut* encoder solves reconstruction
    by copying neighbouring patches, so its heads attend **locally** (short mean distance); a
    *semantic* encoder grows **long-range** heads (longer mean distance). Given the self-attention
    probability matrix, this returns the mean — over samples, heads, and query patches — of the
    attention-weighted distance from each query patch to the key patches it attends to, in patch
    (grid-cell) units.

    The class token (sequence index 0) is dropped from both the query and key axes and the
    remaining patch-key attention is renormalized to sum to one, so the readout is a pure
    patch-to-patch spatial statistic.

    Args:
        attn: Attention probabilities with the token axes last, ``[..., T, T]`` where
            ``T = 1 + grid_size**2`` (class token first). Leading axes (batch, head) are averaged.
        grid_size: Side length ``g`` of the square patch grid (``num_patches = g * g``).

    Returns:
        The mean attention distance in patch units, a non-negative float.

    Raises:
        ValueError: If ``attn``'s last two dims are not each ``1 + grid_size**2``.
    """
    a = _as_tensor(attn)
    t = grid_size * grid_size + 1
    if a.shape[-1] != t or a.shape[-2] != t:
        raise ValueError(f"attn last two dims must each be {t} (1 + {grid_size}^2); got {tuple(a.shape[-2:])}.")
    rows = torch.arange(grid_size).repeat_interleave(grid_size)
    cols = torch.arange(grid_size).repeat(grid_size)
    coords = torch.stack([rows, cols], dim=1).to(torch.float32)  # [N, 2] patch-grid centers
    dist = torch.cdist(coords, coords)  # [N, N] patch-center distances
    a_pp = a[..., 1:, 1:]  # drop the cls token from the query + key axes
    a_pp = a_pp / a_pp.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # renormalize over patch keys
    weighted = (a_pp * dist).sum(dim=-1)  # [..., N] expected distance per query patch
    return _to_float(weighted.mean())


# --------------------------------------------------------------------------------------
# Dynamics / stability (label-free)
# --------------------------------------------------------------------------------------


def linear_cka(x: TensorLike, y: TensorLike) -> float:
    """Linear Centered Kernel Alignment between two representations (Kornblith et al. 2019).

    Both matrices describe the *same* set of ``N`` samples under two representations
    (e.g. a fixed probe set at two checkpoints). Returns a similarity in ``[0, 1]`` that is
    invariant to orthogonal transforms and isotropic scaling of either representation
    (that invariance is the point of CKA — it compares representational *content*, not the
    coordinate frame; see :func:`cosine_drift` for a frame-sensitive companion).

    Args:
        x: First representation ``[N, d_x]``.
        y: Second representation ``[N, d_y]`` (same ``N`` as ``x``).

    Returns:
        The linear CKA similarity, a float in ``[0, 1]`` (1.0 = identical up to
        rotation/scaling).

    Raises:
        ValueError: If ``x`` and ``y`` have a different number of samples.
    """
    xt = _check_matrix(_as_tensor(x), "x")
    yt = _check_matrix(_as_tensor(y), "y")
    if xt.shape[0] != yt.shape[0]:
        raise ValueError(f"CKA needs matching sample counts; got {xt.shape[0]} vs {yt.shape[0]}.")
    xc = xt - xt.mean(dim=0, keepdim=True)
    yc = yt - yt.mean(dim=0, keepdim=True)
    # Linear HSIC via the feature-space cross-covariance: ||Yc^T Xc||_F^2.
    hsic_xy = (yc.T @ xc).pow(2).sum()
    hsic_xx = (xc.T @ xc).pow(2).sum()
    hsic_yy = (yc.T @ yc).pow(2).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:  # pragma: no cover - a constant representation; similarity undefined.
        return 0.0
    return _to_float(hsic_xy / denom)


def cka_drift(z_ref_t0: TensorLike, z_ref_t: TensorLike) -> float:
    """Representation drift of a fixed probe set via linear CKA (``1 - CKA``).

    Measures how much the *content* of a fixed probe set's representation has changed
    between two checkpoints. ``0`` means unchanged (identical up to rotation/scaling);
    larger means more drift. Because linear CKA is rotation-invariant, a pure change of
    coordinate frame reads ~0 here — use :func:`cosine_drift` to detect that.

    Args:
        z_ref_t0: Probe-set embeddings at the reference checkpoint ``[N, d0]``.
        z_ref_t: Probe-set embeddings at a later checkpoint ``[N, dt]`` (same ``N``).

    Returns:
        The drift ``1 - linear_cka``, a float in ``[0, 1]``.
    """
    return 1.0 - linear_cka(z_ref_t0, z_ref_t)


def cosine_drift(z_ref_t0: TensorLike, z_ref_t: TensorLike) -> float:
    """Per-sample cosine churn of a fixed probe set (``1 - mean cosine similarity``).

    Unlike :func:`cka_drift`, this is *sensitive* to rotation of the embedding coordinate
    frame: it compares each sample's embedding to its own embedding at the reference
    checkpoint. This is the direct readout of "how fast the coordinate frame moves" — the
    moving-reference problem that destabilizes a fixed novelty reference. Requires the two
    checkpoints to share an embedding dimensionality (same coordinate axes).

    Args:
        z_ref_t0: Probe-set embeddings at the reference checkpoint ``[N, d]``.
        z_ref_t: Probe-set embeddings at a later checkpoint ``[N, d]`` (same ``N`` and ``d``).

    Returns:
        The mean per-sample cosine drift, a float in ``[0, 2]`` (0 = unchanged direction).

    Raises:
        ValueError: If the two matrices do not share a shape.
    """
    z0 = _check_matrix(_as_tensor(z_ref_t0), "z_ref_t0")
    zt = _check_matrix(_as_tensor(z_ref_t), "z_ref_t")
    if z0.shape != zt.shape:
        raise ValueError(f"cosine_drift needs matching shapes; got {tuple(z0.shape)} vs {tuple(zt.shape)}.")
    cos = torch.nn.functional.cosine_similarity(z0, zt, dim=1)
    return _to_float(1.0 - cos.mean())


# --------------------------------------------------------------------------------------
# Downstream / forgetting (labels used HERE ONLY)
# --------------------------------------------------------------------------------------


def _encode(encoder: Encoder, inputs: TensorLike) -> np.ndarray:
    """Run ``encoder`` over ``inputs`` (no grad) and return a 2-D ``float`` numpy array.

    Args:
        encoder: Callable mapping a batch of inputs to a batch of embeddings.
        inputs: Model inputs accepted by ``encoder``.

    Returns:
        The embeddings as a ``[N, d]`` numpy array.
    """
    with torch.no_grad():
        z = encoder(_as_tensor(inputs))
    zt = _as_tensor(z)
    return _check_matrix(zt, "encoder output").numpy()


def _labels(y: TensorLike) -> np.ndarray:
    """Coerce a label vector to a 1-D numpy array.

    Args:
        y: Label vector (tensor or array).

    Returns:
        A 1-D numpy array of labels.
    """
    arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    return arr.reshape(-1)


def knn_probe(
    encoder: Encoder,
    support: Dataset,
    query: Dataset,
    k: int = 20,
    *,
    metric: str = "cosine",
    weights: str = "distance",
) -> float:
    """k-NN probe accuracy on frozen features (Wu et al. 2018; Caron et al. 2021).

    Encodes the ``support`` and ``query`` inputs with the (frozen) ``encoder``, fits a
    nearest-neighbour classifier on the support embeddings/labels, and reports accuracy on
    the query set. Labels are used HERE ONLY — this is an evaluation probe, never a
    training signal.

    Args:
        encoder: Frozen encoder callable ``inputs -> embeddings``.
        support: ``(inputs, labels)`` used as the neighbour database.
        query: ``(inputs, labels)`` to classify and score.
        k: Number of neighbours. Clamped to the support-set size if larger.
        metric: Distance metric passed to scikit-learn (default cosine, the SSL k-NN norm).
        weights: Neighbour weighting (``"distance"`` or ``"uniform"``).

    Returns:
        Top-1 query accuracy in ``[0, 1]``.
    """
    xs, ys = _encode(encoder, support[0]), _labels(support[1])
    xq, yq = _encode(encoder, query[0]), _labels(query[1])
    n_neighbors = max(1, min(k, xs.shape[0]))
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    clf.fit(xs, ys)
    return float((clf.predict(xq) == yq).mean())


def linear_probe(
    encoder: Encoder,
    support: Dataset,
    query: Dataset,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    standardize: bool = True,
) -> float:
    """Linear-probe accuracy on frozen features (standard SSL linear-evaluation protocol).

    Encodes ``support`` and ``query`` with the (frozen) ``encoder`` and fits a multinomial
    logistic-regression head on the support embeddings, then scores the query set. Only a
    *linear* head is trained; the encoder is never updated. Labels are used HERE ONLY.

    Args:
        encoder: Frozen encoder callable ``inputs -> embeddings``.
        support: ``(inputs, labels)`` used to fit the linear head.
        query: ``(inputs, labels)`` to score.
        max_iter: Maximum solver iterations for the logistic-regression head.
        C: Inverse L2-regularization strength for the linear head.
        standardize: Whether to zero-mean/unit-variance the features (fit on support) before
            the linear head — the usual, more stable linear-probe recipe.

    Returns:
        Top-1 query accuracy in ``[0, 1]``.
    """
    xs, ys = _encode(encoder, support[0]), _labels(support[1])
    xq, yq = _encode(encoder, query[0]), _labels(query[1])
    if standardize:
        mean = xs.mean(axis=0, keepdims=True)
        std = xs.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        xs = (xs - mean) / std
        xq = (xq - mean) / std
    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(xs, ys)
    return float((clf.predict(xq) == yq).mean())


def frozen_baseline(
    encoder_init: Encoder,
    support: Dataset,
    query: Dataset,
    *,
    probe: str = "knn",
    **probe_kwargs: object,
) -> float:
    """Baseline **B5**: evaluate a never-updated encoder (init-matched frozen backbone).

    B5 is the "no adaptation" floor of the study — a backbone that never sees a gradient
    step. It is *init-matched*: pass a frozen-pretrained encoder in the pretrained regime,
    or a frozen-random encoder in the from-scratch regime. This is a thin wrapper that runs
    the chosen probe on ``encoder_init``; the point is the frozen encoder, not new logic.

    Args:
        encoder_init: The frozen encoder (pretrained or randomly initialized) to evaluate.
        support: ``(inputs, labels)`` for the probe's support/training split.
        query: ``(inputs, labels)`` for the probe's query/eval split.
        probe: Which downstream probe to run: ``"knn"`` or ``"linear"``.
        **probe_kwargs: Forwarded to :func:`knn_probe` / :func:`linear_probe`.

    Returns:
        The probe's top-1 query accuracy in ``[0, 1]``.

    Raises:
        ValueError: If ``probe`` is not ``"knn"`` or ``"linear"``.
    """
    if probe == "knn":
        return knn_probe(encoder_init, support, query, **probe_kwargs)  # type: ignore[arg-type]
    if probe == "linear":
        return linear_probe(encoder_init, support, query, **probe_kwargs)  # type: ignore[arg-type]
    raise ValueError(f"probe must be 'knn' or 'linear'; got {probe!r}.")
