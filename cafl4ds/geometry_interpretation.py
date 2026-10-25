"""Reference-population geometry, explicit invariances and labeled diagnostics."""

from __future__ import annotations

import numpy as np

CONTROLS = (
    "identity",
    "rotation",
    "scale",
    "translation",
    "anisotropic",
    "drop_top_pc",
    "drop_night_contrast",
    "drop_day_contrast",
    "drop_support_orthogonal",
)
METRICS = (
    "energy_ratio",
    "centroid_shift",
    "spectral_l1",
    "effective_rank_ratio",
    "gram_change",
    "within_reference",
    "cross_reference",
    "outside_reference",
    "top_subspace_overlap",
    "reference_gap",
    "neighbor_retention",
    "coordinate_motion",
    "aligned_residual",
    "alignment_scale",
)


def matrix(z):
    a = np.asarray(z, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError("Finite sample-by-feature matrix required")
    return a


def view(z, kind):
    a = matrix(z)
    if kind == "angular":
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(norms <= 1e-14):
            raise ValueError("Undefined angular view of zero rows")
        a = a / norms
    elif kind != "raw":
        raise ValueError(kind)
    return a


def geometry(z, kind="raw"):
    a = view(z, kind)
    mean = a.mean(0)
    x = a - mean
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    energy = float(s @ s)
    if energy <= 1e-24:
        return {
            "mean": mean,
            "x": x,
            "energy": energy,
            "p": np.zeros(len(s)),
            "u": u,
            "rank": 0,
            "effective_rank": 0.0,
            "gram": x @ x.T,
        }
    p = s * s / energy
    rank = int(np.sum(s > max(x.shape) * np.finfo(float).eps * s[0]))
    return {
        "mean": mean,
        "x": x,
        "energy": energy,
        "p": p,
        "u": u,
        "rank": rank,
        "effective_rank": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        "gram": (x @ x.T) / energy,
    }


def alignment(before, after):
    """Fit a similarity transport using corresponding bank pixels only."""
    a, b = matrix(before), matrix(after)
    if a.shape != b.shape:
        raise ValueError("Aligned bank identities and feature widths required")
    ma, mb = a.mean(0), b.mean(0)
    a = a - ma
    b = b - mb
    u, s, vt = np.linalg.svd(b.T @ a, full_matrices=False)
    rotation = u @ vt
    den = float(np.sum(b * b))
    scale = float(s.sum() / den) if den > 1e-24 else 0.0
    return rotation, scale, ma, mb


def transported(z, fit):
    rotation, scale, ma, mb = fit
    return (matrix(z) - mb) @ rotation * scale + ma


def neighbors(x, k=10):
    a = matrix(x)
    d = np.maximum(np.sum(a * a, 1)[:, None] + np.sum(a * a, 1)[None, :] - 2 * a @ a.T, 0)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind="stable")
    return order[:, :k]


def compare(before, after, kind="raw", top_k=16, neighbor_k=10):
    a, b = geometry(before, kind), geometry(after, kind)
    if a["x"].shape != b["x"].shape:
        raise ValueError("Same bank identities required")
    if a["energy"] <= 1e-24 or b["energy"] <= 1e-24:
        raise ValueError("Collapsed centered bank; normalized change is undefined")
    k = min(top_k, a["rank"], b["rank"])
    u = a["u"][:, :k]
    delta = b["gram"] - a["gram"]
    # Orthogonal four-block decomposition in SAMPLE space, invariant to feature rotation.
    pp = u.T @ delta @ u
    uq = u.T @ delta - pp @ u.T
    within = float(np.sum(pp * pp))
    cross = float(2 * np.sum(uq * uq))
    total = float(np.sum(delta * delta))
    outside = max(0.0, total - within - cross)
    q = np.eye(len(delta)) - u @ u.T
    np.testing.assert_allclose(outside, np.sum((q @ delta @ q) ** 2), rtol=1e-8, atol=1e-12)
    fit = alignment(view(before, kind), view(after, kind))
    residual = np.linalg.norm(transported(view(after, kind), fit) - view(before, kind)) / np.sqrt(a["energy"])
    na, nb = neighbors(a["x"], neighbor_k), neighbors(b["x"], neighbor_k)
    retention = float(np.mean([len(set(x) & set(y)) / neighbor_k for x, y in zip(na, nb, strict=True)]))
    gap = float(a["p"][k - 1] - a["p"][k]) if k < len(a["p"]) else None
    return {
        "energy_ratio": b["energy"] / a["energy"],
        "centroid_shift": float(np.linalg.norm(b["mean"] - a["mean"]) / np.sqrt(a["energy"] / len(before))),
        "spectral_l1": float(np.sum(abs(a["p"] - b["p"]))),
        "effective_rank_ratio": b["effective_rank"] / a["effective_rank"],
        "gram_change": total,
        "within_reference": within,
        "cross_reference": cross,
        "outside_reference": outside,
        "top_subspace_overlap": float(np.sum((u.T @ b["u"][:, :k]) ** 2) / k),
        "reference_gap": gap,
        "neighbor_retention": retention,
        "coordinate_motion": float(np.linalg.norm(b["x"] - a["x"]) / np.sqrt(a["energy"])),
        "aligned_residual": float(residual),
        "alignment_scale": fit[1],
        "reference_top_energy": float(a["p"][:k].sum()),
        "terminal_top_energy": float(b["p"][:k].sum()),
        "reference_rank": a["rank"],
        "terminal_rank": b["rank"],
        "used_k": k,
        "reference_effective_rank": a["effective_rank"],
        "terminal_effective_rank": b["effective_rank"],
    }


def contrasts(support, labels, domains, draws):
    """Support-defined directions only; no query labels in map or transformations."""
    z = matrix(support)
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    result = []
    for draw, indices in draws.items():
        indices = np.asarray(indices)
        for domain in (0, 1):
            ids = [indices[(domains[indices] == domain) & (labels[indices] == c)] for c in (0, 1)]
            if min(map(len, ids)) < 2:
                raise ValueError("At least two support observations per class/domain required")
            means = [z[ix].mean(0) for ix in ids]
            w = means[1] - means[0]
            norm = np.linalg.norm(w)
            if norm <= 1e-14:
                raise ValueError("Undefined support contrast")
            unit = w / norm
            within = np.concatenate([(z[ix] - m) @ unit for ix, m in zip(ids, means, strict=True)])
            sd = float(np.sqrt(np.mean(within**2)))
            result.append(
                {
                    "draw": int(draw),
                    "domain": domain,
                    "direction": unit,
                    "midpoint": (means[0] + means[1]) / 2,
                    "within_sd": sd,
                    "separation": norm,
                }
            )
    return result


def transformations(z, directions, seed=194044):
    a = matrix(z)
    d = a.shape[1]
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    centered = a - a.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    span = np.stack([x["direction"] for x in directions], axis=1)
    u, s, _ = np.linalg.svd(span, full_matrices=False)
    u = u[:, s > 1e-10 * s[0]]
    nuisance = vt[0] - u @ (u.T @ vt[0])
    length = np.linalg.norm(nuisance)
    if length <= 1e-10:
        raise ValueError("Support-orthogonal control has no stable direction")
    nuisance /= length
    per_domain = []
    for domain in (0, 1):
        w = np.mean([x["direction"] for x in directions if x["domain"] == domain], axis=0)
        per_domain.append(w / np.linalg.norm(w))
    eye = np.eye(d)
    shift = rng.normal(size=d)
    shift *= 2 * np.sqrt(np.mean(a * a)) / np.sqrt(np.mean(shift * shift))
    return {
        "identity": (eye, np.zeros(d), True),
        "rotation": (q, np.zeros(d), True),
        "scale": (2 * eye, np.zeros(d), True),
        "translation": (eye, shift, True),
        "anisotropic": (np.diag(np.geomspace(0.5, 2, d)), np.zeros(d), True),
        "drop_top_pc": (eye - np.outer(vt[0], vt[0]), np.zeros(d), False),
        "drop_night_contrast": (eye - np.outer(per_domain[1], per_domain[1]), np.zeros(d), False),
        "drop_day_contrast": (eye - np.outer(per_domain[0], per_domain[0]), np.zeros(d), False),
        "drop_support_orthogonal": (eye - np.outer(nuisance, nuisance), np.zeros(d), False),
    }


def control_scores(query, transformed_query, directions, affine):
    a, b = matrix(query), matrix(transformed_query)
    operator, shift, invertible = affine
    rows = []
    for h in directions:
        w = h["direction"]
        original = (a - h["midpoint"]) @ w
        if invertible:
            transported_w = np.linalg.solve(operator, w)
            score = (b - shift - h["midpoint"] @ operator) @ transported_w
        else:
            score = (b - h["midpoint"]) @ w
        error = float(np.max(abs(score - original)))
        if invertible or np.allclose(operator @ w, w, atol=1e-10):
            np.testing.assert_allclose(score, original, atol=1e-9, rtol=1e-9)
        rows.append({"draw": h["draw"], "domain": h["domain"], "max_reference_logit_error": error})
    return rows


def synthetic(seed=194044, n=256, d=96):
    """Known low-energy contrast; no claim that nuisance removal is known on real data."""
    rng = np.random.default_rng(seed)
    y = np.tile([-1.0, 1.0], n // 2)
    nuisance = rng.normal(size=(n, d - 1))
    nuisance -= nuisance.mean(0)
    nuisance -= np.outer(y, y @ nuisance / n)
    q, _ = np.linalg.qr(nuisance)
    z = np.c_[q * np.geomspace(4, 1, d - 1), 0.03 * y / np.sqrt(n)]
    day = np.zeros(d)
    day[-1] = 1
    night = day.copy()
    directions = [
        {
            "draw": 0,
            "domain": dom,
            "direction": w,
            "midpoint": np.zeros(d),
            "within_sd": 1.0,
            "separation": 0.06 / np.sqrt(n),
        }
        for dom, w in ((0, day), (1, night))
    ]
    return z, y, directions
