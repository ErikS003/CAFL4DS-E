"""Repeated support allocations and exact, bank-aligned margin accounting."""
from __future__ import annotations

import numpy as np

from cafl4ds.conditioning_fitting import augment, cell_metrics, newton

POPULATIONS = ('val', 'train')
OUTER = (194048, 194049, 194051, 194053, 194057, 194059)
LAMBDAS = (.01, .001)
SLOTS = ('64_draw0', '64_draw1', '64_draw2', '128_full')
QUERIES = ('broad_val', 'fresh_train', 'legacy_val', 'legacy_train')
MEASURES = ('accuracy', 'ce', 'margin_mean')


def allocations(labels, domains, partitions):
    labels, domains, partitions = map(np.asarray, (labels, domains, partitions))
    if not labels.shape == domains.shape == partitions.shape:
        raise ValueError('Aligned metadata required')
    pools = np.empty((2, 6, 128), np.int64)
    draws = np.empty((2, 6, 3, 64), np.int64)
    tests = {}
    for p, population in enumerate(POPULATIONS):
        for o, seed in enumerate(OUTER):
            rng = np.random.default_rng(seed + 1000*p)
            cells = [rng.permutation(np.flatnonzero((partitions == population) & (domains == d) & (labels == c)))
                     for d in (0, 1) for c in (0, 1)]
            if any(len(c) < 50 for c in cells):
                raise ValueError('At least 50 images per support reservoir cell')
            pools[p, o] = np.concatenate([c[:32] for c in cells])
            tests[f'legacy_{p}_{o}'] = np.concatenate([c[32:] for c in cells])
            for inner in range(3):
                r = np.random.default_rng(seed + 10000*p + 100*inner)
                draws[p, o, inner] = np.concatenate([32*k + r.permutation(32)[:16] for k in range(4)])
    return {'pools': pools, 'draws': draws, **tests}


def fit(z, labels, lam):
    x = augment(z)
    return newton(x, labels, np.eye(x.shape[1]), lam, tolerance=1e-10, max_steps=100)


def metrics(z, head, labels, domains):
    return cell_metrics(augment(z) @ head, labels, domains)


def alignment(before, after):
    """Orthogonal Procrustes of centered AFTER bank onto BEFORE; no query labels."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    ma, mb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b-mb).T @ (a-ma), full_matrices=False)
    rotation = u @ vt
    translation = ma-mb@rotation
    return rotation, translation


def decompose(before, after, head_before, head_after, bank_before, bank_after):
    """Exact symmetric logit change: feature movement plus refitted reader movement.

    AFTER features are aligned to BEFORE using an unlabeled fixed bank. The AFTER
    head is transported to preserve its logits. This is coordinate-dependent
    descriptive accounting, not a causal split or a claim of lost information.
    """
    r, t = alignment(bank_before, bank_after)
    za, zb = np.asarray(before, float), np.asarray(after, float) @ r + t
    ba = np.asarray(head_before, float)
    bb = np.r_[r.T @ head_after[:-1], head_after[-1] - t @ (r.T @ head_after[:-1])]
    movement = (zb-za) @ ((ba[:-1]+bb[:-1])/2)
    reader = ((za+zb)/2) @ (bb[:-1]-ba[:-1]) + bb[-1]-ba[-1]
    total = augment(after) @ head_after - augment(before) @ head_before
    np.testing.assert_allclose(total, movement+reader, rtol=1e-10, atol=1e-10)
    return total, movement, reader
