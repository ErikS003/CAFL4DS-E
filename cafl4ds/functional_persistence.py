"""Fixed-reader geometric trajectory targets and disjoint score-evaluation roles."""
from __future__ import annotations

import numpy as np

from cafl4ds.functional_health_profile import profile
from cafl4ds.functional_response_profile import response

HORIZONS = (1, 8, 32, 128, 384)
ROLES = ('full', 'A', 'B')


def support_roles(labels, domains):
    y, d = np.asarray(labels)[:200], np.asarray(domains)[:200]
    roles = np.zeros(200, dtype=np.int64)
    for domain in (0, 1):
        for label in (0, 1):
            ids = np.flatnonzero((d == domain) & (y == label))
            if len(ids) != 50:
                raise ValueError('Exactly50 support images per class/domain required')
            roles[ids[::2]] = 1; roles[ids[1::2]] = 2
    assert np.all(roles > 0) and np.sum(roles == 1) == np.sum(roles == 2) == 100
    return roles


def measure(before, after, heads, pools, labels, domains, folds, roles):
    """after [...,200,96]; outputs [...,pool,prior,role,axis], static analogous.

    Heads remain the original96-label OOF heads. Disjoint roles separate score
    evaluation images globally, but do not independently fit different readers.
    All response denominators/thresholds use the relevant role's BASELINE scores.
    """
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != (200, 96) or after.shape[-2:] != before.shape:
        raise ValueError('Aligned support-only features required')
    result = np.empty((*after.shape[:-2], 6, 2, 3, 8)); static = np.empty((6, 2, 3, 8))
    for o, ids in enumerate(pools):
        y, d = labels[ids], domains[ids]
        for k in range(2):
            a = np.empty(128); b = np.empty((*after.shape[:-2], 128))
            for f in range(4):
                te = folds[o] == f; beta = heads[o, k, f]
                a[te] = before[ids[te]].astype(float)@beta[:-1]+beta[-1]
                b[..., te] = after[..., ids[te], :].astype(float)@beta[:-1]+beta[-1]
            for role in range(3):
                keep = np.ones(128, bool) if role == 0 else roles[ids] == role
                static[o, k, role] = profile(a[keep], y[keep], d[keep])
                for view in np.ndindex(after.shape[:-2]):
                    result[(*view, o, k, role)] = response(a[keep], b[view][keep], y[keep], d[keep])
    assert np.isfinite(result).all() and np.isfinite(static).all()
    return result, static


def contrast(actions):
    """State,action,horizon,pool,prior,role,axis -> replay-minus-current."""
    a = np.asarray(actions)
    if a.shape != (72, 2, 5, 6, 2, 3, 8) or not np.isfinite(a).all():
        raise ValueError('Complete72-state five-horizon trajectory required')
    return a[:, 1]-a[:, 0]


def transfer_summary(local, target, groups, diets):
    """Descriptive same-axis correlations/signs, no fitted causal interpretation."""
    x, y = np.asarray(local), np.asarray(target)
    if x.shape != (72, 8) or y.shape != (72, 5, 8):
        raise ValueError('Aligned pre-action and horizon descriptors required')
    subsets = [('all', 0, np.ones(72, bool))]
    for axis, name in enumerate(('seed', 'source', 'context')):
        subsets += [(name, int(v), groups[:, axis] == v) for v in np.unique(groups[:, axis])]
    subsets += [('diet', int(v), diets == v) for v in (0, 1)]
    rows = []
    for subset, value, keep in subsets:
        for t, horizon in enumerate(HORIZONS):
            for k in range(8):
                a, b = x[keep, k], y[keep, t, k]
                r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-15 and b.std() > 1e-15 else None
                nonzero = (a != 0) & (b != 0)
                rows.append({'subset': subset, 'value': value, 'horizon': horizon, 'axis': k,
                    'r': r, 'nonzero_pairs': int(nonzero.sum()),
                    'nonzero_sign_agreement': float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else None,
                    'target_mean': float(b.mean()), 'target_SD': float(b.std()), 'target_mean_abs': float(abs(b).mean())})
    return rows
