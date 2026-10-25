"""Fixed probe aggregation and finite-grid measurement variability."""
from __future__ import annotations

import numpy as np

BATCHES = (119, 123, 127)
MASK_OFFSETS = (195200, 195300, 195400)


def panels(delta, static):
    """Inputs state,batch,mask,pool,prior,axis and state,pool,prior,axis.

    No benefit targets are accepted. All29 settings are fixed, primary first.
    The reused-study52 anchor is freshly recomputed, never substituted.
    """
    x, h = np.asarray(delta, float), np.asarray(static, float)
    if x.shape != (72, 3, 3, 6, 2, 8) or h.shape != (72, 6, 2, 8):
        raise ValueError('Complete fixed 72-state probe grid required')
    if not np.isfinite(x).all() or not np.isfinite(h).all():
        raise ValueError('Finite profiles required')
    result = [('all9', x[..., 0, :].mean((1, 2, 3)), h[:, :, 0].mean(1)),
              ('anchor', x[:, 2, 0, :, 0].mean(1), h[:, :, 0].mean(1)),
              ('all9_weak', x[..., 1, :].mean((1, 2, 3)), h[:, :, 1].mean(1))]
    for b, m in np.ndindex(3, 3):
        if (b, m) != (2, 0):
            result.append((f'single_b{b}_m{m}', x[:, b, m, :, 0].mean(1), h[:, :, 0].mean(1)))
    for b in range(3):
        result.append((f'batch{b}', x[:, b, :, :, 0].mean((1, 2)), h[:, :, 0].mean(1)))
    for m in range(3):
        result.append((f'mask{m}', x[:, :, m, :, 0].mean((1, 2)), h[:, :, 0].mean(1)))
    for b in range(3):
        selected = np.take(x, np.flatnonzero(np.arange(3) != b), axis=1)
        result.append((f'drop_batch{b}', selected[..., 0, :].mean((1, 2, 3)), h[:, :, 0].mean(1)))
    for m in range(3):
        selected = np.take(x, np.flatnonzero(np.arange(3) != m), axis=2)
        result.append((f'drop_mask{m}', selected[..., 0, :].mean((1, 2, 3)), h[:, :, 0].mean(1)))
    for o in range(6):
        result.append((f'pool{o}', x[:, :, :, o, 0].mean((1, 2)), h[:, o, 0]))
    assert len(result) == 29
    assert all(r.shape == s.shape == (72, 8) for _, r, s in result)
    return result


def grid_variance(grid):
    """Exact balanced3x3 finite-grid ANOVA; no population/noise-variance claim."""
    x = np.asarray(grid, float)
    if x.shape[-3:-1] != (3, 3) or not np.isfinite(x).all():
        raise ValueError('Finite grid ending in batch,mask,axis required')
    mu = x.mean((-3, -2), keepdims=True)
    batch = x.mean(-2, keepdims=True)-mu
    mask = x.mean(-3, keepdims=True)-mu
    interaction = x-mu-batch-mask
    components = np.stack([((x-mu)**2).mean((-3, -2)), (batch**2).mean((-3, -2)),
                           (mask**2).mean((-3, -2)), (interaction**2).mean((-3, -2))], -1)
    np.testing.assert_allclose(components[..., 0], components[..., 1:].sum(-1), atol=1e-18, rtol=1e-10)
    return components
