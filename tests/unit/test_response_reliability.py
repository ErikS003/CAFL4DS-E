import numpy as np
import pytest

from cafl4ds.response_reliability import grid_variance, panels


def test_factorial_variance_components_and_interaction():
    batch = np.array([-1., 0., 1.])[:, None]; mask = np.array([-2., 0., 2.])[None, :]
    x = batch+mask+batch*mask
    v = grid_variance(x[..., None])[0]
    np.testing.assert_allclose(v, [v[1:].sum(), 2/3, 8/3, 16/9])
    np.testing.assert_array_equal(grid_variance(np.ones((3, 3, 2))), np.zeros((2, 4)))


def test_panels_primary_anchor_and_deletions():
    rng = np.random.default_rng(53); x = rng.normal(size=(72, 3, 3, 6, 2, 8)); h = rng.normal(size=(72, 6, 2, 8))
    p = {name: (r, s) for name, r, s in panels(x, h)}
    assert len(p) == 29
    assert all(r.shape == s.shape == (72, 8) for r, s in p.values())
    np.testing.assert_array_equal(p['all9'][0], x[..., 0, :].mean((1, 2, 3)))
    np.testing.assert_array_equal(p['anchor'][0], x[:, 2, 0, :, 0].mean(1))
    changed = x.copy(); changed[:, 0] += 1e3
    pp = {name: r for name, r, _ in panels(changed, h)}
    np.testing.assert_array_equal(p['drop_batch0'][0], pp['drop_batch0'])
    changed = x.copy(); changed[:, :, 1] += 1e3
    pp = {name: r for name, r, _ in panels(changed, h)}
    np.testing.assert_array_equal(p['drop_mask1'][0], pp['drop_mask1'])
    with pytest.raises(ValueError):
        panels(x[:1], h[:1])


def test_new_grid_matches_original_anchor_arithmetic_exactly():
    from cafl4ds.functional_response_profile import response
    from scripts.run_response_reliability import measure
    rng = np.random.default_rng(195353)
    before = rng.normal(size=(200, 96)).astype(np.float32)
    after = rng.normal(size=(3, 3, 2, 200, 96)).astype(np.float32)
    heads = rng.normal(size=(6, 2, 4, 97))
    labels = np.tile(np.repeat([0, 1], 50), 2); domains = np.repeat([0, 1], 100)
    ids = np.concatenate([np.arange(50*c, 50*c+32) for c in range(4)])
    pools = np.tile(ids, (6, 1)); folds = np.tile(np.tile(np.arange(32)%4, 4), (6, 1))
    value = measure(before, after, heads, pools, labels, domains, folds)[2, 0]
    legacy = np.stack([after[2, 0], after[2, 0]], axis=1)
    expected = np.empty((2, 6, 2, 8))
    for o in range(6):
        for k in range(2):
            a = np.empty(128); b = np.empty((2, 2, 128))
            for f in range(4):
                te = folds[o] == f; beta = heads[o, k, f]
                a[te] = before[ids[te]].astype(float)@beta[:-1]+beta[-1]
                b[:, :, te] = legacy[:, :, ids[te]].astype(float)@beta[:-1]+beta[-1]
            for action in range(2):
                expected[action, o, k] = response(a, b[action, 1], labels[ids], domains[ids])
    np.testing.assert_array_equal(value, expected)
