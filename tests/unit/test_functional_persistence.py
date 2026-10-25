import numpy as np
import pytest

from cafl4ds.functional_persistence import contrast, measure, support_roles, transfer_summary
from cafl4ds.functional_response_profile import response


def fixture():
    rng = np.random.default_rng(54)
    labels = np.tile(np.repeat([0, 1], 50), 2); domains = np.repeat([0, 1], 100)
    pools = np.tile(np.concatenate([np.arange(i*50, i*50+32) for i in range(4)]), (6, 1))
    folds = np.tile(np.tile(np.arange(32)%4, 4), (6, 1))
    return rng, labels, domains, pools, folds


def test_roles_globally_disjoint_and_balanced():
    _, y, d, pools, _ = fixture(); roles = support_roles(y, d)
    for dom in (0, 1):
        for label in (0, 1):
            assert np.sum((y == label) & (d == dom) & (roles == 1)) == 25
    assert not set(np.flatnonzero(roles == 1)) & set(np.flatnonzero(roles == 2))
    assert all(np.sum(roles[p] == 1) == 64 for p in pools)


def test_geometry_response_fixed_baseline_and_class_shift():
    rng, y, d, pools, folds = fixture(); roles = support_roles(y, d)
    before = rng.normal(size=(200, 96)).astype(np.float32)
    heads = np.zeros((6, 2, 4, 97)); heads[..., 0] = 1
    after = np.tile(before, (2, 5, 1, 1)); after[1, :, :, 0] += 2
    measured, static = measure(before, after, heads, pools, y, d, folds, roles)
    assert measured.shape == (2, 5, 6, 2, 3, 8) and static.shape == (6, 2, 3, 8)
    np.testing.assert_array_equal(measured[0], np.zeros_like(measured[0]))
    ids = pools[0]; keep = roles[ids] == 2
    np.testing.assert_array_equal(measured[1, 0, 0, 0, 2], response(before[ids[keep], 0], after[1, 0, ids[keep], 0], y[ids[keep]], d[ids[keep]]))
    assert measured[1, 0, 0, 0, 0, 5] > 1 and abs(measured[1, 0, 0, 0, 0, 4]) < 1e-7


def test_horizon_contrast_and_transfer_grouping():
    a = np.zeros((72, 2, 5, 6, 2, 3, 8)); a[:, 1] = np.arange(5)[None, :, None, None, None, None]
    c = contrast(a); assert c.shape == (72, 5, 6, 2, 3, 8)
    np.testing.assert_array_equal(c[:, 4], np.full((72, 6, 2, 3, 8), 4.))
    groups = np.array([(s, b, f) for s in (1, 2, 3) for b in (1, 2, 3) for f in (1, 2, 3, 4) for _ in range(2)])
    x = np.arange(72)[:, None]*np.ones((1, 8)); y = np.tile(x[:, None], (1, 5, 1))
    rows = transfer_summary(x, y, groups, np.tile([0, 1], 36))
    assert len(rows) == 13*5*8 and all(abs(v['r']-1) < 1e-12 for v in rows)
    with pytest.raises(ValueError):
        contrast(a[:1])
