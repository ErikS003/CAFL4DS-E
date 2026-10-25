import numpy as np
from cafl4ds.reader_precision import agreement, nested_roles, paired_arrays


def test_nested_roles_are_stratified_disjoint_and_exact_unions():
    y = np.tile(np.repeat([0, 1], 50), 2); d = np.repeat([0, 1], 100)
    c, sub, e, small, _ = nested_roles(y, d)
    for draw, g in np.ndindex(6, 2):
        assert set(sub[draw, g, 0]) | set(sub[draw, g, 1]) == set(c[draw, g])
        assert not set(sub[draw, g, 0]) & set(sub[draw, g, 1])
        assert set(small[draw, g, 0]) | set(small[draw, g, 1]) == set(e[draw, g])
        assert not set(small[draw, g, 0]) & set(small[draw, g, 1])
        for ids in (*sub[draw, g], *small[draw, g]):
            assert all(np.sum((y[ids] == a) & (d[ids] == b)) == len(ids)//4 for a, b in np.ndindex(2, 2))


def test_concordance_rejects_shrinkage_and_bias_even_with_perfect_correlation():
    x = np.arange(12.).reshape(6, 1, 2)
    exact = agreement([(x, x)]); shrink = agreement([(x, x*.1)]); bias = agreement([(x, x+4)])
    assert exact['concordance'] == 1 and exact['paired_MSE'] == 0
    assert shrink['median_Pearson_r'] > .999 and shrink['concordance'] < .2
    assert bias['concordance'] < 1 and bias['paired_MSE'] == 16
    assert agreement([(x*0, x*0)])['concordance'] is None


def test_measurement_errors_averaged_after_pairing_and_all_switches_exposed():
    grid = np.arange(24.).reshape(3, 2, 2, 2)
    a, b = paired_arrays(grid, 0); assert np.array_equal(a, grid[:, :, 0])
    a, b = paired_arrays(grid, 2); assert np.array_equal(a[..., 0], grid[:, :, 0, 0])
    assert np.array_equal(b[..., 0], grid[:, :, 1, 1])
    x = np.arange(6.).reshape(3, 1, 2)
    result = agreement([(x, x+2), (x, x-2)])
    assert result['paired_MSE'] == 4  # averaging measurements first would give zero
