import numpy as np

from cafl4ds.functional_response_profile import response
from cafl4ds.reader_bank_cross import allocations, crossed_variance, descriptors, screen_errors, sigma


def metadata():
    return np.tile(np.repeat([0, 1], 50), 2), np.repeat([0, 1], 100)


def test_disjoint_stratified_roles_and_no_feature_dependent_selection():
    y, d = metadata(); c, e, unused = allocations(y, d)
    assert c.shape == (6, 2, 64) and e.shape == (6, 2, 32)
    for j in range(6):
        assert len(set(np.r_[c[j].ravel(), e[j].ravel(), unused[j]])) == 200
        for ids in c[j]:
            assert all(np.sum((y[ids] == label) & (d[ids] == dom)) == 16 for dom in (0, 1) for label in (0, 1))
        for ids in e[j]:
            assert all(np.sum((y[ids] == label) & (d[ids] == dom)) == 8 for dom in (0, 1) for label in (0, 1))


def test_native_response_and_fixed_calibration_scale():
    y, d = metadata(); _, e, _ = allocations(y, d); ids = e[0, 0]
    rng = np.random.default_rng(55); a = rng.normal(size=32); b = a+rng.normal(0, .2, size=(3, 32))
    cs = 2*sigma(a, y[ids], d[ids]); value, static = descriptors(a, b, y[ids], d[ids], cs)
    for j in range(3):
        np.testing.assert_allclose(value[j, 0], response(a, b[j], y[ids], d[ids]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(value[:, 1, [0, 1, 4, 5]], value[:, 0, [0, 1, 4, 5]]/2)
    np.testing.assert_array_equal(value[:, 1, [2, 6]], value[:, 0, [2, 6]])
    np.testing.assert_allclose(static[1, [0, 1, 4, 5]], static[0, [0, 1, 4, 5]]/2)


def test_crossed_factors_and_error_aggregation_before_screen():
    r = np.array([-1., 1.])[:, None]; e = np.array([-2., 2.])[None, :]
    result = crossed_variance((r+e+r*e)[..., None])[0]
    np.testing.assert_allclose(result, [9, 1, 4, 4])
    groups = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3, 4) for _ in range(2)])
    errors = np.tile([2., 2., 2., 1.], (72, 1)); assert screen_errors(errors, groups)['development_screen_passed']
    assert not screen_errors(np.zeros_like(errors), groups)['development_screen_passed']
