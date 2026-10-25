import numpy as np
from cafl4ds.dispersion_health import measure, directions, variances


def fixture():
    rng = np.random.default_rng(195700); y = np.tile(np.repeat([0, 1], 8), 2); d = np.repeat([0, 1], 16)
    x = rng.normal(size=(32, 96)); heads = rng.normal(size=(2, 97)); q = rng.normal(size=(18, 96)); q[:2] = heads[:, :-1]; q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x, heads, q, y, d


def test_uniform_scaling_and_translation_leave_directional_excess_zero():
    x, heads, q, y, d = fixture()
    b = np.stack([x*2, x*.5+3])
    c, _, _, _, _, floor = measure(x, b, q, heads, y, d)
    np.testing.assert_allclose(c[:, :, 0], np.array([np.log(2), -np.log(2)])[:, None]+np.zeros((2, 2)), atol=1e-14)
    np.testing.assert_allclose(c[:, :, 2:], 0, atol=1e-14); assert floor == 0


def test_variance_trace_and_projection_match_covariance_rayleigh_formula():
    x, _, q, y, d = fixture(); trace, projected = variances(x, q, y, d)
    for dom in (0, 1):
        cov = sum(np.cov(x[(y == c) & (d == dom)], rowvar=False, bias=True) for c in (0, 1))/2
        np.testing.assert_allclose(trace[dom], np.trace(cov), atol=1e-12)
        np.testing.assert_allclose(projected[dom], np.einsum('ij,jk,ik->i', q, cov, q), atol=1e-12)


def test_random_selection_cannot_read_evaluation_images():
    x, heads, _, y, d = fixture(); c = np.array([np.r_[0:4, 8:12, 16:20, 24:28], np.r_[4:8, 12:16, 20:24, 28:32]])
    # Add extra evaluation rows; calibration IDs do not include them.
    a = np.r_[x, x+100]; yy = np.r_[y, y]; dd = np.r_[d, d]
    first = directions(a, heads, c, yy, dd, 0, 0)
    a[32:] *= -1000
    second = directions(a, heads, c, yy, dd, 0, 0)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
