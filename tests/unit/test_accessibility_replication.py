import numpy as np
import pytest

from cafl4ds.accessibility_replication import allocations, decompose, fit, metrics
from cafl4ds.conditioning_fitting import augment, objective


def test_roles_and_nested_draws():
    y = np.tile(np.repeat([0, 1], 50), 4)
    d = np.tile(np.repeat([0, 1], 100), 2)
    p = np.repeat(['val', 'train'], 200)
    a = allocations(y, d, p)
    b = allocations(y, d, p)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert len({tuple(x) for x in a['pools'][0]}) == 6
    for pop in range(2):
        for outer in range(6):
            pool = a['pools'][pop, outer]
            assert set(pool).isdisjoint(a[f'legacy_{pop}_{outer}'])
            for ids in a['draws'][pop, outer]:
                assert len(set(ids)) == 64 and set(ids) <= set(range(128))
                assert [sum((d[pool[ids]] == dd) & (y[pool[ids]] == c)) for dd in (0, 1) for c in (0, 1)] == [16]*4
    with pytest.raises(ValueError):
        allocations(y[:100], d[:100], p[:100])


def test_rotation_translation_has_zero_aligned_feature_contribution():
    rng = np.random.default_rng(48)
    bank, query = rng.normal(size=(150, 7)), rng.normal(size=(40, 7))
    r, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    t, head = rng.normal(size=7), rng.normal(size=8)
    after_head = np.r_[r.T @ head[:-1], head[-1] - t @ (r.T @ head[:-1])]
    total, movement, reader = decompose(query, query@r+t, head, after_head, bank, bank@r+t)
    for x in (total, movement, reader):
        np.testing.assert_allclose(x, 0, atol=1e-12)


def test_exact_accounting_with_distorted_features_and_refitting():
    rng = np.random.default_rng(49)
    a, b = rng.normal(size=(70, 7)), rng.normal(size=(70, 7))
    h, g = rng.normal(size=8), rng.normal(size=8)
    total, feature, reader = decompose(a, b, h, g, a, b)
    np.testing.assert_allclose(feature+reader, total, atol=1e-12)
    assert np.std(feature) > .1 and np.std(reader) > .1


def test_fit_and_class_metrics_independently():
    rng = np.random.default_rng(50)
    z = rng.normal(size=(128, 9)); y = np.tile(np.repeat([0, 1], 32), 2); d = np.repeat([0, 1], 64)
    h, receipt = fit(z, y, .01)
    assert receipt['gradient_inf'] <= 1e-10
    assert np.max(abs(objective(h, augment(z), y, np.eye(10), .01)[1])) <= 1e-10
    result = metrics(z, h, y, d)
    s = augment(z) @ h
    correct = (s > 0) == y
    np.testing.assert_allclose(result['balanced']['accuracy'], 100*correct.mean())
    np.testing.assert_allclose(result['balanced']['ce'], np.logaddexp(0, -(2*y-1)*s).mean())
