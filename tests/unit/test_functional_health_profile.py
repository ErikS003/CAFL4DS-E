import numpy as np
import pytest

from cafl4ds.functional_health_profile import designs, forecast, profile, screen


def fixture():
    groups = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3)
                       for c in (1, 2, 3, 4) for _ in range(2)])
    rng = np.random.default_rng(51)
    return rng, groups, np.tile([0, 1], 36)


def test_profile_translation_and_positive_scaling():
    y = np.tile(np.repeat([0, 1], 16), 2)
    d = np.repeat([0, 1], 32)
    s = np.tile(np.r_[np.linspace(-2, 0, 16), np.linspace(0, 2, 16)], 2)
    h = profile(s, y, d).reshape(2, 4)
    scaled = profile(3*s, y, d).reshape(2, 4)
    np.testing.assert_allclose(h[:, [0, 1, 3]], scaled[:, [0, 1, 3]])
    np.testing.assert_allclose(scaled[:, 2], h[:, 2]+np.log(3))
    shifted = profile(s+1, y, d).reshape(2, 4)
    np.testing.assert_allclose(h[:, [0, 2]], shifted[:, [0, 2]])
    assert np.all(shifted[:, 1] > h[:, 1])


def test_degenerate_profile_and_invalid_metadata():
    y = np.tile([0, 0, 1, 1], 2); d = np.repeat([0, 1], 4)
    assert np.isfinite(profile(np.zeros(8), y, d)).all()
    with pytest.raises(ValueError):
        profile(np.zeros(3), y, d)


def test_exact_group_exclusion_and_independent_ridge():
    rng, g, diets = fixture()
    signals, h = rng.normal(size=(72, 16)), rng.normal(size=(72, 8))
    target = rng.normal(size=(72, 14))
    pred, folds, error = forecast(signals, diets, g, h, target)
    assert len(folds) == 36 and error < 1e-10
    for f in folds:
        assert all(not set(g[f['train'], k]) & set(g[f['test'], k]) for k in range(3))
    # Mutating targets of every excluded row cannot change that fold's forecast.
    f = folds[0]; t = target.copy(); t[f['test']] += 1000
    other, _, _ = forecast(signals, diets, g, h, t)
    for m in pred:
        np.testing.assert_array_equal(pred[m][f['test']], other[m][f['test']])
    assert [x.shape[1] for x in designs(signals, diets, h).values()] == [0, 1, 7, 9, 9, 15, 17, 23]


def test_screen_requires_every_seed_and_source():
    _, g, _ = fixture(); target = np.zeros(72)
    p = {'cheap': np.ones(72), 'functional_added': np.zeros(72)}
    assert screen(target, p, g)['development_screen_passed']
    p['functional_added'][g[:, 0] == 1] = 1.1
    assert not screen(target, p, g)['development_screen_passed']
