import numpy as np

from cafl4ds.functional_response_profile import designs, forecast, response, screen


def test_response_controls_and_frozen_boundary():
    y = np.tile(np.repeat([0, 1], 16), 2); d = np.repeat([0, 1], 32)
    a = np.tile(np.r_[np.linspace(-2, 0, 16), np.linspace(0, 2, 16)], 2)
    np.testing.assert_array_equal(response(a, a, y, d), np.zeros(8))
    shifted = response(a, a+1, y, d).reshape(2, 4)
    np.testing.assert_allclose(shifted[:, [0, 2]], 0, atol=1e-15)
    assert np.all(shifted[:, 1] > 0)
    separation = response(a, a+.1*(2*y-1), y, d).reshape(2, 4)
    assert np.all(separation[:, 0] > 0)
    np.testing.assert_allclose(separation[:, 1], 0, atol=1e-15)


def test_forecast_excludes_three_axes_and_reproduces_static():
    from cafl4ds.functional_health_profile import forecast as previous
    rng = np.random.default_rng(52)
    g = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3, 4) for _ in range(2)])
    diet = np.tile([0, 1], 36); signals = rng.normal(size=(72, 16))
    h, r, target = rng.normal(size=(72, 8)), rng.normal(size=(72, 8)), rng.normal(size=(72, 14))
    p, folds, error = forecast(signals, diet, g, h, r, target)
    old, _, _ = previous(signals, diet, g, h, target)
    np.testing.assert_array_equal(p['cheap'], old['cheap'])
    np.testing.assert_array_equal(p['static'], old['functional_added'])
    assert len(folds) == 36 and error < 1e-10
    assert [x.shape[1] for x in designs(signals, diet, h, r).values()] == [0, 7, 15, 9, 15, 23, 17]


def test_screen_must_beat_both_baselines():
    g = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3, 4) for _ in range(2)])
    p = {'cheap': np.ones(72), 'static': np.ones(72)*2, 'static_response_added': np.zeros(72)}
    assert screen(np.zeros(72), p, g)['development_screen_passed']
    p['static_response_added'][:] = 1.1
    assert not screen(np.zeros(72), p, g)['development_screen_passed']
