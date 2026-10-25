"""Test nesting, strict two-axis exclusion and causal feature windows."""

import numpy as np
import pytest

from cafl4ds.domain_health_calibration import dynamics, excluded_ridge, schedule


def test_doses_have_equal_exposure_and_nested_identities():
    pool = list(range(256))
    rows = {arm: schedule(pool, arm, 12) for arm in ("current", "balanced", "history")}
    for t in range(12):
        for arm in rows:
            a = rows[arm][t]
            assert len(a["current"]) + len(a["past"]) == 32
            assert len(set(a["past"])) == len(a["past"])
        assert set(rows["balanced"][t]["current"]) <= set(rows["current"][t]["current"])
        assert set(rows["balanced"][t]["past"]) <= set(rows["history"][t]["past"])


def test_excluding_both_axes_prevents_target_leakage():
    sources = np.repeat([1, 2, 3], 12)
    contexts = np.tile(np.repeat([1, 2], 6), 3)
    x = np.random.default_rng(3).normal(size=(36, 5))
    y = x[:, :3].copy()
    a, folds = excluded_ridge(x, y, sources, contexts)
    changed = y.copy()
    changed[(sources == 1) | (contexts == 2)] += 10000
    b, _ = excluded_ridge(x, changed, sources, contexts)
    np.testing.assert_array_equal(a[(sources == 1) & (contexts == 2)], b[(sources == 1) & (contexts == 2)])
    for f in folds:
        assert len(f["train"]) == 12 and len(f["test"]) == 6
        assert not (sources[f["train"]] == f["source"]).any()
        assert not (contexts[f["train"]] == f["context"]).any()


def test_future_training_does_not_enter_early_features():
    trace = [{"response": [i, 2 * i]} for i in range(32)]
    a = dynamics(trace, 16)
    trace[16]["response"] = [10000, 10000]
    np.testing.assert_array_equal(a, dynamics(trace, 16))
    np.testing.assert_array_equal(a, [13.5, 27, 12, 24])
    with pytest.raises(ValueError):
        dynamics(trace, 3)
