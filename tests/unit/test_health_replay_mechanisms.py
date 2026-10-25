"""Check eligibility, matched diets, factorial signs and true source exclusion."""

import numpy as np
import pytest

from cafl4ds.health_replay_mechanisms import exposures, factorial, held_source_ridge, pools, schedule


def test_replay_exposures_count_as_previously_trained():
    trace = [{"diet": {"current_ids": [0, 2], "replay_ids": [1, 2]}}]
    np.testing.assert_array_equal(exposures(trace, 4), [1, 1, 2, 0])


def test_pool_eligibility_and_nested_coverage():
    count = np.arange(8192) % 2
    result = pools(count, 7)
    assert result == pools(count, 7)
    for name, ids in result.items():
        age, status, size = name.split("_")
        assert len(ids) == len(set(ids)) == int(size)
        assert np.all(np.asarray(ids) < 2048) if age == "old" else np.all(np.asarray(ids) >= 6144)
        assert np.all(count[ids] > 0) if status == "trained" else np.all(count[ids] == 0)
        assert result[f"{age}_{status}_32"] == result[f"{age}_{status}_256"][:32]


def test_insufficient_pool_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        pools(np.ones(8192), 7)


def test_matched_current_and_pool_slots():
    a, b = schedule(list(range(256))), schedule(list(range(1000, 1256)))
    for left, right in zip(a, b, strict=True):
        assert left["current"] == right["current"]
        assert left["past"] == [x - 1000 for x in right["past"]]
        assert len(set(left["past"])) == len(left["past"]) == 16
    assert all(r["current"] == list(range(32)) and not r["past"] for r in schedule(None))


def test_factorial_orthogonal_effects():
    a = np.zeros((3, 3, 2, 2, 2, 1))
    a[:, :, 0] += 2
    a[:, :, :, 1] += 3
    a[:, :, :, :, 1] += 5
    result = factorial(a)
    for name, value in (("old_minus_recent", 2), ("unused_minus_trained", 3), ("pool256_minus32", 5)):
        np.testing.assert_array_equal(result[name], np.full((3, 1), value))


def test_held_group_targets_cannot_change_own_prediction():
    x = np.arange(36).reshape(12, 3).astype(float)
    y = np.arange(12).astype(float)
    group = np.repeat([1, 2, 3], 4)
    a, folds = held_source_ridge(x, y, group)
    changed = y.copy()
    changed[group == 2] += 10000
    b, _ = held_source_ridge(x, changed, group)
    np.testing.assert_array_equal(a[group == 2], b[group == 2])
    for fold in folds:
        assert not set(group[fold["training_rows"]]) & {fold["held"]}
        np.testing.assert_array_equal(fold["center"], x[fold["training_rows"]].mean(0))
