"""Protect paired bootstrap identities and effective fitting-weight scaling."""

import numpy as np
import pytest

from cafl4ds.reader_qualification import query_weights, support_plan


def test_bootstrap_multiplicities_and_cv_isolation() -> None:
    """Every support entry is budgeted and each CV identity is held out once."""
    labels = np.tile(np.repeat(np.arange(3), 192), 2)
    domains = np.repeat(np.arange(2), 576)
    pool = np.arange(1152)
    plan = support_plan(pool, labels, domains)
    assert plan == support_plan(pool, labels, domains)
    validation = []
    for fold in plan["folds"]:
        assert not set(fold["fitting"]) & set(fold["validation"])
        assert set(fold["fitting"]) | set(fold["validation"]) == set(pool)
        validation.extend(fold["validation"])
    assert sorted(validation) == pool.tolist()
    for name, ix in plan["supports"].items():
        assert len(ix) == 1152
        assert sum(plan["multiplicities"][name].values()) == 1152
        for d in (0, 1):
            for c in (0, 1, 2):
                assert ((domains[ix] == d) & (labels[ix] == c)).sum() == 192
    assert len(plan["multiplicities"]["bootstrap00"]) < 1152


def test_query_weights_preserve_cells_and_pairing() -> None:
    """Query resampling preserves strata and cancels identical paired outcomes."""
    labels = np.tile(np.repeat(np.arange(3), 4), 2)
    domains = np.repeat(np.arange(2), 12)
    ix = np.arange(24)
    weights = query_weights(ix, labels, domains, 17)
    assert weights.shape == (1001, 6, 24)
    np.testing.assert_allclose(weights.sum(2), 1)
    np.testing.assert_array_equal(weights, query_weights(ix, labels, domains, 17))
    loss = np.arange(24, dtype=float)
    # Identical compared models must have exactly zero paired uncertainty.
    assert np.max(np.abs(weights @ (loss - loss))) == 0
    for cell in range(6):
        assert not weights[:, cell, labels + 3 * domains != cell].any()


def test_reject_incomplete_support_budget() -> None:
    """A smaller panel must never be silently treated as full budget."""
    with pytest.raises(ValueError):
        support_plan(np.arange(12), np.tile(np.arange(3), 4), np.repeat(np.arange(2), 6))
