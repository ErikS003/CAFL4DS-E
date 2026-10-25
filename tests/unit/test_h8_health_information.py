"""History leakage, matched shams and measured-health interpretation."""

import numpy as np
import pytest

from cafl4ds.h8_health_information import H8HealthInformation, geometry_features, point_admission, training_shuffle


def example():
    rng = np.random.default_rng(33)
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], 3)
    contexts = np.tile(["day_end", "night_end", "return_early"], 6)
    return rng.normal(size=(18, 6)), rng.normal(size=(18, 2, 15)), rng.normal(size=(18, 2)), groups, contexts


def test_held_outcome_never_changes_its_own_prediction_or_normalization():
    x, h, y, groups, contexts = example()
    study = H8HealthInformation(sham_count=2)
    before = study.crossfit(x, h, {}, y, groups, contexts)
    changed = y.copy()
    changed[groups == "a"] += 10000
    after = study.crossfit(x, h, {}, changed, groups, contexts)
    for name in before["predictions"]:
        np.testing.assert_array_equal(
            before["predictions"][name][groups == "a"], after["predictions"][name][groups == "a"]
        )
        assert not np.any(before["operators"][name][np.ix_(groups == "a", groups == "a")])
    model = before["folds"]["a"]["models"]["controls"]["actual"]
    np.testing.assert_allclose(model["center"], x[groups != "a"].mean(0))


def test_sham_preserves_context_profiles_without_held_history_donors():
    _, _, _, groups, contexts = example()
    order = training_shuffle(groups, contexts, "a", 900)
    train = groups != "a"
    assert np.all(groups[order[train]] != "a")
    assert np.all(groups[order[train]] != groups[train])
    np.testing.assert_array_equal(contexts[order], contexts)
    np.testing.assert_array_equal(order[~train], np.flatnonzero(~train))
    for g in set(groups[train]):
        assert len(set(groups[order[groups == g]])) == 1


def test_invariant_geometry_components_and_basis_sensitive_covariance():
    rng = np.random.default_rng(14)
    warm = rng.normal(size=(32, 8))
    z = warm + rng.normal(size=warm.shape) * 0.1
    domains = np.repeat([0, 1], 16)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    a, b = geometry_features(z, warm, domains), geometry_features(z @ q, warm @ q, domains)
    invariant = [i for i in range(15) if i % 5 != 2]
    np.testing.assert_allclose(a[invariant], b[invariant], atol=1e-10)
    # Off-diagonal covariance depends on the feature coordinate basis by definition.
    assert np.isfinite(a).all() and not np.allclose(a[[2, 7, 12]], b[[2, 7, 12]])


def test_any_domain_budget_blocks_point_acceptance_despite_balanced_gain():
    np.testing.assert_array_equal(point_admission([[4, -2], [1, 0], [0, 0]]), [False, True, False])


def test_invalid_roster_and_nonfinite_features_rejected():
    x, h, y, groups, contexts = example()
    with pytest.raises(ValueError, match="Duplicate"):
        H8HealthInformation().crossfit(x, h, {}, y, groups, np.repeat("same", 18))
    h[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        H8HealthInformation().crossfit(x, h, {}, y, groups, contexts)


def test_failed_target_cannot_be_promoted_by_regression_scores():
    x, h, y, groups, contexts = example()
    study = H8HealthInformation(sham_count=2)
    fitted = study.crossfit(x, h, {}, y, groups, contexts)
    rows = [{"useful": True, "harmful": False} for _ in y]
    summary = study.summarize(y, groups, fitted, np.tile(y, (10, 1, 1)), rows, False)
    assert not summary["bridge_pass"]
    assert not summary["filter_qualified"]
    assert not summary["functional_negative_controls_eligible"]
