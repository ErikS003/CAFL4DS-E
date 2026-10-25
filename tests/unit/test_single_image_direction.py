import numpy as np

from cafl4ds.single_image_direction import balanced_threshold, event_counts, forecasts, paired_intervals, score


def settings():
    return dict(
        prefix_end=41,
        inner_end=21,
        test_start=50,
        period=30,
        families=["logistic", "boosting"],
        threshold_min=0.05,
        threshold_max=0.95,
        threshold_step=0.05,
        estimators={
            "logistic": dict(_target_="sklearn.linear_model.LogisticRegression", C=1, max_iter=1000),
            "boosting": dict(
                _target_="sklearn.ensemble.HistGradientBoostingClassifier",
                max_iter=10,
                min_samples_leaf=3,
                early_stopping=False,
            ),
        },
    )


def test_view_counts_do_not_turn_mixed_parent_into_mean_sign():
    delta = np.array([[-2, -2, -2, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=float)
    np.testing.assert_array_equal(event_counts(delta), [[3, 3], [0, 0]])


def test_scores_equal_explicit_single_view_outcomes():
    counts = np.array([[4, 2], [1, 5], [3, 3]])
    p = np.array([0.2, 0.8, 0.6])
    labels = np.concatenate([np.r_[np.zeros(n), np.ones(k)] for n, k in counts])
    expanded = np.repeat(p, 6)
    s = score(counts, p)
    np.testing.assert_allclose(s["brier"], ((expanded - labels) ** 2).mean())
    decision = expanded >= 0.5
    np.testing.assert_allclose(
        s["balanced_accuracy"], 0.5 * ((decision[labels == 1]).mean() + (~decision[labels == 0]).mean())
    )


def test_future_and_other_seed_labels_cannot_change_earlier_fit():
    rng = np.random.default_rng(2)
    origins = np.repeat(np.arange(0, 100, 10), 2)
    seeds = np.tile([1, 2], 10)
    x = rng.normal(size=(20, 3))
    counts = np.column_stack((np.tile([2, 4], 10), np.tile([4, 2], 10)))
    a, s, _ = forecasts({"state": x, "image": np.c_[x, x[:, 0] ** 2]}, counts, seeds, origins, settings())
    changed = counts.copy()
    changed[origins >= 50] = changed[origins >= 50, ::-1]
    b, t, _ = forecasts({"state": x, "image": np.c_[x, x[:, 0] ** 2]}, changed, seeds, origins, settings())
    assert s == t
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_balanced_threshold_is_not_forced_to_majority_decision():
    counts = np.array([[5, 1], [5, 1], [1, 5], [1, 5]])
    threshold = balanced_threshold(counts, np.array([0.1, 0.2, 0.3, 0.4]), settings())
    assert 0.2 < threshold <= 0.3


def test_bootstrap_resamples_source_histories_with_all_views():
    counts = np.tile([[4, 2], [2, 4]], (3, 1))
    groups = np.repeat(["a", "b", "c"], 2)
    ci = paired_intervals(counts, np.tile([0.2, 0.8], 3), 0.5, np.full(6, 0.5), 0.5, groups, 200, 4)
    assert ci["brier_difference"][1] < 0
    assert ci["balanced_accuracy_difference"][0] > 0
