import numpy as np
import pytest

from cafl4ds.replay_opportunity import analysis, bootstrap, bootstrap_counts, choices, metrics, query_halves, transfer


def test_clip_split_is_deterministic_and_never_splits_a_clip():
    clips = np.array([f"d{d}l{l}c{c}" for d in (0, 1) for l in (0, 1) for c in range(5) for _ in range(2)])
    labels = np.repeat([0, 1, 0, 1], 10)
    domains = np.repeat([0, 1], 20)
    a, b = query_halves(clips, labels, domains)
    assert not (a & b).any() and (a | b).all()
    assert not set(clips[a]) & set(clips[b])
    np.testing.assert_array_equal(a, query_halves(clips, labels, domains)[0])
    for d in (0, 1):
        for l in (0, 1):
            assert (a & (domains == d) & (labels == l)).sum() == 4


def test_unequal_query_cells_keep_equal_class_domain_weights():
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
    domains = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    correct = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(metrics(correct, labels, domains, np.ones(9, bool)), [50, 50, 50])


def test_ties_favor_freeze_and_domain_harm_excludes_a_high_balanced_score():
    scores = np.full((6, 4, 3), 50.0)
    np.testing.assert_array_equal(choices(scores), np.zeros(6))
    scores[:, 1] = [51, 54, 48]
    np.testing.assert_array_equal(choices(scores), np.zeros(6))
    np.testing.assert_array_equal(choices(scores, False), np.ones(6))


def test_evaluation_outcomes_cannot_change_selection():
    selection = np.full((6, 4, 3), 50.0)
    selection[:, 1] += 2
    a = transfer(selection, selection)
    evaluation = selection.copy()
    evaluation[:, 2] += 20
    b = transfer(selection, evaluation)
    np.testing.assert_array_equal(a["choices"], b["choices"])
    assert a["fixed_choice"] == b["fixed_choice"] == 1


def test_full_entrypoint_can_detect_heterogeneous_utility_and_no_headroom():
    labels = np.tile([0, 0, 1, 1], 2)
    domains = np.repeat([0, 1], 4)
    clips = np.array([str(i) for i in range(8)])
    a, _ = query_halves(clips, labels, domains)
    panels, queries = bootstrap_counts(a, labels, domains, 20, 7)
    assert (panels.sum(1) == 12).all()
    correct = np.full((6, 12, 4, 8), 0.5)
    for case in range(6):
        correct[case, :, 1 + case // 2] = 0.8
    gains, selected = bootstrap(correct, labels, domains, a, panels, queries)
    scores = np.stack([metrics(correct.mean(1), labels, domains, m) for m in (np.ones(8, bool), a, ~a)])
    result = analysis(scores[None], gains, np.stack([scores, scores]))
    assert result["screen_passed"]
    assert result["decision"] == "prepare_fresh_mechanism_experiment"
    assert not result["controller_qualified"]
    zero = np.full_like(scores, 50)
    result = analysis(zero[None], np.zeros_like(gains), np.stack([zero, zero]))
    assert not result["screen_passed"]
    assert result["decision"] == "stop_this_cached_adaptive_opportunity"


def test_cross_stratum_clip_is_rejected():
    with pytest.raises(ValueError, match="crosses strata"):
        query_halves(["same", "same"], np.array([0, 1]), np.array([0, 0]))
