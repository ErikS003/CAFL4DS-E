"""Check retention, hindsight ties, budget feasibility and query separation."""

import numpy as np

from cafl4ds.stopping_headroom import DOSES, fixed_choice, mixture, oracle_choices, query_halves


def test_ties_prefer_fewer_updates():
    """Constant capability should not invent a reason to consume work."""
    scores = np.full((18, 5, 3), 50.0)
    assert fixed_choice(scores) == 0
    np.testing.assert_array_equal(oracle_choices(scores, False), np.zeros(18))


def test_domain_guard_rejects_balanced_gain_with_hidden_loss():
    """A large day gain cannot buy an excessive night loss."""
    scores = np.array([[[50, 50, 50], [52, 56, 48], [51, 52, 50], [49, 49, 49], [48, 48, 48]]])
    assert oracle_choices(scores, False)[0] == 1
    assert oracle_choices(scores, True)[0] == 2


def test_mixture_matches_work_without_using_case_identity():
    """One common distribution must obey the expected update allowance."""
    scores = np.tile(np.column_stack((50 + DOSES / 8, 50 + DOSES / 8, 50 + DOSES / 8)), (18, 1, 1))
    q = mixture(scores, 6)
    assert abs(q.sum() - 1) < 1e-9
    assert abs(q @ DOSES - 6) < 1e-9
    assert np.all(q >= 0)


def test_query_halves_are_disjoint_and_balanced():
    """No selected query label can appear in that fold's evaluation half."""
    labels = np.tile(np.repeat([0, 1], 100), 2)
    domains = np.repeat([0, 1], 200)
    a, b = query_halves(np.arange(400), labels, domains)
    assert not np.any(a & b)
    assert np.all(a | b)
    for domain in (0, 1):
        for label in (0, 1):
            assert sum(a & (labels == label) & (domains == domain)) == 50
            assert sum(b & (labels == label) & (domains == domain)) == 50


def test_history_heterogeneity_can_create_hindsight_headroom():
    """The diagnostic should recognize opposing useful stopping choices."""
    scores = np.full((18, 5, 3), 50.0)
    scores[:9, 1] = 54
    scores[9:, 4] = 54
    choices = oracle_choices(scores, True)
    oracle_mean = scores[np.arange(18), choices, 0].mean()
    best_fixed_mean = scores[:, fixed_choice(scores), 0].mean()
    assert oracle_mean - best_fixed_mean == 2
