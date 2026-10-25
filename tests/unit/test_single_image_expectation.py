"""Check repeat separation and reject agreement attributable only to parent state."""

import numpy as np

from cafl4ds.single_image_expectation import agreement, center_candidates, measurement_gate, replication_pairs
from scripts.verify_single_image_expectation_analysis import independent_pairs, verify_result


def inputs():
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(6), 2)
    draws = rng.integers(0, 6, (100, 6))
    return rng, groups, draws


def test_centering_removes_parent_only_agreement():
    rng, groups, draws = inputs()
    a = np.arange(12)[:, None] * 100 + rng.normal(size=(12, 3))
    b = np.arange(12)[:, None] * 100 + rng.normal(size=(12, 3))
    assert agreement(a, b, groups, draws)["correlation"] > 0.99
    centered = agreement(center_candidates(a), center_candidates(b), groups, draws)
    assert abs(centered["correlation"]) < 0.5
    assert not measurement_gate(centered, centered)


def test_view_halves_do_not_share_training_draws():
    rng, _, _ = inputs()
    effects = rng.normal(size=(12, 3, 6, 3))
    a, b = replication_pairs(effects)["views_3_raw"]
    changed = effects.copy()
    changed[:, :, 3:] += 100
    aa, bb = replication_pairs(changed)["views_3_raw"]
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_allclose(bb - b, 100)


def test_held_reader_is_excluded_from_its_two_reader_source():
    effects = np.zeros((12, 3, 6, 3))
    effects[:, :, :, 0] = np.array([1, 2, 3])[None, :, None]
    pairs = replication_pairs(effects)
    source, held = pairs["readers_2_centered"]
    np.testing.assert_array_equal(source[:, :, 0], 0)
    np.testing.assert_array_equal(held[:, :, 0], np.tile([-1, 0, 1], (12, 1)))
    source, held = pairs["joint_centered"]
    np.testing.assert_array_equal(held[:, :, 0], 0)
    np.testing.assert_array_equal(source[:, :, 0], np.tile([-1, 0, 1], (12, 1)))


def test_power_identity_and_stable_image_gate():
    rng, groups, draws = inputs()
    a = center_candidates(rng.normal(size=(12, 3)))
    result = agreement(a, a + rng.normal(size=a.shape) * 0.001, groups, draws)
    np.testing.assert_allclose(
        result["cross_repeat_signal_power"] + result["half_difference_noise_power"], result["observed_power"]
    )
    assert measurement_gate(result, result)


def test_exact_zero_effects_are_not_positive_measurement_evidence():
    _, groups, draws = inputs()
    result = agreement(np.zeros((12, 3)), np.zeros((12, 3)), groups, draws)
    assert result["correlation"] is None and result["nonzero_direction_agreement"] is None
    assert result["zero_fraction_a"] == 1 and result["both_nonzero_cases"] == 0
    assert not measurement_gate(result, result)


def test_independent_auditor_matches_all_repeat_constructions():
    rng, _, _ = inputs()
    effects = rng.normal(size=(12, 3, 6, 3))
    first, second = replication_pairs(effects), independent_pairs(effects)
    assert first.keys() == second.keys()
    for key in first:
        for a, b in zip(first[key], second[key], strict=True):
            np.testing.assert_allclose(a, b, atol=1e-14)


def test_independent_statistics_and_cluster_bootstrap_match_including_zero_groups():
    rng, groups, draws = inputs()
    for shape in ((12, 3), (12, 3, 6)):
        a, b = rng.normal(size=shape), rng.normal(size=shape)
        a[:4], b[:4] = 0, 0
        saved = agreement(a, b, groups, draws)
        verify_result(a, b, saved, groups, draws, 1e-10)
    zeros = np.zeros((12, 3))
    verify_result(zeros, zeros, agreement(zeros, zeros, groups, draws), groups, draws, 1e-10)
