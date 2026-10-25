"""Meaningful paired-estimator and support-law controls on synthetic outcomes."""

import numpy as np
import pytest

from cafl4ds.target_alignment_measurement import TargetAlignmentMeasurement, resampling_counts, support_draws


def population():
    labels = np.tile(np.repeat([0, 1], 20), 2)
    domains = np.repeat([0, 1], 40)
    return labels, domains


def test_support_allocation_and_no_within_panel_duplicates():
    labels, domains = population()
    draws = support_draws(labels, domains, np.arange(80), [61001, 61007])
    assert all(len(np.unique(v)) == 50 for v in draws.values())
    for i, selected in enumerate(draws.values()):
        assert [int((labels[selected] == c).sum()) for c in (0, 1)] == [25, 25]
        assert [int((domains[selected] == d).sum()) for d in (0, 1)] == ([24, 26] if i == 0 else [26, 24])
    assert np.array_equal(draws[61001], support_draws(labels, domains, np.arange(80), [61001])[61001])


def test_identical_models_have_exact_zero_uncertainty():
    y, d = population()
    rng = np.random.default_rng(3)
    p = rng.random((12, 80, 2))
    pc, qc = resampling_counts(y, d, 12, 100, 5)
    result = TargetAlignmentMeasurement().compare(p, p, y, d, pc, qc)
    assert np.array_equal(result["mean_gain_pp"], np.zeros(3))
    assert all(np.array_equal(result[k]["half_width"], np.zeros(3)) for k in ("support", "query", "crossed"))


def test_domain_opposition_cannot_disappear_in_domain_outcomes():
    y, d = population()
    before = np.eye(2)[np.where(d == 0, 1 - y, y)][None].repeat(12, axis=0)
    after = 1 - before
    pc, qc = resampling_counts(y, d, 12, 100, 5)
    result = TargetAlignmentMeasurement().compare(before, after, y, d, pc, qc)
    np.testing.assert_array_equal(result["mean_gain_pp"], [0, 100, -100])
    np.testing.assert_allclose(result["crossed"]["lower"], [0, 100, -100], rtol=0, atol=1e-12)


def test_crossed_interval_matches_literal_paired_resampling():
    y, d = population()
    rng = np.random.default_rng(7)
    before, after = rng.random((2, 4, 80, 2))
    pc, qc = resampling_counts(y, d, 4, 80, 8)
    result = TargetAlignmentMeasurement().compare(before, after, y, d, pc, qc)
    diff = 100 * ((after.argmax(-1) == y).astype(float) - (before.argmax(-1) == y))
    samples = []
    for panels, queries in zip(pc, qc, strict=True):
        panel_ids = np.repeat(np.arange(4), panels)
        domain_results = []
        for domain in (0, 1):
            cell_results = []
            for label in (0, 1):
                ids = np.flatnonzero((d == domain) & (y == label))
                selected = np.repeat(ids, queries[ids])
                cell_results.append(diff[np.ix_(panel_ids, selected)].mean())
            domain_results.append(np.mean(cell_results))
        samples.append([np.mean(domain_results), *domain_results])
    np.testing.assert_allclose(result["crossed"]["lower"], np.quantile(samples, 0.025, axis=0), atol=1e-12)
    np.testing.assert_allclose(result["crossed"]["upper"], np.quantile(samples, 0.975, axis=0), atol=1e-12)


def test_invalid_query_multiplicities_fail():
    y, d = population()
    pc, qc = resampling_counts(y, d, 4, 20, 8)
    qc[0, 0] += 1
    with pytest.raises(ValueError, match="query"):
        TargetAlignmentMeasurement().compare(np.ones((4, 80, 2)), np.ones((4, 80, 2)), y, d, pc, qc)


def test_population_gate_requires_distinct_sources():
    good = {
        "mean_gain_pp": [2, 2, 2],
        "support": {"half_width": [0, 0.2, 0.2]},
        "query": {"half_width": [0, 0.2, 0.2]},
        "crossed": {"half_width": [0, 0.3, 0.3], "lower": [1.7, 1.7, 1.7], "upper": [2.3, 2.3, 2.3]},
    }
    bad = {
        **good,
        "mean_gain_pp": [-2, -2, -2],
        "crossed": {"half_width": [0, 0.3, 0.3], "lower": [-2.3] * 3, "upper": [-1.7] * 3},
    }
    records = [{"source": "one", "fresh": x} for x in [good, bad] * 4]
    result = TargetAlignmentMeasurement().summarize(records)
    assert not result["measurement_qualified"]
    assert result["gates"]["support_precision"]
    assert not result["gates"]["useful_sources"]
