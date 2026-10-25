"""Verify leakage exclusion and genuinely matched probe regularization."""

import torch

from cafl4ds.probe_diagnostic import balanced_supports, configurations, fit_reader
from cafl4ds.replay_health_support import precise_probe


def test_support_sampling_is_balanced_and_uses_only_reserved_support() -> None:
    """Query indices are outside the eligible union even when their labels match."""
    pool = torch.arange(1152)
    labels = torch.arange(6).repeat_interleave(192) % 3
    domains = torch.arange(6).repeat_interleave(192) // 3
    metadata = {"supports": {"pooled": pool}, "labels": labels, "domains": domains}
    subsets = balanced_supports(metadata)
    assert subsets == balanced_supports(metadata)
    assert len({tuple(subsets[f"sample{i}"]) for i in range(4)}) == 4
    for name, indices in subsets.items():
        if name == "pooled":
            continue
        assert len(indices) == len(set(indices)) == 384
        assert set(indices) <= set(pool.tolist())
        assert all(
            int(((labels[indices] == c) & (domains[indices] == d)).sum()) == 64 for d in (0, 1) for c in (0, 1, 2)
        )


def test_matched_penalty_preserves_mean_loss_regularization() -> None:
    """Increasing labeled support cannot silently weaken the normalized penalty."""
    readers = configurations()
    assert len(readers) == len({r["reader"] for r in readers}) == 24
    for row in readers:
        if row["penalty"] == "matched":
            assert abs(row["actual_C"] * row["n_support"] - row["reference_C"] * 384) < 1e-12


def test_panel_statistics_reproduce_existing_probe_and_ignore_query_labels() -> None:
    """The generalized fitter retains the original solver and fitting-only labels."""
    z = torch.randn(90, 5, generator=torch.Generator().manual_seed(4))
    labels = torch.arange(90) % 3
    fitting = list(range(60))
    expected = precise_probe(z, labels, torch.tensor(fitting), 0.1, 1e-10)
    actual = fit_reader(z, labels, fitting, fitting, 0.1)
    changed = labels.clone()
    changed[60:] = (changed[60:] + 1) % 3
    unaffected = fit_reader(z, changed, fitting, fitting, 0.1)
    assert (actual.classifier.coef_ == expected.classifier.coef_).all()
    assert (actual.classifier.coef_ == unaffected.classifier.coef_).all()
