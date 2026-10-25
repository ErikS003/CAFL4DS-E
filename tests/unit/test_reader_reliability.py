"""Verify private balanced support draws and support-only reader fitting."""

import numpy as np
import pytest
import torch

from cafl4ds.reader_reliability import fit_reader, read_cached, support_draws


def evaluation_indices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Six cells with twelve support and six query images each."""
    labels = torch.tensor([label for domain in range(2) for label in range(3) for _ in range(18)])
    domains = torch.tensor([domain for domain in range(2) for label in range(3) for _ in range(18)])
    query = torch.tensor([cell * 18 + i for cell in range(6) for i in range(12, 18)])
    original = torch.tensor([cell * 18 + i for cell in range(6) for i in range(6)])
    return labels, domains, query, original


def test_draws_are_private_balanced_and_query_disjoint() -> None:
    """Support sampling cannot consume training RNG or leak query observations."""
    labels, domains, query, original = evaluation_indices()
    rng = torch.get_rng_state().clone()
    draws = support_draws(labels, domains, query, original)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(draws) == 22
    assert len({tuple(x.tolist()) for name, x in draws.items() if name.startswith("draw36")}) == 20
    for name, support in draws.items():
        assert not torch.isin(support, query).any()
        assert len(support.unique()) == len(support)
        for domain in range(2):
            for label in range(3):
                assert ((labels[support] == label) & (domains[support] == domain)).sum() == (
                    12 if name == "full72" else 6
                )


def test_query_labels_and_features_cannot_change_fitted_head() -> None:
    """Changing forbidden query information leaves support-normalized coefficients identical."""
    labels, domains, query, support = evaluation_indices()
    z = torch.randn(108, 8, generator=torch.Generator().manual_seed(22))
    reader = fit_reader(z, labels, support)
    changed_labels, changed_z = labels.clone(), z.clone()
    changed_labels[query] = (labels[query] + 1) % 3
    changed_z[query] = 1000
    other = fit_reader(changed_z, changed_labels, support)
    assert np.array_equal(reader.mean, other.mean)
    assert np.array_equal(reader.scale, other.scale)
    assert np.array_equal(reader.classifier.coef_, other.classifier.coef_)
    with pytest.raises(ValueError, match="overlap"):
        read_cached(z, labels, domains, {"development": query, "holdout": query}, query[:6])
