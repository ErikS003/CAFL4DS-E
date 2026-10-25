"""Check support-fold isolation and deterministic calibration selection."""

import numpy as np
import torch

from cafl4ds.reader_calibration import balanced_folds, choose_c, fit_probe, support_cv
from tests.unit.test_reader_reliability import evaluation_indices


def test_support_folds_are_balanced_private_and_partition_support() -> None:
    """Every support image is held out once; no query/RNG enters the folds."""
    labels, domains, query, _ = evaluation_indices()
    support = torch.arange(len(labels))[~torch.isin(torch.arange(len(labels)), query)]
    rng = torch.get_rng_state().clone()
    folds = balanced_folds(labels, domains, support, 71011)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(torch.cat([v for _, v in folds]).sort().values, support.sort().values)
    for fitting, validation in folds:
        assert len(fitting) == 48 and len(validation) == 24
        assert not torch.isin(fitting, validation).any()
        assert not torch.isin(torch.cat([fitting, validation]), query).any()
        for domain in (0, 1):
            for label in (0, 1, 2):
                assert ((labels[fitting] == label) & (domains[fitting] == domain)).sum() == 8
                assert ((labels[validation] == label) & (domains[validation] == domain)).sum() == 4


def test_query_changes_leave_cv_and_selection_identical() -> None:
    """Adversarial changes to query labels/features cannot affect selection evidence."""
    labels, domains, query, _ = evaluation_indices()
    support = torch.arange(len(labels))[~torch.isin(torch.arange(len(labels)), query)]
    z = torch.randn(len(labels), 8, generator=torch.Generator().manual_seed(8))
    before = support_cv(z, labels, domains, support)
    changed_labels = labels.clone()
    changed_z = z.clone()
    changed_labels[query] = (labels[query] + 1) % 3
    changed_z[query] = 999
    assert support_cv(changed_z, changed_labels, domains, support) == before
    fitting, validation = balanced_folds(labels, domains, support, 71011)[0]
    probe = fit_probe(z, labels, fitting, 0.1)
    changed_z[validation] = 999
    changed_labels[validation] = (labels[validation] + 1) % 3
    other = fit_probe(changed_z, changed_labels, fitting, 0.1)
    assert np.array_equal(probe.mean, other.mean)
    assert np.array_equal(probe.scale, other.scale)
    assert np.array_equal(probe.classifier.coef_, other.classifier.coef_)


def test_c_ties_prefer_stronger_regularization() -> None:
    """The prespecified tolerance cannot turn numerical ties into a recipe search."""
    assert choose_c({0.01: 1.0 + 5e-11, 0.1: 1.0, 1.0: 1.2, 10.0: 2.0}) == 0.01
    assert choose_c({0.01: 1.0 + 2e-10, 0.1: 1.0, 1.0: 1.2, 10.0: 2.0}) == 0.1
