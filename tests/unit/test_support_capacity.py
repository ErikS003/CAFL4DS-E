"""Verify the expanded reader cannot consume outcome-query labels during selection."""

import pytest
import torch

from cafl4ds.support_capacity import capacity_cv, capacity_folds, expanded_support


def panel() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Six cells: twelve reserved support, thirty-two monitor and six outcome images."""
    labels = torch.tensor([label for domain in (0, 1) for label in (0, 1, 2) for _ in range(50)])
    domains = torch.tensor([domain for domain in (0, 1) for label in (0, 1, 2) for _ in range(50)])
    reserved = torch.tensor([cell * 50 + i for cell in range(6) for i in range(12)])
    monitor = torch.tensor([cell * 50 + i for cell in range(6) for i in range(12, 44)])
    roles = {
        "development": torch.tensor([cell * 50 + i for cell in range(6) for i in range(44, 47)]),
        "holdout": torch.tensor([cell * 50 + i for cell in range(6) for i in range(47, 50)]),
    }
    return labels, domains, reserved, monitor, roles


def test_expanded_support_roles_and_balanced_private_folds() -> None:
    """All support is partitioned once and never overlaps either outcome set."""
    labels, domains, reserved, monitor, roles = panel()
    support = expanded_support(reserved, monitor, roles)
    rng = torch.get_rng_state().clone()
    folds = capacity_folds(labels, domains, support, 71011)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(torch.cat([v for _, v in folds]).sort().values, support.sort().values)
    for fitting, validation in folds:
        assert len(fitting) == 198 and len(validation) == 66
        assert not torch.isin(fitting, validation).any()
        for domain in (0, 1):
            for label in (0, 1, 2):
                assert ((labels[fitting] == label) & (domains[fitting] == domain)).sum() == 33
                assert ((labels[validation] == label) & (domains[validation] == domain)).sum() == 11
    with pytest.raises(ValueError, match="overlap"):
        expanded_support(reserved, monitor, {**roles, "development": monitor[:3]})


def test_query_perturbation_cannot_change_support_cv() -> None:
    """Changing both query labels and features leaves every CV score unchanged."""
    labels, domains, reserved, monitor, roles = panel()
    support = expanded_support(reserved, monitor, roles)
    z = torch.randn(len(labels), 8, generator=torch.Generator().manual_seed(8))
    before = capacity_cv(z, labels, domains, support)
    query = torch.cat(list(roles.values()))
    z[query] = 999
    labels[query] = (labels[query] + 1) % 3
    assert capacity_cv(z, labels, domains, support) == before
