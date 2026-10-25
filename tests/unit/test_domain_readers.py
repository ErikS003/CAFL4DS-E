"""Check that conditional readers isolate support domains and reject leakage."""

import pytest
import torch

from cafl4ds.domain_readers import reader_losses


def test_conditional_domain_isolation() -> None:
    """Night support changes cannot affect daytime conditional predictions."""
    generator = torch.Generator().manual_seed(821)
    labels = torch.arange(3).repeat(24)
    domains = torch.tensor([0] * 36 + [1] * 36)
    features = torch.randn(72, 6, generator=generator) + torch.nn.functional.one_hot(labels, 6).float() * 3
    support = torch.cat([torch.arange(30), torch.arange(36, 66)])
    query = torch.cat([torch.arange(30, 36), torch.arange(66, 72)])
    recipe = {"0": 0.1, "1": 0.1}
    original = reader_losses(features, labels, domains, support, query, True, recipe)
    changed = features.clone()
    changed[domains == 1] = torch.randn(36, 6, generator=generator) * 10
    altered = reader_losses(changed, labels, domains, support, query, True, recipe)
    for metric in original:
        assert torch.equal(original[metric][:6], altered[metric][:6])
        assert torch.isfinite(original[metric]).all()
    with pytest.raises(ValueError, match="overlap"):
        reader_losses(features, labels, domains, support, support[:6], True, recipe)
