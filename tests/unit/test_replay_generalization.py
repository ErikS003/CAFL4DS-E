"""Novel order construction must work with the limiting64-image outgoing pool."""

import pytest
import torch

from scripts.run_replay_generalization import new_order


@pytest.mark.parametrize("condition", ["shuffled_incoming", "interleaved"])
def test_order_preserves_history_and_unique_images_with_limited_outgoing(condition: str) -> None:
    """Do not reuse history when only two outgoing batches remain available."""
    order = list(range(768))
    domains = torch.ones(768, dtype=torch.long)
    domains[:256] = 0
    domains[704:] = 0
    result = new_order(order, domains, condition, 80001)
    assert result[:320] == order[:320]
    assert sorted(result) == order
    assert result == new_order(order, domains, condition, 80001)
    expected = [1] * 6 if condition == "shuffled_incoming" else [1, 0, 1, 0, 1, 1]
    actual = domains[torch.tensor(result[320:512])].reshape(6, 32)
    for batch, domain in zip(actual, expected, strict=True):
        assert bool((batch == domain).all())
