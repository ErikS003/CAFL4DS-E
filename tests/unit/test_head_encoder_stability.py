"""Verify frozen encoder invariance, reference equivalence and no-op selection."""

import torch

from cafl4ds.head_encoder_stability import select_rate, train_snapshots
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.probe_head_initialization import train_snapshots as original
from cafl4ds.reader_calibration import fit_probe


def test_frozen_and_reference_trajectories() -> None:
    """Frozen embeddings cannot move; reference updates exactly match the prior trainer."""
    torch.set_num_threads(1)
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.arange(12) % 3
    with torch.no_grad():
        z = encoder.embed(images)
    probe = fit_probe(z, labels, torch.arange(12), 0.1)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    rng = torch.get_rng_state().clone()
    frozen, trace = train_snapshots(encoder, images, labels, probe, "head_only", 82, (0, 1))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, frozen[1][0].state_dict()[k]) for k, v in before.items())
    assert trace[0]["encoder_updates"] == 0 and trace[0]["updates"] == 1
    assert not torch.equal(frozen[0][1].linear.weight, frozen[1][1].linear.weight)
    reference, _ = train_snapshots(encoder, images, labels, probe, "reference_1e4", 82, (1,))
    expected, _ = original(encoder, images, labels, probe, "standardized_probe", 82, (1,))
    for i in (0, 1):
        assert all(torch.equal(v, reference[1][i].state_dict()[k]) for k, v in expected[1][i].state_dict().items())


def test_noop_and_low_rate_ties() -> None:
    """Ties preserve the encoder and prefer the smallest update."""
    rows = [
        {"arm": a, "epoch": e, "linear_error_balanced": 0.4}
        for a, e in [("reference_1e4", 10), ("gentle_1e5", 0), ("head_only", 0)]
    ]
    assert select_rate(rows)["arm"] == "head_only"
    assert select_rate(rows)["epoch"] == 0
