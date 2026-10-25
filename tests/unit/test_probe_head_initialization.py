"""Exact head conversion, normalization isolation and no-op selection checks."""

import numpy as np
import torch

from cafl4ds.control_recovery import control_snapshots
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.probe_head_initialization import SupervisedHead, select_head, train_snapshots
from cafl4ds.reader_calibration import fit_probe


def test_probe_head_reproduces_fitted_probabilities() -> None:
    """The converted head must implement the fitted logistic function."""
    z = torch.randn(60, 6, generator=torch.Generator().manual_seed(91))
    labels = torch.arange(60) % 3
    probe = fit_probe(z, labels, torch.arange(48), 0.1)
    head = SupervisedHead(probe, "standardized_probe")
    with torch.no_grad():
        probabilities = head(z).softmax(1).numpy()
    expected = probe.classifier.predict_proba((z.numpy() - probe.mean) / probe.scale)
    np.testing.assert_allclose(probabilities, expected, atol=1e-6, rtol=1e-6)


def test_training_is_private_and_zero_epoch_preserves_encoder() -> None:
    """Same random weights across raw/normalized arms; no-op snapshots move nothing."""
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.arange(12) % 3
    with torch.no_grad():
        z = encoder.embed(images)
    probe = fit_probe(z, labels, torch.arange(12), 0.1)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    rng = torch.get_rng_state().clone()
    raw, _ = train_snapshots(encoder, images, labels, probe, "raw_random", 82, (0, 1))
    normalized, _ = train_snapshots(encoder, images, labels, probe, "standardized_random", 82, (0,))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, raw[0][0].state_dict()[k]) for k, v in before.items())
    assert torch.equal(raw[0][1].linear.weight, normalized[0][1].linear.weight)
    assert torch.equal(raw[0][1].linear.bias, normalized[0][1].linear.bias)
    assert any(not torch.equal(v, raw[1][0].state_dict()[k]) for k, v in before.items())
    original, _ = control_snapshots(encoder, images, labels, 82, 1e-4, (1,))
    assert all(torch.equal(v, raw[1][0].state_dict()[k]) for k, v in original[1][0].state_dict().items())
    assert all(torch.equal(v, raw[1][1].linear.state_dict()[k]) for k, v in original[1][1].state_dict().items())


def test_internal_ties_choose_no_encoder_update() -> None:
    """A trained arm cannot win merely by tying the unchanged representation."""
    rows = [
        {"arm": a, "epoch": e, "linear_error_balanced": 0.4, "refit_ce_balanced": ce}
        for a, e, ce in [("standardized_probe", 10, 0.1), ("standardized_random", 0, 2.0), ("raw_random", 0, 3.0)]
    ]
    assert select_head(rows) == {"arm": "raw_random", "epoch": 0, "internal_mean_error": 0.4}
