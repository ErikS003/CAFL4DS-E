"""Private training and internal-only selection invariants."""

import torch

from cafl4ds.control_recovery import control_snapshots, select_control
from cafl4ds.models.vit import TinyViTEncoder


def test_selection_ignores_ce_and_prefers_earlier_ties() -> None:
    """Accuracy chooses the candidate even when secondary CE would prefer another."""
    rows = [
        {"learning_rate": lr, "epoch": epoch, "linear_error_balanced": error, "refit_ce_balanced": ce}
        for lr, epoch, error, ce in [
            (1e-4, 1, 0.3, 0.1),
            (1e-5, 5, 0.2, 9.0),
            (3e-5, 1, 0.2, 10.0),
            (1e-5, 1, 0.2, 11.0),
        ]
    ]
    assert select_control(rows) == {"learning_rate": 1e-5, "epoch": 1, "internal_mean_error": 0.2}


def test_snapshots_preserve_model_rng_and_earlier_checkpoint() -> None:
    """Extending a trajectory leaves its early checkpoint unchanged."""
    encoder = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=12, depth=1, num_heads=3)
    images = torch.rand(9, 3, 8, 8)
    labels = torch.arange(9) % 3
    original = {k: v.clone() for k, v in encoder.state_dict().items()}
    rng = torch.get_rng_state().clone()
    short, _ = control_snapshots(encoder, images, labels, 9, 1e-5, (1,))
    longer, trace = control_snapshots(encoder, images, labels, 9, 1e-5, (1, 2))
    assert len(trace) == 2 and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in original.items())
    assert all(torch.equal(v, longer[1][0].state_dict()[k]) for k, v in short[1][0].state_dict().items())
    assert any(not torch.equal(v, original[k]) for k, v in longer[2][0].state_dict().items())
