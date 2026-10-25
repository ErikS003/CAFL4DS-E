import numpy as np
import pytest
import torch

from cafl4ds.matched_fitting import bootstrap_primary, effects, fit, reference_fixed_fit
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import fine_tune_gpu


class SmallEncoder(torch.nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 4)
        self.register_buffer("constant", torch.tensor(3.0))

    def embed(self, x):
        return self.layer(x).tanh()


def fixture():
    torch.set_num_threads(4)
    torch.manual_seed(71)
    return SmallEncoder(), torch.randn(40, 3), torch.arange(40) % 2


def test_fixed_state_and_literal_reproduction():
    model, x, y = fixture()
    before = fingerprint(model.state_dict())
    a, h, audit = fit(model, x, y, 83)
    b, j, losses = reference_fixed_fit(model, x, y, 83)
    assert fingerprint(a.state_dict()) == fingerprint(b.state_dict()) == before
    assert fingerprint(h.state_dict()) == fingerprint(j.state_dict())
    assert audit["backbone_delta_l2"] == 0
    assert losses == [r[0] for r in audit["trace_loss_joint_gradient"]]


def test_instrumentation_matches_canonical_and_pairing():
    model, x, y = fixture()
    a, h, audit = fit(model, x, y, 83, 1e-5)
    b, j, old = fine_tune_gpu(model, x, y, 83)
    assert fingerprint(a.state_dict()) == fingerprint(b.state_dict())
    assert fingerprint(h.state_dict()) == fingerprint(j.state_dict())
    for key, value in old.items():
        assert audit[key] == value
    _, _, fixed = fit(model, x, y, 83)
    for key in ("initial_head", "batch_order_sha256", "initial_loss"):
        assert audit[key] == fixed[key]


def test_difference_of_differences_and_parent_reuse():
    labels = np.array([0, 1, 0, 1])
    domains = np.array([0, 0, 1, 1])
    joint = np.ones((3, 4, 3, 2, 4), dtype=np.uint8)
    fixed = joint.copy()
    joint[:, :, 1:, :, 3] = 0
    scores, gains, interaction = effects(joint, fixed, labels, domains)
    assert scores.shape == (2, 3, 4, 3, 2, 7)
    np.testing.assert_array_equal(gains[0, ..., 6], -100)
    np.testing.assert_array_equal(interaction[..., 6], 100)
    with pytest.raises(ValueError):
        effects(joint, fixed[..., :2], labels, domains)


def test_primary_bootstrap_pairing_and_cell_selection():
    delta = np.zeros((3, 12, 2048))
    delta[0, :, :1536] = 999
    delta[0, :, 1536:] = -7
    delta[1, :, 1536:] = 2
    delta[2] = delta[1] - delta[0]
    values, sw, qw = bootstrap_primary(delta, draws=20)
    np.testing.assert_allclose(values, np.tile([-7, 2, 9], (20, 1)), atol=1e-12)
    np.testing.assert_allclose(values[:, 2], values[:, 1] - values[:, 0], atol=1e-12)
    np.testing.assert_allclose(sw.sum(-1), 1)
    np.testing.assert_allclose(qw.sum(-1), 1)
