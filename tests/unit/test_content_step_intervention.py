import numpy as np
import pytest
import torch

from cafl4ds.content_step_intervention import (
    conditional_samples,
    contrasts,
    displacement,
    load_xor,
    save_xor,
    set_encoder_step,
)
from scripts.run_paired_update_response import fingerprint


def test_encoder_norm_direction_and_adam_state():
    torch.manual_seed(3)
    encoder = torch.nn.Linear(3, 4)
    decoder = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=0.01)
    before = [p.detach().clone() for p in encoder.parameters()]
    decoder(encoder(torch.randn(8, 3))).square().mean().backward()
    opt.step()
    state, head = fingerprint(opt.state_dict()), fingerprint(decoder.state_dict())
    proposed = [p.detach().clone() - a for p, a in zip(encoder.parameters(), before, strict=True)]
    raw = displacement(list(encoder.parameters()), before)
    _, actual, factor = set_encoder_step(encoder.parameters(), before, raw / 2)
    assert abs(actual - raw / 2) < 1e-7 and factor == 0.5
    for p, a, d in zip(encoder.parameters(), before, proposed, strict=True):
        torch.testing.assert_close(p.detach() - a, d / 2, atol=1e-7, rtol=1e-5)
    assert fingerprint(opt.state_dict()) == state and fingerprint(decoder.state_dict()) == head
    with pytest.raises(ValueError):
        set_encoder_step(encoder.parameters(), before, 0)


def test_xor_roundtrip_including_noncontiguous_and_integer_buffers(tmp_path):
    base = {"x": torch.arange(12.0).reshape(3, 4).T, "n": torch.tensor([8], dtype=torch.int64)}
    state = {"x": base["x"] + 0.001, "n": torch.tensor([9], dtype=torch.int64)}
    path = tmp_path / "weights.npz"
    save_xor(path, base, state)
    assert fingerprint(load_xor(path, base)) == fingerprint(state)


def test_factorial_contrast_signs():
    a = np.zeros((2, 3, 4, 2, 2, 2, 12, 7))
    a[..., 0, 1, :, :] = 2
    a[..., 1, 0, :, :] = 3
    a[..., 1, 1, :, :] = 7
    values = contrasts(a)
    np.testing.assert_array_equal(values[:, 0, 0, 0, 0, 0, 0], [2, 4, 3, 5, 2])


def test_bootstrap_common_pairing_and_primary_cell():
    a = np.zeros((5, 2, 12, 2048))
    a[..., :1536] = 999
    a[..., 1536:] = np.arange(10).reshape(5, 2, 1, 1)
    values, sw, qw = conditional_samples(a, draws=10)
    np.testing.assert_allclose(values, np.tile(np.arange(10).reshape(1, 5, 2), (10, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(sw.sum(1), 1)
    np.testing.assert_allclose(qw.sum(1), 1)
