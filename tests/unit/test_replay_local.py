"""Verify read-only gradients and train-only local response normalization."""

from __future__ import annotations

import copy
from typing import cast

import numpy as np
import torch
from torch import nn

from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.replay_local import encoder_gradient, fit, gradient_signals
from cafl4ds.ssl.base import SSLMethod


class Plant(SSLMethod):
    """A one-parameter encoder with analytically opposed objectives."""

    def __init__(self) -> None:
        """Initialize a zero scalar encoder and a separate module mode."""
        super().__init__(cast(TinyViTEncoder, nn.Linear(1, 1, bias=False)))
        self.encoder.weight.data.zero_()  # type: ignore[attr-defined]
        self.extra = nn.Dropout()

    @property
    def name(self) -> str:
        """Identify the diagnostic plant."""
        return "gradient-test"

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Different signs of the batch mean yield opposed encoder gradients."""
        torch.rand(1)
        return (self.encoder.weight.mean() - imgs.mean()).square()  # type: ignore[attr-defined]


def test_gradient_sign_and_observation_isolation() -> None:
    """Opposed data gradients are detected without changing learning state."""
    method = Plant().train()
    method.extra.eval()
    parameter = next(method.encoder.parameters())
    parameter.grad = torch.ones_like(parameter) * 7
    state = copy.deepcopy(method.state_dict())
    rng = torch.get_rng_state().clone()
    a = torch.ones(32, 1)
    assert torch.allclose(encoder_gradient(method, a, 123), torch.tensor([-2.0]))
    signals = gradient_signals(method, a, -a)
    assert abs(signals["gradient_cosine"] + 1) < 1e-7
    assert abs(signals["log_current_gradient_norm"] - np.log(2)) < 1e-7
    assert torch.equal(parameter.grad, torch.ones_like(parameter) * 7)
    assert torch.equal(rng, torch.get_rng_state())
    assert method.training and not method.extra.training
    assert all(torch.equal(state[k], v) for k, v in method.state_dict().items())


def test_reference_and_train_only_normalization() -> None:
    """The reference action stays zero even on extreme unseen inputs."""
    rows = [
        {"action": a, "features": {"step": float(i), "history_replay_fraction": 0.5}, "target": (a - 16) * 0.01}
        for i in (2, 4)
        for a in (8, 24)
    ]
    model = fit(rows, "action_history", 0.1)
    assert model.mean[0] == 3
    prediction = model.predict([{"action": 16, "features": {"step": 1e9, "history_replay_fraction": -1e9}}])
    assert prediction.tolist() == [0]
    assert model.mean[0] == 3
