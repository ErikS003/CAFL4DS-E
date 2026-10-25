"""Calibration leakage, measured risk constraints, and exact rollback semantics."""

import numpy as np
import pytest
import torch

from cafl4ds.short_buffer import capture
from cafl4ds.single_image_gate_protection import (
    BoundedForecaster,
    confirmed_anchor,
    development_scores,
    protection_passes,
    rollback,
)


def test_clock_removal_and_bounding_prevent_extrapolated_predictions():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(60, 4))
    y = 2 * x[:, 0] - x[:, 3]
    settings = dict(regressors=dict(linear={"_target_": "sklearn.linear_model.Ridge", "alpha": 1.0}))
    model = BoundedForecaster("linear", x, y, [0, 3], settings)
    extreme = x[:1].copy()
    extreme[:, 1:3] = 1e12
    np.testing.assert_array_equal(model.predict(extreme), model.predict(x[:1]))
    extreme[:, 0] = 1e12
    clipped = extreme.copy()
    clipped[:, 0] = model.maximum[0]
    np.testing.assert_array_equal(model.predict(extreme), model.predict(clipped))


def test_development_guard_checks_subgroups_even_if_overall_improves():
    assert not protection_passes("corrected_development", 0.1, [0.8, 0.64, 0.9], [0.75, 0.7, 0.8], 0.2, 0.02)
    assert protection_passes("corrected_development", 0.1, [0.73, 0.68, 0.78], [0.75, 0.7, 0.8], 0.2, 0.02)
    assert protection_passes("corrected_J", 100.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2, 0.02)


def test_covariance_guard_is_independent_of_development_scores():
    assert protection_passes("corrected_covariance", 0.2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2, 0.02)
    assert not protection_passes("corrected_covariance", 0.2001, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.2, 0.02)
    with pytest.raises(ValueError, match="Nonfinite"):
        protection_passes("corrected_covariance", np.nan, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.2, 0.02)


def test_confirmed_anchor_requires_two_checkpoints_and_never_decreases():
    anchor = np.array([0.6, 0.7, 0.8])
    raised = confirmed_anchor(anchor, [0.9, 0.72, 0.79], [0.65, 0.74, 0.95])
    np.testing.assert_allclose(raised, [0.65, 0.72, 0.8])
    np.testing.assert_array_equal(confirmed_anchor(raised, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), raised)


def test_development_scores_use_support_panel_average_and_domain_masks():
    p = np.array([[[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6]]] * 3)
    probe = dict(probabilities=np.stack((p, p), axis=1))
    panel = dict(query_labels=np.array([0, 1, 1, 0]), query_domains=np.array([0, 0, 1, 1]))
    score = development_scores(probe, panel, 0.05)
    np.testing.assert_allclose(score[:3], [0.5, 0.5, 0.5])
    assert np.all((score[3:] >= 0) & (score[3:] <= 1))


def test_rejected_trial_restores_weights_momentum_modes_and_rng():
    from scripts.run_paired_update_response import fingerprint

    torch.manual_seed(81)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    model.eval()
    before = capture(model, optimizer)
    expected_random = torch.rand(4)
    torch.set_rng_state(before["cpu_rng"])
    model.train()
    optimizer.zero_grad()
    model(torch.randn(1, 3)).sum().backward()
    optimizer.step()
    rollback(model, optimizer, before)
    assert fingerprint(capture(model, optimizer)) == fingerprint(before)
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
