"""Availability, mechanism/source isolation, geometry and state safety of collapse forecasts."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.collapse_risk_regression import ARMS, crossfit, feature, geometry, ridge_fit, warning_report
from cafl4ds.short_buffer import capture, restore
from scripts.run_collapse_risk_regression import observe, update
from scripts.run_paired_update_response import fingerprint


def records():
    rows = []
    for group in range(6):
        for arm in ARMS:
            slope = 0.02 if arm == "blend100" else 0.005 if arm.startswith("soft") else 0.0001
            health = [[30 * np.exp(-slope * j * 8), 0.01 * np.exp(-slope * j), j * 0.001] for j in range(17)]
            rows.append({"group": str(group), "seed": 7, "arm": arm, "health": health, "losses": [-0.1] * 128})
    return rows


def test_geometry_distinguishes_constant_and_spread_bank():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(64, 8))
    constant = np.ones_like(z)
    assert geometry(z, z)[0] > 7
    assert geometry(z, z)[2] < 1e-12
    assert geometry(constant, constant)[0] == pytest.approx(1)
    assert geometry(constant, constant)[1] < 1e-25


def test_future_health_and_loss_cannot_change_prospective_features():
    row = records()[0]
    expected = feature(row["health"], row["losses"], 16)
    row["health"][3:] = [[999, 999, 999]] * 14
    row["losses"][16:] = [999] * 112
    np.testing.assert_array_equal(feature(row["health"], row["losses"], 16), expected)
    with pytest.raises(ValueError):
        feature(row["health"], row["losses"], 8)


def test_source_and_second_mechanism_never_enter_fit_or_boundary():
    fitted = crossfit(records())
    for group, fold in fitted["folds"].items():
        for index in fold["fit_points"]:
            point = fitted["points"][index]
            assert point["group"] != group
            assert not point["arm"].startswith("soft")
    changed = records()
    for row in changed:
        if row["group"] == "0":
            row["health"][-1][0] *= 100
    again = crossfit(changed)
    assert fitted["folds"]["0"] == again["folds"]["0"]


def test_scaling_and_constant_columns_use_only_fit_rows():
    x = np.array([[1, 4], [2, 4], [3, 4]], float)
    y = np.array([1, 2, 3.0])
    _, fit = ridge_fit(x, y, [[1e8, -1e8]])
    assert fit["center"] == [2.0, 4.0]
    assert fit["scale"][1] == 1
    assert fit["beta"][1] == 0


def test_early_events_are_counted_as_unforecastable():
    rows = records()
    for row in rows:
        if row["arm"].startswith("soft"):
            row["health"][1:] = [[1.0, 0.0001, 0.5]] * 16
    warning = warning_report(rows, crossfit(rows))
    assert warning["all_events"] == 18
    assert warning["eligible_events"] == 0
    assert warning["eligible_fraction"] == 0


def test_observer_preserves_bn_modes_optimizer_and_rng_and_replay():
    torch.set_num_threads(2)
    config = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 0.001})
    from cafl4ds.models.vit import TinyViTEncoder
    from cafl4ds.ssl.factory import build_simsiam

    torch.manual_seed(771)
    model = build_simsiam(
        TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3),
        proj_hidden=16,
        proj_dim=8,
        pred_hidden=8,
    )
    optimizer = instantiate(config, params=model.parameters())
    images = torch.rand(12, 3, 16, 16)
    update(model, optimizer, images[:8], 11)
    model.projector.eval()
    state = capture(model, optimizer)
    observe(model, optimizer, images)
    assert fingerprint(capture(model, optimizer)) == fingerprint(state)
    observed = update(model, optimizer, images[:8], 17)
    marker = fingerprint(capture(model, optimizer))
    twin, optim = restore(model, config, state)
    assert observed == update(twin, optim, images[:8], 17)
    assert marker == fingerprint(capture(twin, optim))
