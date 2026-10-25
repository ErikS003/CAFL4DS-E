"""Geometry controls for the fixed-bank health calibration."""

import numpy as np
import pytest

from cafl4ds.health_calibration import changes, forecasts, geometry


def test_hold_is_quiet() -> None:
    """Identical representations have zero motion and zero geometry changes."""
    z = np.random.default_rng(7).normal(size=(256, 16)).astype(np.float32)
    assert all(abs(v) < 1e-6 for v in changes(z, z.copy()).values())


def test_healthy_rotation_has_cka_invariance_but_cosine_motion() -> None:
    """A coordinate rotation must not be mistaken for CKA content degradation."""
    rng = np.random.default_rng(17)
    z = rng.normal(size=(256, 16)).astype(np.float32)
    rotation = np.linalg.qr(rng.normal(size=(16, 16)))[0].astype(np.float32)
    values = changes(z, z @ rotation)
    assert abs(values["cka_drift"]) < 1e-6
    assert values["cosine_drift"] > 0.1
    assert abs(values["rank_drop"]) < 1e-6


def test_rank_guard_detects_dimension_collapse() -> None:
    """The collapse guard reacts to a rank-one population."""
    rng = np.random.default_rng(19)
    z = rng.normal(size=(256, 16)).astype(np.float32)
    collapsed = z[:, :1] * np.ones((1, 16), np.float32)
    assert changes(z, collapsed)["rank_drop"] > 0.8


def test_reference_size_guard() -> None:
    """Reject a bank whose sample count artificially limits the rank spectrum."""
    with pytest.raises(ValueError):
        geometry(np.ones((16, 96), np.float32))


def test_calibration_excludes_whole_source() -> None:
    """Held-source outcome changes cannot alter that source's predictions."""
    points = [
        {"stream": str(g), "terminal_damage": g / 4 + k / 5, "feature": g + k / 10} for g in range(6) for k in range(3)
    ]
    first = forecasts(points, ("feature",))
    altered = [{**p, "terminal_damage": p["terminal_damage"] + (100 if p["stream"] == "0" else 0)} for p in points]
    assert first["predictions"][:3] == forecasts(altered, ("feature",))["predictions"][:3]
