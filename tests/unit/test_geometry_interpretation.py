import numpy as np
import pytest

from cafl4ds.geometry_interpretation import (
    alignment,
    compare,
    contrasts,
    control_scores,
    geometry,
    synthetic,
    transformations,
    transported,
)


def test_raw_geometry_invariance_is_distinct_from_coordinate_motion():
    z, _, heads = synthetic(7, n=128, d=24)
    controls = transformations(z, heads, 8)
    for name in ("rotation", "scale", "translation"):
        a, t, _ = controls[name]
        r = compare(z, z @ a + t, top_k=8)
        assert r["gram_change"] < 1e-20 and r["spectral_l1"] < 1e-10
        assert r["aligned_residual"] < 1e-10 and r["neighbor_retention"] == 1
    assert compare(z, z @ controls["rotation"][0], top_k=8)["coordinate_motion"] > 0.1
    assert compare(z, z * 2, top_k=8)["energy_ratio"] == pytest.approx(4)


def test_angular_translation_is_a_different_view():
    z, _, h = synthetic(2, n=128, d=24)
    _, t, _ = transformations(z, h)["translation"]
    assert compare(z, z + t, top_k=8)["gram_change"] < 1e-20
    assert compare(z, z + t, "angular", top_k=8)["gram_change"] > 1e-8


def test_sample_space_blocks_reconstruct_full_change():
    z, _, h = synthetic(3, n=128, d=24)
    a, _, _ = transformations(z, h)["anisotropic"]
    r = compare(z, z @ a, top_k=8)
    assert r["gram_change"] == pytest.approx(
        r["within_reference"] + r["cross_reference"] + r["outside_reference"], abs=1e-12
    )
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    rr = compare(z @ q, z @ a @ q, top_k=8)
    for key in ("gram_change", "within_reference", "cross_reference", "outside_reference", "top_subspace_overlap"):
        assert r[key] == pytest.approx(rr[key], abs=1e-10)


def test_invertible_transport_and_support_orthogonal_control_preserve_known_logits():
    z, _, heads = synthetic(5, n=128, d=24)
    for name, affine in transformations(z, heads).items():
        a, t, invertible = affine
        rows = control_scores(z, z @ a + t, heads, affine)
        if invertible or name == "drop_support_orthogonal":
            assert max(r["max_reference_logit_error"] for r in rows) < 1e-10
    fit = alignment(z, z * 2 + 3)
    np.testing.assert_allclose(transported(z * 2 + 3, fit), z, atol=1e-12)


def test_low_energy_task_removal_erases_known_contrast_without_large_rank_change():
    z, y, heads = synthetic(1, n=128, d=24)
    a, _, _ = transformations(z, heads)["drop_night_contrast"]
    after = z @ a
    assert np.mean(y * (z @ heads[1]["direction"])) > 0
    np.testing.assert_allclose(after @ heads[1]["direction"], 0, atol=1e-12)
    assert geometry(after)["effective_rank"] / geometry(z)["effective_rank"] > 0.99
    with pytest.raises(ValueError):
        compare(z, np.ones_like(z))


def test_all_support_defined_directions_are_preserved_by_declared_orthogonal_projection():
    rng = np.random.default_rng(11)
    support = rng.normal(size=(200, 96))
    labels = np.tile(np.repeat([0, 1], 50), 2)
    domains = np.repeat([0, 1], 100)
    draws = {
        k: np.concatenate([rng.choice(np.arange(start, start + 50), 12, replace=False) for start in (0, 50, 100, 150)])
        for k in range(12)
    }
    heads = contrasts(support, labels, domains, draws)
    assert len(heads) == 24 and all(h["within_sd"] > 0 for h in heads)
    a, t, invertible = transformations(support[:128], heads, 12)["drop_support_orthogonal"]
    assert not invertible
    query = rng.normal(size=(64, 96))
    rows = control_scores(query, query @ a + t, heads, (a, t, invertible))
    assert max(r["max_reference_logit_error"] for r in rows) < 1e-10
