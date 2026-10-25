"""Objective orientation, causal fitting and admission-budget invariants."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from cafl4ds.single_image_filter import features, fit_forecasters, matched_random, objective
from cafl4ds.single_image_filter_analysis import diagnostic_scores, paired


def settings():
    return OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parents[2] / "cafl4ds/configs/single_image_filter.yaml")
    )


def test_objective_has_explicit_direction_and_warm_penalties():
    s = settings()
    warm = np.array([4, 0.01, 0.001, 0, 0, 10, 1.0])
    good = warm.copy()
    good[-1] = 0.8
    collapsed = good.copy()
    collapsed[:2] /= 2
    np.testing.assert_allclose([objective(v, warm, s) for v in (warm, good, collapsed)], [1, 0.8, 1.05])


def test_matching_is_exact_per_clip_and_uses_private_rng():
    origins = np.arange(576, 960)
    accepted = origins % 7 == 0
    np.random.seed(5)
    before = np.random.get_state()
    r = matched_random(accepted, origins, 96, 123)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    for clip in np.unique(origins // 96):
        m = origins // 96 == clip
        assert r[m].sum() == accepted[m].sum()
    assert not np.array_equal(r, accepted)
    assert not matched_random(np.zeros(384, bool), origins, 96, 123).any()


def test_rejected_arrival_has_zero_observed_change():
    before = np.arange(7.0)
    x = features(before, before - before, np.arange(6), 577, 96, np.zeros(19))
    assert len(x) == 42
    np.testing.assert_array_equal(x[7:14], np.zeros(7))
    assert x[21] == 0 and x[22] == 1


def test_late_labels_cannot_change_frozen_admission_model():
    s = settings()
    s["regressors"]["nonlinear"]["max_iter"] = 3
    rng = np.random.default_rng(5)
    origins = np.tile(np.arange(128, 1089, 32), 2)
    warm = np.array([4, 0.01, 0.001, 0, 0, 10, 1.0])
    before = np.tile(warm, (62, 1))
    after = before[:, None, :] + rng.normal(0, 0.0001, (62, 6, 7))
    a = dict(
        before=before,
        after=after,
        origins=origins,
        seeds=np.repeat([5003, 5009], 31),
        state=rng.normal(size=(62, 23)),
        image=rng.normal(size=(62, 42)),
    )
    models, info = fit_forecasters(a, warm, s)
    changed = dict(a, after=after.copy())
    changed["after"][origins + 1 > 512] *= 2
    other, second = fit_forecasters(changed, warm, s)
    assert info == second
    for design in ("state", "image"):
        for seed in (5003, 5009):
            np.testing.assert_array_equal(
                models[design][seed].predict(a[design]), other[design][seed].predict(a[design])
            )


def test_source_bootstrap_retains_both_seed_rows():
    s = settings()
    result = paired(np.array([[0.0], [2.0], [10.0], [12.0]]), np.array(["a", "a", "b", "b"]), s)
    np.testing.assert_array_equal(result["mean"], [6.0])
    np.testing.assert_array_equal(result["ci95"], [[1.0, 11.0]])


def test_live_calibration_includes_rejected_candidates():
    records = [dict(actual_delta=y, prediction=p, accepted=p < 0) for y, p in ((-2, -1), (2, -1), (-2, 1), (2, 1))]
    s = diagnostic_scores(records, 1e-10)
    assert s["n"] == 4 and s["balanced_sign_accuracy"] == 0.5
    np.testing.assert_allclose(s["rmse"], np.sqrt(5))
    assert s["mae"] == s["zero_mae"] == 2
