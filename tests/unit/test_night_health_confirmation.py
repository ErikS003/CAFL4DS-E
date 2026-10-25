"""Fixed-model transfer and matched night-target sampling safeguards."""

import numpy as np
import pytest

from cafl4ds.night_health_confirmation import NightHealthConfirmation, compare, resamples


def data():
    rng = np.random.default_rng(104)
    return {
        "controls": rng.normal(size=(18, 6)),
        "damage": rng.normal(size=(18, 2)),
        "dino": rng.normal(size=(18, 2, 2)),
        "dino_sham": rng.normal(size=(18, 2, 2)),
    }


def test_predictions_use_frozen_scaling_after_development_changes():
    development = data()
    study = NightHealthConfirmation()
    models = study.fit_development(development)
    x, h, sham = (development[k].copy() for k in ("controls", "dino", "dino_sham"))
    expected = study.predict(models, x, h, sham)
    development["damage"][:] = 1e6
    development["controls"][:] = 1e6
    for name, values in study.predict(models, x, h, sham).items():
        np.testing.assert_array_equal(values, expected[name])


def test_target_is_class_balanced_and_damage_sign_is_correct():
    labels = np.array([0, 0, 0, 1])
    before = np.eye(2)[labels][None].repeat(2, 0)
    after = before.copy()
    after[:, :3] = before[:, :3, ::-1]
    target, damage = compare(before, after, labels, np.array([[1, 1], [2, 0]]), np.ones((2, 4), int))
    assert target["gain_pp"] == -50
    np.testing.assert_allclose(damage, 50)


def test_resamples_preserve_class_sizes_and_pairing():
    labels = np.r_[np.zeros(17, int), np.ones(5, int)]
    panel, query = resamples(labels, panels=8, repetitions=20)
    np.testing.assert_array_equal(panel.sum(1), 8)
    np.testing.assert_array_equal(query[:, labels == 0].sum(1), 17)
    np.testing.assert_array_equal(query[:, labels == 1].sum(1), 5)
    probability = np.broadcast_to(np.eye(2)[labels], (8, 22, 2))
    target, damage = compare(probability, probability, labels, panel, query)
    assert target["gain_pp"] == 0
    np.testing.assert_array_equal(damage, 0)


def test_models_are_equivariant_to_old_target_units():
    a, b = data(), data()
    b["damage"] = b["damage"] * 7 + 3
    study = NightHealthConfirmation()
    pa = study.predict(study.fit_development(a), a["controls"], a["dino"], a["dino_sham"])
    pb = study.predict(study.fit_development(b), a["controls"], a["dino"], a["dino_sham"])
    for name in pa:
        np.testing.assert_allclose(pb[name], pa[name] * 7 + 3, atol=1e-12)


def test_missing_class_rejected():
    with pytest.raises(ValueError, match="Both night classes"):
        resamples(np.zeros(5, int))


def test_transfer_pass_does_not_promote_filter_and_precision_is_required():
    records = []
    for group in range(6):
        for gain in (-2, 2, 0):
            records.append(
                {
                    "source": f"source_{group}",
                    "target": {
                        "gain_pp": gain,
                        "support": {"half_width": 0.1},
                        "crossed": {"lower": gain - 0.2, "upper": gain + 0.2, "half_width": 0.2},
                    },
                }
            )
    damage = -np.array([r["target"]["gain_pp"] for r in records])
    forecasts = {"dino": damage, **{name: np.zeros(18) for name in ("mean", "controls", "sham")}}
    study = NightHealthConfirmation()
    result = study.summarize(records, forecasts, np.tile(damage, (20, 1)))
    assert result["association_confirmed"] and result["calibration_passed"]
    assert not result["filter_qualified"]
    for record in records:
        record["target"]["support"]["half_width"] = 0.6
    assert not study.summarize(records, forecasts, np.tile(damage, (20, 1)))["association_confirmed"]
