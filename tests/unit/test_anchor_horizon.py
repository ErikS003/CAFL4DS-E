"""The reference floor cannot be reset or purchased by another capability's gain."""

import numpy as np
import pytest

from cafl4ds.anchor_horizon import AnchorHorizonStudy


def comparison(lower):
    values = np.asarray(lower, float)
    return {"crossed": {"lower": values, "half_width": np.ones(3) * 0.1}, "mean_gain_pp": values}


def test_large_night_gain_cannot_purchase_excessive_day_loss():
    study = AnchorHorizonStudy()
    result = study.shadow(comparison([3, 0, 6]), comparison([2, -1.01, 5.01]), {"rank": 1, "variance": 1})
    assert result["reward"] > 0.99
    assert not result["checks"]["day_budget"]
    assert not result["retrospective_shadow_accept"]


def test_local_benefit_does_not_reset_cumulative_anchor_loss():
    study = AnchorHorizonStudy()
    result = study.shadow(comparison([0.5, 0.5, 0.5]), comparison([-1.1, -1.1, -1.1]), {"rank": 1, "variance": 1})
    assert result["checks"]["marginal_gain"]
    np.testing.assert_allclose(result["remaining_budget_pp"], [-0.1, -0.1])
    assert not result["retrospective_shadow_accept"]


def test_recovery_restores_headroom_and_geometry_still_blocks():
    study = AnchorHorizonStudy()
    result = study.shadow(comparison([0.5, 0.5, 0.5]), comparison([1, 1, 1]), {"rank": 0.59, "variance": 1})
    np.testing.assert_allclose(result["remaining_budget_pp"], [2, 2])
    assert not result["checks"]["rank_floor"]
    assert not result["retrospective_shadow_accept"]


def test_passing_shadow_is_not_a_confident_online_acceptance():
    result = AnchorHorizonStudy().shadow(
        comparison([0.5, 0.5, 0.5]), comparison([0, -0.5, 0.5]), {"rank": 1, "variance": 1}
    )
    assert result["retrospective_shadow_accept"]
    assert result["online_decision"] == "unknown_unqualified"


def test_nonfinite_or_invalid_scales_are_rejected():
    with pytest.raises(ValueError, match="Positive"):
        AnchorHorizonStudy(gain_scale_pp=0)
    with pytest.raises(ValueError, match="Finite"):
        AnchorHorizonStudy().shadow(comparison([np.nan, 0, 0]), comparison([0, 0, 0]), {"rank": 1, "variance": 1})


def test_horizons_and_slots_do_not_inflate_independent_source_count():
    rows, health = [], []
    for source in ("a", "b"):
        for slot in (0, 16):
            for horizon in (1, 8, 32):
                value = 0.1 if horizon == 1 else -0.1
                marginal = comparison([value] * 3)
                rows.append(
                    {
                        "source": source,
                        "slot": slot,
                        "horizon": horizon,
                        "marginal": marginal,
                        "shadow": {"retrospective_shadow_accept": False},
                    }
                )
                health.append(
                    {
                        "source": source,
                        "slot": slot,
                        "horizon": horizon,
                        "effect": dict.fromkeys(("rank", "variance", "covariance", "cka", "cosine"), value),
                        "paid_h1_persistence": dict.fromkeys(("rank", "variance", "covariance", "cka", "cosine"), 0.1),
                    }
                )
    result = AnchorHorizonStudy().summarize(rows, health)
    assert result["independent_sources"] == 2
    assert result["horizons"]["32"]["contrasts"] == 4
    assert list(result["cross_horizon"]["8"]["nonzero_sign_flip_count_vs_h1"]) == [4, 4, 4]
    assert result["health_forecasts"]["1"]["rank"]["paid_h1_persistence_mse"] == 0
    assert not result["filter_qualified"]
