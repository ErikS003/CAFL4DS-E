"""Check exact switching boundaries and exclusion of future/label information from predictors."""

import pytest

from cafl4ds.action_identification import STATE_FEATURES, action_at, response_features


def test_switching_is_after_six_updates_and_exposure_matched() -> None:
    """Reversed schedules have equal alpha exposure, but different order available to the model."""
    forward = [action_at("zero_to_one", i) for i in range(12)]
    backward = [action_at("one_to_zero", i) for i in range(12)]
    assert forward == [0.0] * 6 + [1.0] * 6
    assert backward == [1.0] * 6 + [0.0] * 6
    assert sum(forward) == sum(backward)
    with pytest.raises(ValueError):
        action_at("zero_to_one", 12)


def test_future_outcomes_and_post_action_health_cannot_enter_features() -> None:
    """Perturbing forbidden information leaves the compact model input exactly unchanged."""
    state = {name: float(i + 1) for i, name in enumerate(STATE_FEATURES)}
    observed = response_features("half_to_one", state, True)
    state.update({"holdout_refit_ce_balanced": -999.0, "post_health_cka_drift": 999.0, "seed": 12345.0})
    assert response_features("half_to_one", state, True) == observed
    assert len(observed) == 21
    action_only = response_features("half_to_one", {}, False)
    assert len(action_only) == 4
    assert response_features("one_to_half", state, True) != observed
