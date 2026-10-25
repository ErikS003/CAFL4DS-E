"""Check structural anchoring and exclusion of post-action information."""

import numpy as np
from sklearn.linear_model import Ridge

from cafl4ds.response_calibration import RECIPE_FIELDS, anchored_features


def test_reference_zero_survives_state_change_and_fitting() -> None:
    """No fitted model can invent an alpha=1 effect through a free state offset."""
    fields = RECIPE_FIELDS["health"]
    centers = dict.fromkeys(fields, 2.0)
    scales = dict.fromkeys(fields, 0.5)
    for recipe, dimension in zip(RECIPE_FIELDS, (3, 6, 12, 18), strict=True):
        for value in (-100.0, 0.0, 100.0):
            state = dict.fromkeys(fields, value)
            reference = anchored_features("constant_1", state, recipe, centers, scales)
            assert len(reference) == dimension
            assert not any(reference.values())
            alternative = anchored_features("half_to_one", state, recipe, centers, scales)
            model = Ridge(alpha=10, fit_intercept=False).fit([list(alternative.values())], [1.0])
            assert np.array_equal(model.predict([list(reference.values())]), [0.0])


def test_labels_and_future_health_cannot_enter_design() -> None:
    """Only declared pre-action state fields enter interactions."""
    fields = RECIPE_FIELDS["health"]
    state = dict.fromkeys(fields, 3.0)
    centers, scales = dict.fromkeys(fields, 0.0), dict.fromkeys(fields, 1.0)
    before = anchored_features("one_to_zero", state, "health", centers, scales)
    state.update(holdout_refit_ce_balanced=-999.0, post_health_cka_drift=999.0)
    assert anchored_features("one_to_zero", state, "health", centers, scales) == before
