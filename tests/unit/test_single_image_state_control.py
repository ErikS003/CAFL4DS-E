"""Input isolation and inherited frozen control behavior."""

import numpy as np
from unittest.mock import patch
from cafl4ds.single_image_state_control import state_vector
from cafl4ds.single_image_filter import matched_random
from cafl4ds.single_image_filtered_paths_analysis import forecast_scores


def test_state_inputs_have_no_candidate_argument_and_track_lag():
    observed = [(np.arange(7.0) + i, None, None) for i in range(4)]
    with patch("cafl4ds.single_image_state_control.optimizer_features", return_value=np.arange(6.0)):
        x = state_vector(None, None, observed, 576, {"frames_per_clip": 96})
    assert x.shape == (23,)
    np.testing.assert_array_equal(x[:7], observed[-1][0])
    np.testing.assert_array_equal(x[7:14], np.ones(7))
    np.testing.assert_array_equal(x[-3:], [576, 1, 0])


def test_random_control_matches_each_clip_even_for_zero_admissions():
    y = np.array([False] * 96 + [True] * 37 + [False] * 59 + [True] * 96)
    a = matched_random(y, np.arange(576, 864), 96, 12)
    np.testing.assert_array_equal(a.reshape(3, 96).sum(1), [0, 37, 96])
    assert not np.array_equal(a, y)


def test_prediction_direction_has_no_credit_for_zero_predictions():
    r = forecast_scores(np.array([-1.0, 0.0, 1.0]), np.zeros(3))
    assert r["balanced_sign_accuracy"] == 0 and r["zero"] == 1
