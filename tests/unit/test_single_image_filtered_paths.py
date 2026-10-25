"""Target meaning, treatment allocation and validation isolation."""

import numpy as np
import pytest

from cafl4ds.single_image_filtered_paths import (
    accuracy_slope,
    development_schedule,
    fit_forecasters,
    probe_scores,
    threshold,
)


def probability_panel(truth_probabilities):
    p = np.asarray(truth_probabilities)
    linear = np.column_stack([p, 1 - p])
    return np.stack([linear, linear])[None]


def test_accuracy_improvement_can_disagree_with_ce():
    # One borderline classification is fixed while an already wrong one worsens.
    before = probe_scores(probability_panel([0.49, 0.2]), [0, 0])
    after = probe_scores(probability_panel([0.51, 0.01]), [0, 0])
    change = after - before
    assert change[0] == 50
    assert change[1] > 0
    assert change[2] < 0
    assert change[3] < 0


def test_probe_scores_preserve_class_tie_and_support_panel_mean():
    panel = probability_panel([0.5, 0.8])
    second = probability_panel([0.4, 0.2])
    p = np.concatenate([panel, second])
    scores = probe_scores(p, [0, 1])
    assert scores[0] == 50
    expected = (probe_scores(panel, [0, 1]) + probe_scores(second, [0, 1])) / 2
    np.testing.assert_allclose(scores, expected)
    with pytest.raises(ValueError):
        probe_scores(p, [0, 1], temperature=0)


def test_accuracy_mapping_cannot_reverse_target_meaning():
    assert accuracy_slope([1, 2], [3, 6], "CE") == 3
    assert accuracy_slope([1, 2], [-3, -6], "CE") == 0
    assert accuracy_slope([0, 0], [2, -1], "Brier") == 0
    assert accuracy_slope([2, 3], [-2, 6], "accuracy") == 1
    assert threshold([-2, -1, 0, 1], 0.6) == 0
    assert not np.any(np.asarray([-2, -1, 0]) > threshold([-2, -1, 0], 0.6))


def test_mask_is_randomized_before_labels_and_sampling_covers_all_phases():
    settings = dict(
        stop=960,
        frames_per_clip=96,
        development_start=192,
        development_probabilities=[0.2, 0.6, 1],
        development_namespace="test",
    )
    origins, mask, rates = development_schedule(settings, "a", 11)
    again = development_schedule(settings, "a", 11)
    for a, b in zip((origins, mask, rates), again):
        np.testing.assert_array_equal(a, b)
    assert mask[:192].all() and len(origins) == 24
    assert np.any(~mask) and mask[rates == 1].all()
    for clip in range(2, 10):
        selected = origins[origins // 96 == clip] % 96
        np.testing.assert_array_equal(selected // 32, [0, 1, 2])


def training_arrays():
    rng = np.random.default_rng(87)
    groups = np.repeat(["fit1", "fit2", "validation"], 40)
    seeds = np.tile(np.repeat([11, 13], 20), 3)
    recipes = np.tile(np.repeat(["full", "mixed"], 10), 6)
    x = rng.normal(size=(120, 3))
    y = x[:, 0] * 0.2
    return dict(
        histories=groups,
        seeds=seeds,
        recipes=recipes,
        state=x[:, :2],
        base=x,
        targets=np.column_stack([y, y * 2, -y, y / 10]),
    )


def test_held_history_values_do_not_enter_validation_fit_or_mapping():
    a = training_arrays()
    settings = dict(
        validation_histories=["validation"],
        trajectory_seeds=[11, 13],
        max_validation_admission=0.6,
        regressors={"linear": {"_target_": "sklearn.linear_model.Ridge", "alpha": 1e-6}},
    )
    _, first = fit_forecasters(a, settings)
    a["targets"][a["histories"] == "validation"] *= 100
    models, second = fit_forecasters(a, settings)
    for key in first["models"]:
        np.testing.assert_array_equal(first["models"][key]["predictions"], second["models"][key]["predictions"])
    assert len(models) == 16 and all(set(v) == {11, 13} for v in models.values())
    assert first["chosen_surrogate"] != "CE"
    assert len(first["gates"]) == 4


def test_rmse_skill_bootstrap_applies_root_after_pooling_errors():
    from cafl4ds.single_image_filtered_paths_analysis import error_skill

    actual = np.array([1, 3, 2, 4])
    prediction = np.array([1, 1, 2, 2])
    groups = np.array(["a", "a", "b", "b"])
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    result = error_skill(actual, prediction, np.zeros(4), groups, draws)
    expected = np.sqrt(np.mean(actual**2)) - np.sqrt(np.mean((actual - prediction) ** 2))
    assert result["rmse"]["mean"] == pytest.approx(expected)
    assert result["mae"]["mean"] == pytest.approx(1.5)


def test_zero_predictions_receive_no_nonzero_direction_credit():
    from cafl4ds.single_image_filtered_paths_analysis import forecast_scores

    result = forecast_scores([0, -1, 1], [0, 0, 0])
    assert result["balanced_sign_accuracy"] == 0
    assert result["nonzero"] == 2 and result["zero"] == 1
