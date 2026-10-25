import numpy as np

from cafl4ds.single_image_transition import cluster_intervals, frozen_predictions, last_effect, transition_features


def test_boundary_is_preupdate_source_position():
    np.testing.assert_array_equal(transition_features([95, 96, 97, 192]), [[0, 95], [1, 0], [0, 1], [1, 0]])


def test_previous_target_is_seed_history_and_time_specific():
    histories = np.array(["a", "a", "b", "a", "a", "a"])
    seeds = np.array([1, 2, 1, 1, 2, 1])
    origins = np.array([10, 10, 10, 20, 20, 21])
    y = np.arange(1.0, 7.0)[:, None]
    np.testing.assert_array_equal(last_effect(y, histories, seeds, origins).ravel(), [0, 0, 0, 1, 2, 1])


def example():
    rng = np.random.default_rng(4)
    origins = np.repeat(np.arange(0, 100, 10), 2)
    seeds = np.tile([1, 2], 10)
    x, y = rng.normal(size=(20, 4)), rng.normal(size=(20, 2))
    settings = dict(
        prefix_end=41, test_start=50, frames_per_clip=30, trees=16, tree_depth=3, tree_min_leaf=2, model_seed=1
    )
    return x, y, seeds, origins, settings


def test_frozen_fit_and_phase_control_never_see_test_labels():
    x, y, seeds, origins, settings = example()
    first = frozen_predictions({"state_transition": x}, y, seeds, origins, np.zeros_like(y), settings)
    changed = y.copy()
    changed[origins > 41] *= 10000
    second = frozen_predictions({"state_transition": x}, changed, seeds, origins, np.zeros_like(y), settings)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_seed_specific_fit_does_not_borrow_other_seed_labels():
    x, y, seeds, origins, settings = example()
    first = frozen_predictions({"image": x}, y, seeds, origins, np.zeros_like(y), settings)
    changed = y.copy()
    changed[seeds == 2] += 1000
    second = frozen_predictions({"image": x}, changed, seeds, origins, np.zeros_like(y), settings)
    for key in first:
        np.testing.assert_array_equal(first[key][seeds[origins >= 50] == 1], second[key][seeds[origins >= 50] == 1])


def test_phase_mean_uses_matching_prefix_phase():
    x, y, seeds, origins, settings = example()
    prediction = frozen_predictions({"image": x}, y, seeds, origins, np.zeros_like(y), settings)["phase_mean"]
    for i, (seed, origin) in enumerate(zip(seeds[origins >= 50], origins[origins >= 50], strict=True)):
        matching = (seeds == seed) & (origins + 1 <= 41) & (origins % 30 == origin % 30)
        np.testing.assert_array_equal(prediction[i], y[matching].mean(0))


def test_cluster_bootstrap_retains_optimizer_seed_replicates():
    # Every source history has the same aggregate difference; its seeds differ.
    groups = np.repeat(["a", "b", "c"], 3)
    y = np.zeros((9, 1))
    prediction = np.tile([0.0, 1.0, 2.0], 3)[:, None]
    ci = cluster_intervals(y, prediction, np.zeros_like(y), groups, 500, 3)
    np.testing.assert_allclose(ci["mae"], [[1, 1]])
    np.testing.assert_allclose(ci["mse"], [[5 / 3, 5 / 3]])
