"""Forecast timing must survive changes to unavailable nonlinear targets."""

import numpy as np
import pytest

from cafl4ds.single_image_nonlinear import chronological_forecasts, previous_effect, view_seed


@pytest.mark.parametrize("family", ["ridge", "rbf", "extra_trees"])
def test_future_and_same_origin_labels_excluded(family):
    rng = np.random.default_rng(3)
    origins = np.repeat(np.arange(0, 100, 10), 2)
    x, y = rng.normal(size=(20, 3)), rng.normal(size=(20, 2))
    settings = dict(
        prefix_end=41,
        validation_end=21,
        test_start=50,
        models=[family],
        designs=["state"],
        ridge_alpha=10,
        rbf_alpha=1,
        trees=10,
        tree_depth=3,
        tree_min_leaf=2,
        model_seed=1,
    )
    a, selection = chronological_forecasts({"state": x}, y, origins, settings)
    changed = y.copy()
    changed[origins >= 60] += 1000
    b, selection2 = chronological_forecasts({"state": x}, changed, origins, settings)
    assert selection["selected"] == selection2["selected"]
    np.testing.assert_array_equal(a[f"online_{family}_state"][:4], b[f"online_{family}_state"][:4])
    np.testing.assert_array_equal(a[f"frozen_{family}_state"], b[f"frozen_{family}_state"])
    assert selection["online_admitted_rows"]["60"] == 12


def test_previous_effect_stays_in_history_and_past():
    groups = np.array(["a", "b", "a", "b", "a"])
    origins = np.array([2, 2, 4, 4, 5])
    y = np.arange(5.0)[:, None]
    np.testing.assert_array_equal(previous_effect(y, groups, origins).ravel(), [0, 0, 0, 1, 0])


def test_view_streams_separate_and_repeatable():
    seeds = [view_seed("a", "g", t, d) for t in (128, 160) for d in range(6)]
    assert len(set(seeds)) == 12
    assert seeds[0] == view_seed("a", "g", 128, 0)
