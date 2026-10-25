"""Visibility, anchoring, nested-source isolation and conservative policy controls."""

import copy

import numpy as np

from cafl4ds.health_utility_analysis import fit_models
from cafl4ds.health_utility_bridge import (
    ACTIONS,
    METRICS,
    candidate_menu,
    choose,
    contrast_features,
    extrema,
    fit_contrast,
    predict_contrast,
    temporal_context,
)


def observation() -> dict:
    """A fixed causal observation without any target fields."""
    return {
        "age": 20,
        "history": np.arange(25).reshape(5, 5).tolist(),
        "previews": {
            a: {"loss": 0.5 + i, "rgb_mean": 0.2, "rgb_std": 0.1, "predicted": {m: float(i) for m in METRICS}}
            for i, a in enumerate(ACTIONS)
        },
    }


def test_menu_is_visible_unique_and_deterministic() -> None:
    """Only32 unique indices from the64 arrived candidates enter each batch."""
    first = candidate_menu(2)
    assert first == candidate_menu(2)
    assert len({tuple(x) for x in first}) == 16
    assert all(len(x) == len(set(x)) == 32 and min(x) >= 0 and max(x) < 64 and sorted(x) == x for x in first)


def test_uniform_anchor_and_no_endpoint_feature_leakage() -> None:
    """Uniform remains zero and changing target geometry/scores cannot alter inputs."""
    o = observation()
    altered = copy.deepcopy(o)
    altered["endpoint_scores"] = [999]
    altered["actual_health"] = [999]
    for recipe in ("action", "history", "bridge"):
        assert np.count_nonzero(contrast_features(o, "uniform", recipe)) == 0
        np.testing.assert_array_equal(
            contrast_features(o, "rank_high", recipe), contrast_features(altered, "rank_high", recipe)
        )
    model = fit_contrast(np.eye(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert predict_contrast(model, np.zeros(4)) == 0


def test_extrema_do_not_use_quality_and_keep_duplicate_actions() -> None:
    """Identical response ties pick the first menu, even when quality labels differ."""
    previews = [{"predicted": {"rank": 0.0, "cka": 0.0}, "quality": i} for i in range(16)]
    assert set(extrema(previews).values()) == {0}


def test_policy_abstention_domain_guards_and_stop_are_distinct() -> None:
    """No reliable-looking gain defaults to uniform; negative net value can only stop separately."""
    p = {a: {d: 0.0 for d in ("balanced", "A", "B")} for a in ACTIONS}
    net = {d: 1.0 for d in ("balanced", "A", "B")}
    assert choose(p, net) == "uniform"
    p["rank_high"]["balanced"] = 0.5
    assert choose(p, net) == "rank_high"
    p["rank_high"]["A"] = -1.1
    assert choose(p, net) == "uniform"
    negative = {d: -3.0 for d in net}
    assert choose(p, negative) == "uniform"
    assert choose(p, negative, stop=True) == "hold"


def test_nested_context_excludes_outer_and_training_history_futures() -> None:
    """Changing either excluded history's future cannot change context coefficients."""
    rng = np.random.default_rng(907)
    series = [{"stream": str(g), "health": np.cumsum(rng.normal(size=(33, 5)), axis=0).tolist()} for g in range(6)]
    first = temporal_context(series, 0, {"0", "1"})
    changed = copy.deepcopy(series)
    for g in (0, 1):
        changed[g]["health"][5:] = (np.array(changed[g]["health"])[5:] + 999).tolist()
    second = temporal_context(changed, 0, {"0", "1"})
    assert first == second


def test_held_history_capability_labels_cannot_change_its_model() -> None:
    """Perturb every held-source development label without changing its fitted fold."""
    rng = np.random.default_rng(313)
    series = [{"stream": str(g), "health": np.cumsum(rng.normal(size=(33, 5)), axis=0).tolist()} for g in range(6)]
    observations, outcomes, holds = [], [], []
    for g in range(6):
        o = observation()
        o.update(parent=g, stream=str(g), history=series[g]["health"][:5])
        observations.append(o)
        holds.append({"scores": {"finetune": {d: 50.0 for d in ("balanced", "A", "B")}}})
        for a in ACTIONS:
            outcomes.append(
                {
                    "parent": g,
                    "action": a,
                    "scores": {"finetune": {d: float(rng.normal(50, 2)) for d in ("balanced", "A", "B")}},
                }
            )
    first = fit_models(observations, outcomes, holds, series)["folds"]["0"]
    changed = copy.deepcopy(outcomes)
    for row in changed:
        if row["parent"] == 0:
            row["scores"]["finetune"] = {d: 999.0 for d in ("balanced", "A", "B")}
    holds[0]["scores"]["finetune"] = {d: -999.0 for d in ("balanced", "A", "B")}
    second = fit_models(observations, changed, holds, series)["folds"]["0"]
    assert first == second
