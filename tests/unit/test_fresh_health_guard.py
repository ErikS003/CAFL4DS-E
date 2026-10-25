"""Guard against source leakage, pseudoreplication and unfair stopping budgets."""

import copy

import numpy as np
import pytest
from omegaconf import OmegaConf

from cafl4ds.fresh_health_guard import age_cutoff, summarize, validate_sources


def config():
    return OmegaConf.create(
        {
            "parents": {"day_end": 128, "night_end": 256, "return_early": 264},
            "landmarks": [0, 4, 8, 16, 32],
            "primary_family": 12,
            "minimum_gain_pp": 0.25,
            "domain_budget_pp": 1.0,
            "age_boundary": 200,
            "age_early_cutoff": 32,
            "age_mature_cutoff": 4,
        }
    )


def intake():
    clips, streams = [], []
    for s in range(6):
        periods = []
        for v in range(3):
            ids = []
            for i in range(8):
                identity = f"s{s}v{v}i{i}"
                ids.append(identity)
                clips.append(
                    {
                        "clip_id": identity,
                        "keyframe_attributes": {
                            "timeofday": "night" if v == 1 else "daytime",
                            "scene": "highway" if i % 2 else "city street",
                        },
                    }
                )
            periods.append(ids)
        streams.append({"periods": periods})
    return {"clips": clips, "streams": streams}


def cases():
    values = []
    for s in range(6):
        for context in config().parents:
            for seed in range(3):
                points = {
                    str(k): {"panel_accuracy_pp": np.full((3, 3), 50 + k / 32 + s).tolist()} for k in config().landmarks
                }
                policies = {
                    mode: {"committed_updates": step}
                    for mode, step in (
                        ("full", 32),
                        ("fixed8", 8),
                        ("age", 4),
                        ("pause", 16),
                        ("rollback", [0, 4, 8][seed]),
                    )
                }
                policies["rollback"]["observations"] = [{"decision": {"active": [False, True]}}]
                values.append(
                    {"source": f"s{s}", "context": context, "seed": seed, "checkpoints": points, "policies": policies}
                )
    return values


def test_freshness_rejects_previously_seen_sources():
    data = intake()
    validate_sources(data, set())
    with pytest.raises(ValueError):
        validate_sources(data, {data["clips"][0]["clip_id"]})


def test_shared_clips_cannot_be_independent_histories():
    data = intake()
    data["streams"][1]["periods"][0] = data["streams"][0]["periods"][0]
    with pytest.raises(ValueError):
        validate_sources(data, set())


def test_fixed_enrollment_strata_cannot_change():
    data = intake()
    data["clips"][0]["keyframe_attributes"]["timeofday"] = "night"
    with pytest.raises(ValueError):
        validate_sources(data, set())


def test_age_schedule_has_a_frozen_boundary():
    assert age_cutoff(199, config()) == 32
    assert age_cutoff(200, config()) == 4


def test_random_stopping_matches_budget_without_matching_case_actions():
    result = summarize(cases(), config())
    for value in result["matched_committed_budget"].values():
        assert value["guard_mean_committed"] == pytest.approx(value["random_mean_committed"])
    assert result["mean_gains_pp"]["matched_random"] == pytest.approx(result["mean_gains_pp"]["rollback"])
    assert not result["fresh_source_lead"]


def test_intervals_use_histories_not_seed_or_panel_repeats():
    values = cases()
    original = summarize(values, config())
    changed = copy.deepcopy(values)
    for row in changed:
        for checkpoint in row["checkpoints"].values():
            checkpoint["panel_accuracy_pp"] *= 2
    result = summarize(changed, config())
    assert result["primary_comparisons"] == original["primary_comparisons"]
    assert len(result["source_means_pp"]["rollback"]) == 6
