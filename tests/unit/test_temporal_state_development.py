"""Verify the scientific horizon stays inside a continuous clip and balances domains."""

import pytest

from scripts.run_temporal_state_development import HORIZON, OFFSETS, SCENES, order_cohort


def test_decisions_have_continuous_future_and_balanced_domains() -> None:
    """Across cohorts, both scene and day/night anchors receive equal coverage."""
    clips = [
        {"clip_id": f"{day}-{scene}-{repeat}", "keyframe_attributes": {"timeofday": day, "scene": scene}}
        for day in ("daytime", "night")
        for scene in SCENES
        for repeat in range(4)
    ]
    counts = {}
    for cohort in range(6):
        order = order_cohort(clips, cohort)
        assert len({r["clip_id"] for r in order}) == 24
        anchors = []
        for offset in OFFSETS:
            assert offset // 4 == (offset + HORIZON - 1) // 4
            assert offset % 4 == 2
            attrs = order[offset // 4]["keyframe_attributes"]
            key = (attrs["timeofday"], attrs["scene"])
            counts[key] = counts.get(key, 0) + 1
            anchors.append(attrs)
        assert anchors[0]["scene"] == anchors[1]["scene"]
        assert anchors[0]["timeofday"] != anchors[1]["timeofday"]
    assert len(counts) == 6 and set(counts.values()) == {2}


def test_missing_cell_is_rejected() -> None:
    """Do not silently remove residential or change the cell quotas."""
    with pytest.raises(ValueError, match="exactly four clips"):
        order_cohort([], 0)
