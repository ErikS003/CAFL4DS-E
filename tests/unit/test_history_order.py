import pytest

from cafl4ds.history_order import CASES, STATES, history_indices, paired_interval, preference
from scripts.decode_history_video import timestamps


def test_fixed_age_and_primary_source_budget():
    assert len(CASES) == 34 and set(STATES.values()) == {8448}
    assert sum(c["context"] == "day" for c in CASES.values()) == 24
    values = list(range(12))
    ordinary = paired_interval(values)
    adjusted = paired_interval(values, True)
    assert ordinary["mean_pp"] == 5.5 and adjusted["ci95_pp"][0] < ordinary["ci95_pp"][0]
    with pytest.raises(ValueError):
        paired_interval([1, 2, 3])


def test_order_indices_preserve_image_multiset_but_change_recent_past():
    dn = history_indices("DN")
    nd = history_indices("ND")
    assert sorted(dn) == sorted(nd) and dn[-128:] != nd[-128:]
    assert len(dn) // 32 == len(nd) // 32 == 264


def test_real_timestamp_sampling_has_no_duplicate_frames():
    log = "\n".join(f"n: {n} pts: {n} pts_time:{n / 30:.6f} s:1280x720" for n in range(1000))
    selected = timestamps(log)
    assert len(selected) == 512 and len({r["source_frame_index"] for r in selected}) == 512
    assert all(r["pts_seconds"] >= k / 16 for k, r in enumerate(selected))


def test_short_video_is_rejected_without_interpolation():
    log = "\n".join(f"n: {n} pts: {n} pts_time:{n / 30:.6f} s:1280x720" for n in range(600))
    with pytest.raises(ValueError):
        timestamps(log)


def test_retention_guards_remain_required():
    effects = {m: {"mean_pp": 2, "ci95_pp": [1, 3]} for m in ("balanced", "A", "B")}
    assert preference(effects)["full"]
    effects["B"] = {"mean_pp": -2, "ci95_pp": [-3, -1]}
    assert not preference(effects)["full"]
