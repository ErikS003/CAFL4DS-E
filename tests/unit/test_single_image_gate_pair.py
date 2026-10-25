"""Cached pair membership preserves temporal exposure and avoids repetition."""

import pytest

from cafl4ds.single_image_gate_pair import select_clips


def records():
    return [
        dict(
            clip_id=f"{history}-{position}",
            history=history,
            position=position,
            keyframe_attributes={"timeofday": "daytime" if position % 2 == 0 else "night"},
        )
        for history in ("first", "second", "third")
        for position in range(4)
    ]


def test_prefix_and_phase_survive_deterministic_cached_extension():
    selected = select_clips(list(reversed(records())), "first", "DN", 0, 10 * 96)
    assert [r["clip_id"] for r in selected[:4]] == [f"first-{i}" for i in range(4)]
    assert len({r["clip_id"] for r in selected}) == 10
    assert [r["keyframe_attributes"]["timeofday"] for r in selected] == ["daytime", "night"] * 5
    assert selected == select_clips(records(), "first", "DN", 0, 10 * 96)


def test_short_intake_cannot_silently_repeat_frames():
    with pytest.raises(ValueError, match="Insufficient"):
        select_clips(records(), "first", "DN", 0, 14 * 96)


def test_wrong_prefix_phase_is_rejected():
    with pytest.raises(ValueError, match="prefix violates"):
        select_clips(records(), "first", "DN", 96, 6 * 96)


def test_duplicate_clip_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_clips([*records(), records()[0]], "first", "DN", 0, 6 * 96)


def test_common_readout_accepts_extra_inherited_development_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_single_image_gate_pair as runner

    monkeypatch.setattr(runner, "restore", lambda *args: (object(), object()))
    monkeypatch.setattr(runner, "measure", lambda *args: dict(metrics={"accuracy": 59.0}))
    monkeypatch.setattr(runner, "read", lambda *args: dict(metrics={"accuracy": 59.0, "development_probe_ce": 0.7}))
    env = (None, SimpleNamespace(optim=None), None)
    runner.common_readout(tmp_path / "new", tmp_path / "old", env, {})
