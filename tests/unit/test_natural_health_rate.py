"""Check continuity, long-clock alarms and preservation of source order."""

from pathlib import Path

import numpy as np
import pytest

from cafl4ds.data.attributes import _filter_valid_records
from cafl4ds.dnd_health_rate import HealthRate
from cafl4ds.natural_health_rate import NaturalHealthRate, OrderedFrames, paced_mask, path_metrics
from scripts.prepare_natural_health_rate import timestamps, warm_ids
from scripts.run_natural_health_rate import summarize


def test_prior_clock_and_decisions_are_preserved():
    bank = np.random.default_rng(81).normal(size=(32, 8))
    old, new = HealthRate(), NaturalHealthRate()
    for step in range(1, 449):
        take, numerator = old.admit(step)
        assert new.admit(step) == (take, numerator / 4)
        features = bank if step < 140 or 220 < step < 330 else np.zeros_like(bank)
        assert old.observe(step, features) == new.observe(step, features)
    assert [r["step"] for r in old.events if r["alarm"]] == [r["step"] for r in new.events if r["alarm"]]


def test_long_stream_keeps_observing_and_training():
    bank = np.random.default_rng(82).normal(size=(32, 8))
    controller = NaturalHealthRate()
    takes = []
    for step in range(1, 2049):
        takes.append(controller.admit(step)[0])
        controller.observe(step, bank if step <= 132 else np.zeros_like(bank))
    alarms = [r["step"] for r in controller.events if r["alarm"]]
    assert alarms[0] == 148 and alarms[-1] > 2000
    assert set(np.diff(alarms)) == {16}
    assert sum(takes) == 148 + (2048 - 148) // 4
    np.testing.assert_array_equal(controller.reference, bank)


def test_recovery_uses_arrivals_and_later_alarm_resets_it():
    bank = np.random.default_rng(83).normal(size=(32, 8))
    controller = NaturalHealthRate()
    controller.observe(132, bank)
    controller.observe(140, np.zeros_like(bank))
    assert controller.observe(148, np.zeros_like(bank))
    groups = [[controller.admit(s) for s in range(start, start + 32)] for start in (149, 181, 213, 245)]
    assert [sum(v[0] for v in group) for group in groups] == [8, 16, 24, 32]
    for step in range(156, 501, 8):
        controller.observe(step, bank)
    assert not controller.observe(508, np.zeros_like(bank))
    assert controller.observe(516, np.zeros_like(bank))
    assert [controller.admit(s)[0] for s in range(517, 521)] == [False, False, False, True]


def test_auxiliary_checkpoints_do_not_advance_persistence():
    bank = np.random.default_rng(84).normal(size=(32, 8))
    controller = NaturalHealthRate()
    controller.observe(132, bank)
    controller.observe(140, np.zeros_like(bank))
    assert not controller.observe(144, np.zeros_like(bank))
    assert controller.streak == 1
    controller.observe(148, bank)
    assert controller.streak == 0
    with pytest.raises(ValueError, match="replaced"):
        controller.observe(132, bank)


def test_no_alarm_stream_continues_full_training():
    bank = np.random.default_rng(85).normal(size=(32, 8))
    controller = NaturalHealthRate()
    for step in range(1, 1025):
        assert controller.admit(step) == (True, 1)
        assert not controller.observe(step, bank)
    assert controller.last_alarm is None
    assert paced_mask(4096, None, 0).all()
    assert paced_mask(4096, 4096, 0).all()


@pytest.mark.parametrize("count", [0, 1, 11, 51, 100])
def test_pacing_preserves_exact_budget_and_prefix(count):
    mask = paced_mask(248, 148, count)
    assert mask[:148].all() and int(mask[148:].sum()) == count
    for end in range(1, 101):
        assert int(mask[148 : 148 + end].sum()) == end * count // 100


def test_ordered_batches_cross_clip_boundaries_without_lookahead(tmp_path):
    clips = []
    for slot, (start, stop) in enumerate(((0, 45), (45, 101))):
        path = tmp_path / f"{slot}.npy"
        np.save(path, np.arange(start, stop, dtype=np.uint8)[:, None, None, None])
        clips.append({"frames_path": str(path), "attributes": {"timeofday": "unused"}})
    stream = OrderedFrames(clips)
    for i in range(3):
        np.testing.assert_array_equal(stream.batch(i).ravel(), np.arange(i * 32, (i + 1) * 32))
    with pytest.raises(IndexError):
        stream.batch(3)


def test_warm_exclusion_matches_established_partition():
    metadata, records = [], []
    for light, weather, scene, count in [
        ("daytime", "clear", "city street", 60),
        ("night", "rainy", "city street", 60),
        ("daytime", "clear", "residential", 60),
        ("daytime", "clear", "tunnel", 3),
    ]:
        for i in range(count):
            clip = f"{light}-{weather}-{scene}-{i}"
            attrs = {"timeofday": light, "weather": weather, "scene": scene}
            metadata.append({"clip_id": clip, "attributes": attrs})
            records.append((Path(f"{clip}.jpg"), (light, weather), scene, {}))
    metadata.append({"clip_id": "unknown", "attributes": {"timeofday": "undefined"}})
    expected = _filter_valid_records(
        records, min_canary_count=50, partition_role="warm", warm_fraction=0.2, partition_seed=20260921
    )
    assert warm_ids(metadata) == {r[0].stem for r in expected}


def test_timestamp_gaps_duplicates_and_source_order_are_retained():
    pts = [0, 0.03, 0.063, 0.063, 0.5, 0.49, 0.6]
    log = "\n".join(f"n: {i} pts: {int(t * 1000)} pts_time:{t} s:1280x720" for i, t in enumerate(pts))
    trace = timestamps(log, 100)
    assert [r["source_frame_index"] for r in trace] == [0, 2, 4, 6]
    assert [r["pts_seconds"] for r in trace] == [0, 0.063, 0.5, 0.6]


def test_deficits_use_arrivals_instead_of_invented_interclip_time():
    curve = np.array([[50, 50, 50], [54, 55, 53], [52, 51, 53], [53, 54, 52]])
    result = path_metrics(curve, [0, 132, 260, 388])
    np.testing.assert_array_equal(result[3:6], [2, 4, 1])
    np.testing.assert_array_equal(result[9:12], [128, 128, 0])
    assert result[-1] == 4


def test_reader_maxima_are_distinct_from_pooled_drawdown_and_stop_is_exact():
    points = [0, 132, 260, 388]
    levels = np.array([[50, 54, 50, 54], [50, 50, 54, 54], [50, 52, 52, 54]])
    readers = np.repeat(levels[:, :, None], 3, axis=2)
    observations = {str(p): {"panel_accuracy_pp": readers[:, i]} for i, p in enumerate(points)}
    records = {arm: {"updates": 388, "observations": observations} for arm in ("full", "adaptive", "quarter", "paced")}
    result = summarize(records, points, 132)
    full, stop = result["rules"]["full"], result["rules"]["permanent"]
    assert full["metrics"][-1] == 0
    assert np.mean(np.array(full["reader_metrics"])[:, -1]) == pytest.approx(4 / 3)
    np.testing.assert_array_equal(stop["curve"][-1], [52, 52, 52])
    np.testing.assert_array_equal(np.array(stop["reader_curves"])[:, -1], readers[:, 1])
    assert stop["updates"] == 132
    assert result["uncertainty_intervals"] is None and not result["controller_qualified"]
