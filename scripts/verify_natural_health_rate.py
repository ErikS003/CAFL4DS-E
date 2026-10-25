"""Independently reconstruct stream order, decisions and raw capability scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.data.attributes import _filter_valid_records
from scripts.run_natural_health_rate import CONFIG
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read, write
from scripts.verify_dnd_signal_diagnostic import geometry


def intake_audit(config):
    intake = ROOT / config.intake
    target = intake / "independent-verification.json"
    assert not target.exists(), "Preserve the first intake audit"
    check(read(intake / "preparation-lock.json")["hashes"])
    metadata = read(intake / "metadata.json")["records"]
    valid = []
    for r in metadata:
        a = r["attributes"]
        if not isinstance(a, dict):
            continue
        timeofday, weather, scene = (a.get(k) for k in ("timeofday", "weather", "scene"))
        if all(isinstance(v, str) and v not in ("", "undefined") for v in (timeofday, weather, scene)):
            valid.append((Path(r["clip_id"] + ".jpg"), (timeofday, weather), scene, {}))
    warm = {
        r[0].stem
        for r in _filter_valid_records(
            valid, min_canary_count=50, partition_role="warm", warm_fraction=0.2, partition_seed=20260921
        )
    }
    normalized = hashlib.sha256(
        json.dumps(sorted((p.name, regime, scene) for p, regime, scene, _ in valid), sort_keys=True).encode()
    ).hexdigest()
    historical_partition = ROOT / "outputs/scheduling/20261013-intake/selection.json"
    assert normalized == read(historical_partition)["partition_metadata_sha256"]
    selection = read(intake / "selection.json")
    assert warm == set(selection["warm_clip_ids"])
    excluded = warm | set(selection["monitor_clip_ids"]) | set(selection["evaluation_clip_ids"])
    index = read(ROOT / config.archive_index)
    assert all(a["offset"] < b["offset"] for a, b in zip(index["entries"][:-1], index["entries"][1:], strict=True))
    prefix = [
        (i, e)
        for i, e in enumerate(index["entries"])
        if "/train/" in e["path"] and Path(e["path"]).stem not in excluded
    ][: config.max_clips]
    assert [e["path"] for _, e in prefix] == [e["path"] for e in selection["clips"]]
    assert [i for i, _ in prefix] == [e["archive_position"] for e in selection["clips"]]
    frames = read(ROOT / config.frames / "intake.json")
    assert frames["complete"] and len(frames["clips"]) <= len(prefix)
    assert [c["path"] for c in frames["clips"]] == [e["path"] for _, e in prefix[: len(frames["clips"])]]
    metadata_by_id = {r["clip_id"]: r["attributes"] for r in metadata}
    counts, total, hashes = {}, 0, {}
    for clip in frames["clips"]:
        assert clip["clip_id"] not in excluded
        assert clip["attributes"] == metadata_by_id[clip["clip_id"]]
        assert digest(clip["frames_path"]) == clip["frames_sha256"]
        path = Path(clip["frames_path"])
        with path.open("rb") as handle:
            pixels = np.load(handle, allow_pickle=False)
        assert pixels.dtype == np.uint8 and list(pixels.shape) == clip["shape"]
        assert tuple(pixels.shape[1:]) == (64, 64, 3)
        stamps = clip["frames"]
        assert len(stamps) == len(pixels)
        assert np.all(np.diff([r["pts_seconds"] for r in stamps]) > 0)
        assert np.all(np.diff([r["source_frame_index"] for r in stamps]) > 0)
        for row, rgb in zip(stamps, pixels, strict=True):
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb64_sha256"]
        native = (path.parent / "decoder.log").read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)", native)
        sampled = []
        for number, pts, seconds, width, height in matches:
            t = float(seconds)
            if len(sampled) == len(stamps):
                break
            if t >= len(sampled) / 16 and (not sampled or t > sampled[-1][2]):
                assert (int(width), int(height)) == (1280, 720)
                sampled.append((int(number), int(pts), t))
        assert sampled == [(r["source_frame_index"], r["pts"], r["pts_seconds"]) for r in stamps]
        total += len(pixels)
        attrs = clip["attributes"] if isinstance(clip["attributes"], dict) else {}
        label = "/".join(str(attrs.get(k, "unknown")) for k in ("timeofday", "weather", "scene"))
        counts[label] = counts.get(label, 0) + len(pixels)
        for p in (path, path.parent / "source.json", path.parent / "manifest.json", path.parent / "decoder.log"):
            hashes[str(p)] = digest(p)
    assert total == frames["frames_decoded"] <= config.max_batches * 32
    assert frames["frames_used"] == total // 32 * 32
    assert frames["batches"] == total // 32 >= config.minimum_batches
    assert total == config.max_batches * 32 or len(frames["clips"]) == config.max_clips
    assert frames["download_bytes"] <= config.network_budget_gib * 2**30
    for p in (intake / "selection.json", intake / "metadata.json", ROOT / config.frames / "intake.json"):
        hashes[str(p)] = digest(p)
    hashes[str(historical_partition)] = digest(historical_partition)
    write(
        target,
        {
            "passed": True,
            "source_sha256": digest(Path(__file__)),
            "clips": len(frames["clips"]),
            "frames": total,
            "batches": total // 32,
            "warm_overlap": 0,
            "evaluation_overlap": 0,
            "exact_archive_prefix": True,
            "historical_warm_metadata_matches": True,
            "normalized_partition_metadata_sha256": normalized,
            "strict_source_timestamps": True,
            "keyframe_attribute_frame_counts": counts,
            "capture_chronology_established": False,
            "hashes": hashes,
        },
    )
    print({"intake_audit": "passed", "clips": len(frames["clips"]), "batches": total // 32}, flush=True)


def independent_metrics(curve, points):
    deficits = np.array([curve[: i + 1].max(0) - row for i, row in enumerate(curve)])
    widths = np.array([b - a if a >= 132 else 0 for a, b in zip(points[:-1], points[1:], strict=True)])
    drop = deficits.max(0)
    return np.r_[
        curve[-1],
        drop,
        (deficits[:-1] * widths[:, None]).sum(0) / widths.sum(),
        ((deficits[:-1] > 1) * widths[:, None]).sum(0),
        max(drop[1:]),
    ]


def scientific_audit(config):
    out = ROOT / config.output
    assert not (out / "independent-verification.json").exists()
    lock, complete = read(out / "input_lock.json"), read(out / "complete.json")
    check(lock["hashes"])
    check(complete["hashes"])
    assert read(ROOT / config.intake / "independent-verification.json")["passed"]
    check(read(ROOT / config.intake / "independent-verification.json")["hashes"])
    collection, summary = read(out / "collection.json"), read(out / "summary.json")
    records, points, fork, horizon = (
        collection["records"],
        collection["points"],
        collection["first_alarm"],
        lock["horizon"],
    )
    assert points == sorted(set(lock["base_points"]) | ({fork} if fork is not None else set()))
    with np.load(ROOT / config.evaluation / "data.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    scores, curves, reader_curves, physical = {}, {}, {}, 0
    initial = torch.load(out / "initial_state.pt", map_location="cpu", weights_only=True)
    assert not initial["optimizer"]["state"], "The online optimizer must start empty"
    initial_marker = fingerprint(initial)
    frame_intake = read(ROOT / config.frames / "intake.json")
    input_arrays = [np.load(c["frames_path"], mmap_mode="r", allow_pickle=False) for c in frame_intake["clips"]]
    edges = np.r_[0, np.cumsum([len(a) for a in input_arrays])]
    input_digests = {}
    for step in range(1, horizon + 1):
        start, end = (step - 1) * 32, step * 32
        fragments = [
            a[max(start - lo, 0) : min(end - lo, len(a))]
            for a, lo, hi in zip(input_arrays, edges[:-1], edges[1:], strict=True)
            if start < hi and end > lo
        ]
        batch = np.concatenate(fragments)
        assert batch.shape == (32, 64, 64, 3)
        input_digests[step] = hashlib.sha256(batch.tobytes()).hexdigest()
    for arm in config.arms:
        row = records[arm]
        assert row["controller_deployed"] == (arm == "adaptive")
        check(read(out / arm / "complete.json")["hashes"])
        assert row["initial_fingerprint"] == initial_marker
        with np.load(Path(row["embedding_path"])) as banks:
            expected_monitor_steps = {str(step) for step in range(132, horizon + 1, 8)}
            assert set(row["monitors"]) == expected_monitor_steps
            assert set(banks.files) == expected_monitor_steps | {str(p) for p in points}
            events, last, streak, credit, dynamic, dynamic_rates = [], None, 0, 0, [], []
            for step in range(1, horizon + 1):
                numerator = 4 if last is None else min(4, 1 + (step - last - 1) // 32)
                credit += numerator
                take = credit >= 4
                if take:
                    credit -= 4
                dynamic.append(bool(take))
                dynamic_rates.append(numerator / 4)
                if step <= 132 or (step - 132) % 8:
                    continue
                values = geometry(banks["132"], banks[str(step)])
                unhealthy = bool(np.any(values >= 0.1 - 1e-12))
                streak = streak + 1 if unhealthy else 0
                fired = streak >= 2
                events.append(
                    {"step": step, "values": values.tolist(), "unhealthy": unhealthy, "streak": streak, "alarm": fired}
                )
                if fired:
                    last, streak, credit = step, 0, 0
            assert len(events) == len(row["events"])
            for actual, expected in zip(row["events"], events, strict=True):
                assert all(actual[k] == expected[k] for k in ("step", "unhealthy", "streak", "alarm"))
                np.testing.assert_allclose(actual["values"], expected["values"], atol=1e-10)
                assert row["monitors"][str(actual["step"])]["health_alarm"] == actual["alarm"]
            first = next((e["step"] for e in events if e["alarm"]), None)
            assert first == row["first_alarm"] == fork
            if arm == "adaptive":
                decisions, rates = dynamic, dynamic_rates
            else:
                count = (
                    ((horizon - fork) // 4 if arm == "quarter" else sum(records["adaptive"]["accepted"][fork:]))
                    if fork is not None
                    else 0
                )
                decisions, rates = [], []
                for step in range(1, horizon + 1):
                    before = arm == "full" or fork is None or step <= fork
                    j, remaining = (step - fork - 1, horizon - fork) if fork is not None else (0, 1)
                    decisions.append(bool(before or (j + 1) * count // remaining > j * count // remaining))
                    rates.append(1.0 if before else 0.25 if arm == "quarter" else count / remaining)
            assert decisions == row["accepted"]
            np.testing.assert_array_equal(row["rates"], rates)
            assert row["updates"] == sum(decisions)
            assert len(row["trace"]) == horizon and [r["accepted"] for r in row["trace"]] == decisions
            for response in row["trace"]:
                assert response["input_sha256"] == (input_digests[response["step"]] if response["accepted"] else None)
            if row["execution"] == "mask_identical_alias":
                original = records[row["trajectory_from"]]
                assert original["accepted"] == decisions
                assert original["initial_fingerprint"] == row["initial_fingerprint"]
                assert original["final_fingerprint"] == row["final_fingerprint"]
            else:
                physical += sum(decisions)
                assert row["skips_verified"] == horizon - sum(decisions)
            panel = []
            for point in points:
                assert np.isfinite(banks[str(point)]).all()
                observation, reader = row["observations"][str(point)], []
                if point == 0:
                    assert observation["learner_fingerprint"] == initial_marker
                if str(point) in row["monitors"]:
                    assert observation["learner_fingerprint"] == row["monitors"][str(point)]["learner_fingerprint"]
                for draw in config.draws:
                    path = Path(observation["readout"]) / f"panel_{draw}.npz"
                    meta = read(path.with_suffix(".json"))
                    assert meta["learner_fingerprint"] == observation["learner_fingerprint"]
                    assert meta["encoder_fingerprint"] == observation["encoder_fingerprint"] and meta["draw"] == draw
                    assert digest(path) == meta["sha256"]
                    if str(path) not in scores:
                        with np.load(path) as data:
                            probability = data["probabilities"]
                        assert probability.shape == (4063, 2) and np.isfinite(probability).all()
                        pred = probability[:3663].argmax(1)
                        domain = [
                            100 * np.mean([np.mean(pred[(domains == d) & (labels == c)] == c) for c in (0, 1)])
                            for d in (0, 1)
                        ]
                        scores[str(path)] = [np.mean(domain), *domain]
                    reader.append(scores[str(path)])
                np.testing.assert_allclose(reader, observation["panel_accuracy_pp"], atol=1e-10)
                panel.append(reader)
                if fork is not None and point <= fork:
                    assert (
                        observation["learner_fingerprint"]
                        == records["full"]["observations"][str(point)]["learner_fingerprint"]
                    )
            curves[arm] = np.mean(panel, axis=1)
            reader_curves[arm] = np.array(panel).transpose(1, 0, 2)
            np.testing.assert_allclose(summary["rules"][arm]["reader_curves"], np.array(panel).transpose(1, 0, 2))
        state = torch.load(Path(row["state_path"]), map_location="cpu", weights_only=True)
        assert (
            fingerprint(state) == row["final_fingerprint"] == row["observations"][str(horizon)]["learner_fingerprint"]
        )
    curves["frozen"] = np.repeat(curves["full"][:1], len(points), axis=0)
    curves["permanent"] = (
        np.array([curves["full"][points.index(min(p, fork))] for p in points]) if fork is not None else curves["full"]
    )
    reader_curves["frozen"] = np.repeat(reader_curves["full"][:, :1], len(points), axis=1)
    reader_curves["permanent"] = (
        reader_curves["full"][:, [points.index(min(p, fork)) for p in points]]
        if fork is not None
        else reader_curves["full"]
    )
    metric_values = {}
    for arm, curve in curves.items():
        np.testing.assert_allclose(summary["rules"][arm]["reader_curves"], reader_curves[arm])
        np.testing.assert_allclose(
            summary["rules"][arm]["reader_metrics"],
            [independent_metrics(reader, points) for reader in reader_curves[arm]],
            atol=1e-10,
        )
        updates = (
            records[arm]["updates"]
            if arm in records
            else (0 if arm == "frozen" else fork if fork is not None else horizon)
        )
        assert summary["rules"][arm]["updates"] == updates
        assert summary["rules"][arm]["accepted_images"] == 32 * updates
        values = independent_metrics(curve, points)
        metric_values[arm] = values
        np.testing.assert_allclose(summary["rules"][arm]["curve"], curve, atol=1e-10)
        np.testing.assert_allclose(summary["rules"][arm]["metrics"], values, atol=1e-10)
        np.testing.assert_allclose(summary["rules"][arm]["gain_over_warm"], curve[-1] - curves["full"][0], atol=1e-10)
    for arm in ("full", "permanent", "quarter", "paced"):
        np.testing.assert_allclose(
            summary["contrasts"][arm]["final_difference_pp"], curves["adaptive"][-1] - curves[arm][-1]
        )
        np.testing.assert_allclose(
            summary["contrasts"][arm]["drawdown_reduction_pp"], metric_values[arm][-1] - metric_values["adaptive"][-1]
        )
    assert records["adaptive"]["updates"] == records["paced"]["updates"] and summary["exact_work_matched"]
    assert not summary["controller_qualified"] and summary["uncertainty_intervals"] is None
    assert summary["ordered_streams"] == summary["ssl_seeds"] == 1
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collects, replays = [r for r in attempts if not r["replay"]], [r for r in attempts if r["replay"]]
    fits = len(list(out.glob("readouts/*/panel_*.npz")))
    for group in (collects, replays):
        assert sum(r["ssl_started"] for r in group) == sum(r["ssl_completed"] for r in group) == physical
    assert sum(r["fits_started"] for r in collects) == sum(r["fits_completed"] for r in collects) == fits
    assert fits <= lock["max_classifier_fits"] <= 432 and physical <= lock["max_ssl_updates"]
    assert collection["physical_work"]["ssl_completed"] == physical
    assert collection["physical_work"]["fits_completed"] == fits
    assert sum(r["replay_fits"] for r in replays) == 2
    assert sum(r["replay_fits_started"] for r in replays) == 2
    assert read(out / "training-replay-verification.json")["passed"]
    check(lock["hashes"])
    check(complete["hashes"])
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "source_sha256": digest(Path(__file__)),
            "physical_ssl_updates": physical,
            "new_classifier_fits": fits,
            "exact_classifier_replays": 2,
            "raw_probability_files": len(scores),
            "logical_branches": 4,
            "physical_branches": sum(r["execution"] == "physical" for r in records.values()),
            "unbounded_monitor_and_admission_verified": True,
            "exact_work_verified": True,
            "actual_accepted_input_order_verified": True,
            "no_independent_source_confidence_claim": True,
        },
    )
    print({"scientific_audit": "passed", "ssl_updates": physical, "new_classifier_fits": fits}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("intake", "science"), required=True)
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG)
    if args.stage == "intake":
        intake_audit(config)
    else:
        scientific_audit(config)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
