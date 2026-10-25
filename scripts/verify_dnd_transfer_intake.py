"""Independent source allocation, chronological frame and exclusion audit before training."""

import hashlib
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.prepare_dnd_health_transfer import CONFIG
from scripts.run_preaction_stopping import ROOT, check, digest, read, write


def main():
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.intake
    frames = ROOT / config.frames
    assert not (out / "independent-verification.json").exists()
    prep = read(out / "preparation-lock.json")
    check(prep["hashes"])
    initial = read(out / "selection-initial.json")
    selection = read(out / "selection.json")
    intake = read(frames / "intake.json")
    assert intake["complete"] and len(intake["clips"]) == 224
    check(initial["prior_manifests"])
    excluded = set(initial["local_video_ids"])
    for name in initial["prior_manifests"]:
        previous = read(name)
        excluded.update(c["clip_id"] for c in previous.get("clips", []) if isinstance(c, dict) and "clip_id" in c)
        excluded.update(previous.get("excluded_clip_ids", []))
    excluded.update(
        Path(r["path"]).stem for r in read(ROOT / config.monitor / "input_lock.json")["reference"]["images"]
    )
    excluded.update(Path(r["path"]).stem for r in read(ROOT / config.evaluation / "data-role-manifest.json")["records"])
    assert excluded == set(initial["excluded_clip_ids"])
    pool = read(ROOT / config.pool)
    ranked = {}
    for cell, records in pool.items():
        ranked[cell] = sorted(
            [c for c in records if c["clip_id"] not in excluded],
            key=lambda c: hashlib.sha256(("dnd-health-transfer:20261016:" + c["clip_id"]).encode()).digest(),
        )
    offsets = {cell: 0 for cell in ranked}
    expected = []
    for i, (day, night) in enumerate(zip(config.day_updates, config.night_updates, strict=True)):
        for visit, (light, steps) in enumerate((("daytime", day), ("night", night), ("daytime", 448 - day - night))):
            for pair in range(steps // 32):
                for scene in ("city street", "highway"):
                    cell = f"{light}/{scene}"
                    c = ranked[cell][offsets[cell]]
                    offsets[cell] += 1
                    expected.append((c["clip_id"], i, visit, cell))
    assert [(c["clip_id"], c["history"], c["visit"], c["cell"]) for c in initial["clips"]] == expected
    failures = read(out / "failures.json") if (out / "failures.json").exists() else []
    assert len(failures) <= 8
    running_ids = [c["clip_id"] for c in initial["clips"]]
    failed_ids = set()
    reserves = read(out / "reserves.json")
    for failure in failures:
        slot = failure["slot"]
        assert running_ids[slot] == failure["clip_id"]
        failed_ids.add(failure["clip_id"])
        replacement = next(
            c["clip_id"] for c in reserves[failure["cell"]] if c["clip_id"] not in set(running_ids) | failed_ids
        )
        running_ids[slot] = replacement
    assert running_ids == [c["clip_id"] for c in selection["clips"]]
    for slot, (before, after) in enumerate(zip(initial["clips"], selection["clips"], strict=True)):
        assert after["slot"] == slot and (before["history"], before["visit"], before["cell"]) == (
            after["history"],
            after["visit"],
            after["cell"],
        )
        if before["clip_id"] != after["clip_id"]:
            assert any(f["slot"] == slot and f["clip_id"] == before["clip_id"] for f in failures)
            assert after["clip_id"] in {c["clip_id"] for c in read(out / "reserves.json")[after["cell"]]}
    selected = {c["clip_id"] for c in selection["clips"]}
    assert len(selected) == 224 and not selected & excluded
    inventory_path = out / "workspace-video-inventory.json"
    inventory = read(inventory_path)
    assert inventory["passed"] and not inventory["selected_overlap"]
    assert {Path(p).stem for p in inventory["previous_video_paths"]} <= set(initial["local_video_ids"])
    assert intake["selection_sha256"] == digest(out / "selection.json")
    index = read(ROOT / config.archive_index)
    archive = {Path(e["path"]).stem: e for e in index["entries"] if "/train/" in e["path"]}
    records = {c["clip_id"]: c for c in intake["clips"]}
    hashes = {**prep["hashes"], **initial["prior_manifests"], str(inventory_path): digest(inventory_path)}
    if failures:
        hashes[str(out / "failures.json")] = digest(out / "failures.json")
        for failure in failures:
            raw = out / "temporary" / f"{failure['clip_id']}.mov"
            assert digest(raw) == failure["raw_sha256"]
            hashes[str(raw)] = failure["raw_sha256"]
            log = frames / failure["clip_id"] / "decoder.log"
            hashes[str(log)] = digest(log)
    frame_count = 0
    for selected_clip in selection["clips"]:
        c = records[selected_clip["clip_id"]]
        assert c["history"] == selected_clip["history"] and c["visit"] == selected_clip["visit"]
        member = archive[c["clip_id"]]
        assert (
            c["archive_path"] == member["path"]
            and c["archive_crc32"] == member["crc32"]
            and c["archive_bytes"] == member["bytes"]
        )
        assert c["temporary_raw_removed"] and c["lossless_compression_preserved_hash"]
        assert not Path(c["video_path"]).exists()
        assert c["video_sha256"] == read(Path(c["frames_path"]).parent / "manifest.json")["video_sha256"]
        path = Path(c["frames_path"])
        assert digest(path) == c["frames_sha256"]
        hashes[str(path)] = c["frames_sha256"]
        log = path.parent / "decoder.log"
        assert digest(log) == c["decoder_log_sha256"]
        hashes[str(log)] = c["decoder_log_sha256"]
        matches = re.findall(
            r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)",
            log.read_text(encoding="utf-8", errors="replace"),
        )
        timestamps = []
        for number, pts, seconds, width, height in matches:
            if len(timestamps) == 512:
                break
            value = float(seconds)
            if value < len(timestamps) / 16 or (timestamps and value <= timestamps[-1][2]):
                continue
            assert (int(width), int(height)) == (1280, 720)
            timestamps.append((int(number), int(pts), value))
        assert len(timestamps) == 512
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (512, 64, 64, 3) and array.dtype == np.uint8
        for j, (frame, record, observed) in enumerate(zip(array, c["frames"], timestamps, strict=True)):
            assert (record["source_frame_index"], record["pts"], record["pts_seconds"]) == observed
            assert (
                record["pts_seconds"] >= j / 16
                and hashlib.sha256(frame.tobytes()).hexdigest() == record["rgb64_sha256"]
            )
            frame_count += 1
        hashes[str(path.parent / "manifest.json")] = digest(path.parent / "manifest.json")
    assert selection["histories"] == intake["histories"] and len(selection["histories"]) == 8
    for history in selection["histories"]:
        ids = [c for period in history["periods"] for c in period]
        assert len(ids) == len(set(ids)) == 28
        assert all(records[c]["history"] == int(history["name"][-2:]) for c in ids)
        for period in history["periods"]:
            cells = [records[c]["cell"] for c in period]
            assert sum(c.endswith("city street") for c in cells) == sum(c.endswith("highway") for c in cells)
        assert [len(p) * 16 for p in history["periods"]] == [
            history["day_updates"],
            history["night_updates"],
            448 - history["day_updates"] - history["night_updates"],
        ]
    assert read(out / "progress.json")["download_bytes"] <= 6 * 2**30
    for path in (
        out / "selection-initial.json",
        out / "selection.json",
        out / "preparation-lock.json",
        frames / "intake.json",
    ):
        hashes[str(path)] = digest(path)
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "new_clip_ids": 224,
            "clip_disjoint_histories": 8,
            "previous_exclusions": len(excluded),
            "chronological_rgb_frames_verified": frame_count,
            "replacements": len(failures),
            "warm_exclusion_basis": "unchanged audited eligible stream pool",
            "capture_sessions_known": False,
            "hashes": hashes,
            "source_sha256": digest(__file__),
        },
    )
    print({"intake_verification": "passed", "clips": 224, "frames": frame_count}, flush=True)


if __name__ == "__main__":
    main()
