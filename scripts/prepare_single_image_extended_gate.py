"""Freeze 1,200 unseen video IDs and acquire bounded decoded temporal clips."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import time
import zipfile
import zlib
from pathlib import Path

import imageio_ffmpeg
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.single_image_execution_amendment import validate_hashes
from scripts.prepare_bdd_temporal_archive import SplitHTTPFile
from scripts.prepare_single_image_transition import decode96
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_extended_gate.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-extended-gate-frozen.md"
POOL = ROOT / "outputs/scheduling/20261013-intake/selection.pool.json"
INDEX = ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json"


def scientific_sources():
    return [
        CONFIG,
        PROTOCOL,
        Path(__file__),
        ROOT / "scripts/run_single_image_extended_gate.py",
        ROOT / "scripts/report_single_image_extended_gate.py",
        ROOT / "cafl4ds/single_image_extended_gate.py",
        ROOT / "cafl4ds/single_image_long_gate.py",
        ROOT / "scripts/run_single_image_long_gate.py",
        ROOT / "scripts/run_single_image_filter_factorial.py",
        ROOT / "cafl4ds/single_image_filter_factorial.py",
        ROOT / "cafl4ds/single_image_filter.py",
        ROOT / "cafl4ds/single_image_forecast.py",
        ROOT / "cafl4ds/single_image_signal.py",
        ROOT / "cafl4ds/short_buffer.py",
        ROOT / "scripts/run_single_image_nonlinear.py",
        ROOT / "scripts/prepare_bdd_temporal_archive.py",
        ROOT / "scripts/prepare_single_image_transition.py",
        ROOT / "cafl4ds/models/vit.py",
        ROOT / "cafl4ds/ssl/mae.py",
        ROOT / "cafl4ds/ssl/augment.py",
    ]


def plan(root):
    """Freeze membership, sources and reserve ordering before any new-stream SSL."""
    if (root / "selection-lock.json").exists():
        validate_plan(root)
        return
    settings = OmegaConf.to_container(OmegaConf.load(CONFIG))
    original = read(ROOT / "outputs/scheduling/20261013-intake/selection.json")
    assert digest(POOL) == original["eligible_pool_sha256"]
    source = ROOT / settings["source"]
    inherited = read(source / "intake/intake.json")
    inventory = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "outputs"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    known = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", inventory))
    for path in sorted((ROOT / ".cache").glob("*known-videos.txt")):
        known.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", path.read_text()))
    for previous in [
        source,
        ROOT / "outputs/single-image-filtered-paths/20261022-mixed-calibration",
        ROOT / "outputs/single-image-state-control/20261022-frozen-ablation",
    ]:
        selected = read(previous / "intake/selection.json")
        known.update(selected["excluded_clip_ids"])
        known.update(q["clip_id"] for q in selected["query"])
        known.update(c["clip_id"] for rows in selected["candidates"].values() for c in rows)
    index = read(INDEX)
    available = {Path(row["path"]).stem for row in index["entries"] if "/train/" in row["path"]}
    pool = read(POOL)
    count = settings["offered"] // 96
    assert count == 200 and settings["offered"] % 96 == 0
    candidates, reserves = {}, {}
    for light in ("daytime", "night"):
        eligible = [
            row for row in pool[light + "/highway"] if row["clip_id"] not in known and row["clip_id"] in available
        ]
        eligible.sort(
            key=lambda r: hashlib.sha256(f"{settings['selection_namespace']}:{r['clip_id']}".encode()).digest()
        )
        required = len(inherited["histories"]) * count // 2
        assert len(eligible) >= required + settings["reserves_per_light"], (light, len(eligible))
        candidates[light] = eligible[:required]
        reserves[light] = eligible[required : required + settings["reserves_per_light"]]
    offsets, slots = dict(daytime=0, night=0), []
    for history, meta in inherited["histories"].items():
        for position in range(count):
            symbol = meta["pattern"][(settings["start"] // 96 + position) % len(meta["pattern"])]
            light = "daytime" if symbol == "D" else "night"
            chosen = candidates[light][offsets[light]]
            offsets[light] += 1
            metadata = Path(chosen["keyframe_path"]).parents[3] / "labels/train" / f"{chosen['clip_id']}.json"
            assert read(metadata)["attributes"] == chosen["keyframe_attributes"]
            slots.append(dict(history=history, position=position, light=light, selected=chosen))
    ids = [r["selected"]["clip_id"] for r in slots]
    assert len(ids) == len(set(ids)) == 1200 and not set(ids) & known
    root.mkdir(parents=True, exist_ok=True)
    (root / "intake").mkdir(exist_ok=True)
    atomic_json(
        root / "selection-lock.json",
        dict(
            settings=settings,
            slots=slots,
            reserves=reserves,
            excluded_clip_ids=sorted(known),
            original_histories=inherited["histories"],
            warm_partition_seed=original["warm_partition_seed"],
            warm_fraction=original["warm_fraction"],
            source_hashes={str(p.resolve()): digest(p) for p in scientific_sources()},
            data_hashes={str(p.resolve()): digest(p) for p in [POOL, INDEX, source / "intake/intake.json"]},
            stream_outcomes_inspected=False,
        ),
    )
    for path in scientific_sources():
        target = root / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_json(
        root / "status.json",
        dict(
            stage="membership_frozen",
            planned_new_clips=1200,
            offered_per_model=settings["offered"],
            new_frames=1200 * 96,
        ),
    )


def validate_plan(root):
    lock = read(root / "selection-lock.json")
    validate_hashes(root, {**lock["source_hashes"], **lock["data_hashes"]})
    return lock


def clear_incomplete_decoder(directory, parent):
    assert directory.resolve().is_relative_to(parent.resolve())
    for child in directory.iterdir():
        assert child.name in {"decoder.log", "frames.npy", "manifest.json"} and child.is_file(), child
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()


def acquire(root, limit=None):
    """Fetch bounded ranges, retain compact pixels, and release each source MOV."""
    lock = validate_plan(root)
    settings, out = lock["settings"], root / "intake"
    if (out / "intake.json").exists():
        assert read(out / "intake.json")["complete"]
        return True
    for name in ("records", "frames", "download"):
        (out / name).mkdir(exist_ok=True)
    failures = read(out / "failures.json") if (out / "failures.json").exists() else dict(records=[])
    bad = {r["clip_id"] for r in failures["records"]}
    records = {}
    for path in sorted((out / "records").glob("*.json")):
        record = read(path)
        assert digest(Path(record["frames_path"])) == record["frames_sha256"]
        records[record["slot"]] = record
    used = {r["clip_id"] for r in records.values()}
    previous = read(out / "transfer.json")["received"] if (out / "transfer.json").exists() else 0
    index = read(INDEX)
    remote = SplitHTTPFile(index["parts"], index["revision"], int(settings["max_download_gib"] * 2**30) - previous)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    started = time.perf_counter()

    def progress():
        atomic_json(out / "transfer.json", dict(received=previous + remote.received))
        status = dict(
            stage="acquiring_new_clips",
            pid=__import__("os").getpid(),
            completed_clips=len(records),
            total_clips=len(lock["slots"]),
            decoded_new_frames=len(records) * 96,
            received_gib=(previous + remote.received) / 2**30,
            elapsed_seconds=time.perf_counter() - started,
            free_gib=shutil.disk_usage(ROOT).free / 2**30,
        )
        atomic_json(root / "status.json", status)
        print(status, flush=True)

    progress()
    with zipfile.ZipFile(remote) as archive:
        pinned = {r["path"]: r for r in index["entries"]}
        for slot, meta in enumerate(lock["slots"]):
            if slot in records:
                continue
            if limit is not None and len(records) >= limit:
                return False
            if shutil.disk_usage(ROOT).free < settings["minimum_free_gib"] * 2**30:
                raise RuntimeError("Storage guard stopped acquisition before exhausting free space")
            row = meta["selected"]
            while True:
                if row["clip_id"] in bad:
                    row = next(r for r in lock["reserves"][meta["light"]] if r["clip_id"] not in bad | used)
                identifier = row["clip_id"]
                metadata = Path(row["keyframe_path"]).parents[3] / "labels/train" / f"{identifier}.json"
                assert read(metadata)["attributes"] == row["keyframe_attributes"]
                name = f"bdd100k/videos/100k/train/{identifier}.mov"
                entry, expected = archive.getinfo(name), pinned[name]
                assert (entry.file_size, entry.compress_size, entry.CRC, entry.header_offset) == (
                    expected["bytes"],
                    expected["compressed_bytes"],
                    expected["crc32"],
                    expected["offset"],
                )
                assert entry.file_size <= 128 * 1024**2 and entry.compress_size <= 128 * 1024**2
                video = out / "download" / f"{identifier}.mov"
                data = video.read_bytes() if video.exists() else b""
                if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                    for attempt in range(settings["download_attempts"]):
                        try:
                            data = archive.read(entry)
                            break
                        except Exception:
                            progress()
                            if attempt + 1 == settings["download_attempts"]:
                                raise
                            time.sleep(min(2**attempt, 8))
                    video.write_bytes(data)
                assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC
                video_hash = hashlib.sha256(data).hexdigest()
                del data
                directory = out / "frames" / identifier
                if directory.exists() and not (directory / "manifest.json").exists():
                    clear_incomplete_decoder(directory, out / "frames")
                try:
                    decoded = (
                        read(directory / "manifest.json")
                        if (directory / "manifest.json").exists()
                        else decode96(video, directory, ffmpeg, 96)
                    )
                    assert digest(Path(decoded["frames_path"])) == decoded["frames_sha256"]
                except ValueError as error:
                    failures["records"].append(dict(clip_id=identifier, reason=repr(error), before_ssl=True))
                    atomic_json(out / "failures.json", failures)
                    bad.add(identifier)
                    video.unlink(missing_ok=True)
                    continue
                record = dict(
                    **row,
                    slot=slot,
                    history=meta["history"],
                    position=meta["position"],
                    frames_path=decoded["frames_path"],
                    frames_sha256=decoded["frames_sha256"],
                    decoder_manifest=str((directory / "manifest.json").resolve()),
                    video_sha256=video_hash,
                    video_bytes=entry.file_size,
                    archive_entry=name,
                    crc32=entry.CRC,
                    video_cache_retained=False,
                )
                record["pool_role"] = record["role"]
                record["role"] = "held_new_test_stream"
                atomic_json(out / "records" / f"slot_{slot:04d}.json", record)
                records[slot] = record
                used.add(identifier)
                video.unlink()
                progress()
                break
    assert len(records) == 1200
    assert len(used) == 1200 and not used & set(lock["excluded_clip_ids"])
    histories = {}
    for history, old in lock["original_histories"].items():
        continuation = sorted([r for r in records.values() if r["history"] == history], key=lambda r: r["position"])
        assert len(continuation) == 200
        assert sum(r["keyframe_attributes"]["timeofday"] == "daytime" for r in continuation) == 100
        histories[history] = dict(
            pattern=old["pattern"],
            clip_ids=old["clip_ids"][:6] + [r["clip_id"] for r in continuation],
            paths=old["paths"][:6] + [r["frames_path"] for r in continuation],
        )
    atomic_json(
        out / "intake.json",
        dict(
            complete=True,
            clips=list(records.values()),
            histories=histories,
            received=previous + remote.received,
            failures=failures["records"],
            source_authenticity="Pinned community mirror ZIP CRC and SHA; session independence not established",
            new_unique_clip_ids=len(used),
            reused_prefix_clip_count=36,
            ffmpeg_sha256=digest(ffmpeg),
        ),
    )
    progress()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "acquire"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "plan":
        plan(args.output.resolve())
    else:
        acquire(args.output.resolve(), args.limit)
