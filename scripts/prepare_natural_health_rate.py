"""Preserve pinned video-member order, then decode an outcome-free stream prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from scripts.prepare_bdd_temporal_archive import SplitHTTPFile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/natural_health_rate.yaml"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def warm_ids(metadata):
    valid = []
    for row in metadata:
        attrs = row["attributes"]
        if not isinstance(attrs, dict):
            continue
        fields = tuple(attrs.get(k) for k in ("timeofday", "weather", "scene"))
        if all(isinstance(v, str) and v not in ("", "undefined") for v in fields):
            valid.append((row["clip_id"], fields))
    counts = Counter(v[1][2] for v in valid)
    cells = defaultdict(list)
    for clip, fields in valid:
        if counts[fields[2]] >= 50:
            cells[fields].append(clip)
    result = set()
    for clips in cells.values():
        ordered = sorted(clips, key=lambda c: hashlib.sha256(f"20260921:{c}.jpg".encode()).digest())
        count = min(len(ordered) - 1, max(1, round(len(ordered) * 0.2)))
        result.update(ordered[:count])
    return result


def select(config):
    out = ROOT / config.intake
    assert not out.exists(), "Preserve existing preparation"
    out.mkdir(parents=True)
    root = Path(config.bdd_root)
    metadata, source_hash = [], hashlib.sha256()
    for i, path in enumerate(sorted((root / "labels/train").glob("*.json")), 1):
        raw = path.read_bytes()
        source_hash.update(path.name.encode() + b"\0" + hashlib.sha256(raw).digest())
        row = json.loads(raw)
        clip = Path(row["name"]).stem
        assert (root / "images/100k/train" / f"{clip}.jpg").is_file()
        metadata.append({"clip_id": clip, "attributes": row.get("attributes", {})})
        if i % 10000 == 0:
            print({"metadata_records": i}, flush=True)
    assert len(metadata) == 70000 and len({r["clip_id"] for r in metadata}) == 70000
    warm = warm_ids(metadata)
    prior = read(ROOT / config.prior / "input_lock.json")
    monitor = {Path(r["path"]).stem for r in prior["monitor_reference"]["images"]}
    evaluation = read(ROOT / config.evaluation / "data-role-manifest.json")
    heldout = {Path(r["path"]).stem for r in evaluation["records"]}
    excluded = warm | monitor | heldout
    annotations = {r["clip_id"]: r["attributes"] for r in metadata}
    index = read(ROOT / config.archive_index)
    selected, exclusions = [], Counter()
    for position, entry in enumerate(index["entries"]):
        if "/train/" not in entry["path"]:
            exclusions["non_train"] += 1
            continue
        clip = Path(entry["path"]).stem
        if clip in excluded:
            exclusions["warm_or_monitor_or_evaluation"] += 1
            continue
        assert clip in annotations, "Train member missing complete local metadata"
        selected.append({**entry, "clip_id": clip, "archive_position": position, "attributes": annotations[clip]})
        if len(selected) == config.max_clips:
            break
    assert len(selected) == config.max_clips
    assert len({r["clip_id"] for r in selected}) == len(selected)
    assert max(r["bytes"] for r in selected) <= 128 * 2**20
    assert sum(r["compressed_bytes"] for r in selected) + 64 * 2**20 < config.network_budget_gib * 2**30
    write(out / "metadata.json", {"records": metadata, "source_raw_digest": source_hash.hexdigest()})
    write(
        out / "selection.json",
        {
            "clips": selected,
            "warm_clip_ids": sorted(warm),
            "monitor_clip_ids": sorted(monitor),
            "evaluation_clip_ids": sorted(heldout),
            "excluded_before_prefix": dict(exclusions),
            "ordering": "Pinned video archive member order; no attribute grouping or balancing",
            "archive_index_sha256": digest(ROOT / config.archive_index),
            "source_metadata_digest": source_hash.hexdigest(),
            "before_any_learner_outcomes": True,
        },
    )
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/prepare_bdd_temporal_archive.py",
        ROOT / "docs/project-plan/natural-health-rate-frozen.md",
        ROOT / config.archive_index,
        out / "metadata.json",
        out / "selection.json",
    ]
    write(out / "preparation-lock.json", {"hashes": {str(p): digest(p) for p in paths}, "created_unix": time.time()})
    print({"selected": len(selected), "warm_excluded": len(warm), "first_clip": selected[0]["clip_id"]}, flush=True)


def timestamps(text, limit):
    pattern = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)")
    selected = []
    for match in pattern.finditer(text):
        number, pts, seconds, width, height = match.groups()
        second = float(seconds)
        if len(selected) == limit:
            break
        if second < len(selected) / 16 or (selected and second <= selected[-1]["pts_seconds"]):
            continue
        assert (int(width), int(height)) == (1280, 720), "Unexpected source dimensions"
        selected.append({"source_frame_index": int(number), "pts": int(pts), "pts_seconds": second})
    return selected


def decode(ffmpeg, raw_path, directory, limit):
    assert not directory.exists(), "Preserve incomplete decoder attempts"
    directory.mkdir(parents=True)
    log = directory / "decoder.log"
    count_limit = min(limit, 4096)
    command = [
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-threads",
        "1",
        "-i",
        str(raw_path),
        "-map",
        "0:v:0",
        "-vf",
        "showinfo,select=gte(t\\,selected_n/16)*(isnan(prev_selected_t)+gt(t\\,prev_selected_t))",
        "-frames:v",
        str(count_limit),
        "-fps_mode",
        "passthrough",
        "-an",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    frames = []
    with log.open("wb") as errors:
        child = subprocess.Popen(  # noqa: S603 - verified decoder and explicit argument vector
            command, stdout=subprocess.PIPE, stderr=errors, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        try:
            while len(frames) < count_limit:
                raw = child.stdout.read(1280 * 720 * 3)
                if not raw:
                    break
                assert len(raw) == 1280 * 720 * 3, "Incomplete native frame"
                image = Image.frombytes("RGB", (1280, 720), raw)
                frames.append(np.array(image.resize((64, 64), Image.Resampling.BILINEAR)))
            assert not child.stdout.read(1) and child.wait(timeout=60) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    assert frames, "Empty video; do not replace the clip"
    trace = timestamps(log.read_text(encoding="utf-8", errors="replace"), count_limit)
    assert len(trace) == len(frames)
    array = np.stack(frames)
    path = directory / "frames.npy"
    np.save(path, array, allow_pickle=False)
    for event, frame in zip(trace, array, strict=True):
        event["rgb64_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
    result = {
        "frames_path": str(path),
        "frames_sha256": digest(path),
        "shape": list(array.shape),
        "frames": trace,
        "decoder_log_sha256": digest(log),
        "terminal_prefix": len(frames) == limit,
    }
    write(directory / "manifest.json", result)
    return result


def acquire(config):
    intake, frames_root = ROOT / config.intake, ROOT / config.frames
    for name, expected in read(intake / "preparation-lock.json")["hashes"].items():
        assert digest(name) == expected, name
    frames_root.mkdir(parents=True, exist_ok=True)
    temporary = intake / "temporary"
    temporary.mkdir(exist_ok=True)
    index, selection = read(ROOT / config.archive_index), read(intake / "selection.json")
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    progress_path = intake / "progress.json"
    previous = read(progress_path).get("download_bytes", 0) if progress_path.exists() else 0
    remote = SplitHTTPFile(index["parts"], index["revision"], int(config.network_budget_gib * 2**30 - previous))
    records, total = [], 0

    def progress():
        write(
            progress_path,
            {
                "stage": "acquire",
                "clips": len(records),
                "frames": total,
                "target_frames": config.max_batches * 32,
                "download_bytes": previous + remote.received,
                "free_bytes": shutil.disk_usage(frames_root).free,
            },
        )

    try:
        with zipfile.ZipFile(remote) as archive:
            for selected in selection["clips"]:
                if total >= config.max_batches * 32:
                    break
                clip = selected["clip_id"]
                directory = frames_root / clip
                saved = directory / "source.json"
                if saved.exists():
                    record = read(saved)
                    assert digest(record["frames_path"]) == record["frames_sha256"]
                    records.append(record)
                    total += record["shape"][0]
                    progress()
                    continue
                assert shutil.disk_usage(frames_root).free > 768 * 2**20
                member = archive.getinfo(selected["path"])
                assert (member.file_size, member.compress_size, member.CRC, member.header_offset) == (
                    selected["bytes"],
                    selected["compressed_bytes"],
                    selected["crc32"],
                    selected["offset"],
                )
                raw_path = temporary / Path(member.filename).name
                if not raw_path.exists():
                    for attempt in range(3):
                        try:
                            payload = archive.read(member)
                            break
                        except Exception as error:
                            progress()
                            write(
                                intake / f"network-attempt-{time.time_ns()}.json",
                                {
                                    "clip_id": clip,
                                    "attempt": attempt,
                                    "error": repr(error),
                                    "download_bytes": previous + remote.received,
                                },
                            )
                            if attempt == 2:
                                raise
                            time.sleep(2)
                    assert len(payload) == member.file_size and zlib.crc32(payload) == member.CRC
                    raw_path.write_bytes(payload)
                    del payload
                assert raw_path.stat().st_size == member.file_size
                assert zlib.crc32(raw_path.read_bytes()) == member.CRC
                raw_hash = digest(raw_path)
                progress()
                decoded = decode(ffmpeg, raw_path, directory, config.max_batches * 32 - total)
                record = {**selected, **decoded, "video_sha256": raw_hash, "temporary_video_removed": True}
                write(saved, record)
                assert raw_path.resolve().parent == temporary.resolve() and raw_path.suffix.lower() in (".mov", ".mp4")
                raw_path.unlink()
                records.append(record)
                total += record["shape"][0]
                progress()
                print({"clips": len(records), "frames": total, "target": config.max_batches * 32}, flush=True)
    except Exception as error:
        progress()
        write(
            intake / f"failure-{time.time_ns()}.json",
            {"error": repr(error), "download_bytes": previous + remote.received},
        )
        raise
    used = total // 32 * 32
    assert used // 32 >= config.minimum_batches, "Cohort too short; do not change it using learner outcomes"
    write(
        frames_root / "intake.json",
        {
            "complete": True,
            "clips": records,
            "frames_decoded": total,
            "frames_used": used,
            "terminal_frames_not_batched": total - used,
            "batches": used // 32,
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": digest(ffmpeg),
            "download_bytes": previous + remote.received,
            "selection_sha256": digest(intake / "selection.json"),
            "natural_capture_chronology_established": False,
        },
    )
    print({"intake_complete": True, "batches": used // 32, "clips": len(records)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "acquire", "all"), default="all")
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG)
    if args.stage in ("select", "all"):
        select(config)
    if args.stage in ("acquire", "all"):
        acquire(config)


if __name__ == "__main__":
    main()
