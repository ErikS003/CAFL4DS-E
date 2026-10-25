"""Acquire fresh warm-disjoint highway clips and decode96 timestamped frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import zipfile
import zlib
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import requests
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from scripts.decode_history_video import timestamps
from scripts.prepare_bdd_temporal_archive import SplitHTTPFile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_transition.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-transition-frozen.md"
POOL = ROOT / "outputs/scheduling/20261013-intake/selection.pool.json"
INDEX = ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode96(video, directory, ffmpeg, count):
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / "decoder.log"
    command = [
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-threads",
        "1",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        "showinfo,select=gte(t\\,selected_n/16)*(isnan(prev_selected_t)+gt(t\\,prev_selected_t))",
        "-frames:v",
        str(count),
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
        child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        try:
            for _ in range(count):
                raw = child.stdout.read(1280 * 720 * 3)
                if len(raw) != 1280 * 720 * 3:
                    raise ValueError("Insufficient96 native RGB frames")
                frames.append(
                    np.array(Image.frombytes("RGB", (1280, 720), raw).resize((64, 64), Image.Resampling.BILINEAR))
                )
            if child.stdout.read(1) or child.wait(timeout=60):
                raise ValueError("Unexpected decoder bytes/exit status")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    trace = timestamps(log.read_text(encoding="utf-8", errors="replace"), count=count)
    array = np.stack(frames)
    np.save(directory / "frames.npy", array, allow_pickle=False)
    for row, pixels in zip(trace, array, strict=True):
        row["rgb64_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
    result = {
        "frames_path": str((directory / "frames.npy").resolve()),
        "frames_sha256": sha(directory / "frames.npy"),
        "shape": list(array.shape),
        "frames": trace,
        "decoder_log_sha256": sha(log),
    }
    atomic_json(directory / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = OmegaConf.to_container(OmegaConf.load(CONFIG))
    if (out / "intake.json").exists() and read(out / "intake.json").get("complete"):
        raise ValueError("Intake already complete; preserve immutable evidence")
    original = read(ROOT / "outputs/scheduling/20261013-intake/selection.json")
    assert sha(POOL) == original["eligible_pool_sha256"]
    pool, index = read(POOL), read(INDEX)
    known = set()
    inventories = sorted((ROOT / ".cache").glob("*known-videos.txt"))
    for p in inventories:
        known.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", p.read_text(encoding="utf-8")))
    inventory = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "outputs"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    known.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", inventory))
    old_lock = ROOT / "outputs/single-image-signal/20261021-followup/input_lock.json"
    known.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", old_lock.read_text(encoding="utf-8")))
    candidates = {}
    for light in ("daytime", "night"):
        eligible = [c for c in pool[f"{light}/highway"] if c["clip_id"] not in known]
        eligible.sort(
            key=lambda c: hashlib.sha256(f"{settings['selection_namespace']}:{c['clip_id']}".encode()).digest()
        )
        candidates[light] = eligible[: 36 + settings["reserves_per_light"]]
        assert len(candidates[light]) == 42
        for c in candidates[light]:
            metadata = Path(c["keyframe_path"]).parents[3] / "labels/train" / f"{c['clip_id']}.json"
            assert read(metadata)["attributes"] == c["keyframe_attributes"]
    selection = {
        "purpose": "new-source transition-controlled singleton forecast",
        "excluded_clip_ids": sorted(known),
        "candidates": candidates,
        "pool_sha256": sha(POOL),
        "archive_index_sha256": sha(INDEX),
        "inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
        "settings": settings,
        "warm_partition_seed": original["warm_partition_seed"],
        "warm_fraction": original["warm_fraction"],
        "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(PROTOCOL),
        "config_sha256": sha(CONFIG),
        "known_inventory_hashes": {str(p): sha(p) for p in inventories},
        "outcomes_inspected": False,
    }
    if (out / "selection.json").exists():
        saved = read(out / "selection.json")
        # Path inventory now includes this intake, so keep its original exclusions/ranking.
        assert saved["source_sha256"] == sha(Path(__file__))
        assert saved["protocol_sha256"] == sha(PROTOCOL) and saved["config_sha256"] == sha(CONFIG)
        selection, candidates = saved, saved["candidates"]
    else:
        atomic_json(out / "selection.json", selection)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    remote = SplitHTTPFile(index["parts"], index["revision"], settings["max_download_mib"] * 1024**2)
    locked = {e["path"]: e for e in index["entries"]}
    previous = (
        read(out / "progress.json")
        if (out / "progress.json").exists()
        else {"clips": [], "failures": [], "attempted": [], "received": 0}
    )
    assert previous["received"] < remote.budget
    remote.budget -= previous["received"]
    records, failures, attempted = previous["clips"], previous["failures"], set(previous["attempted"])
    started = time.perf_counter()
    with zipfile.ZipFile(remote) as archive:
        for light in ("daytime", "night"):
            for c in candidates[light]:
                if sum(r["keyframe_attributes"]["timeofday"] == light for r in records) == 36:
                    break
                identifier = c["clip_id"]
                if identifier in attempted:
                    continue
                name = f"bdd100k/videos/100k/train/{identifier}.mov"
                entry, expected = archive.getinfo(name), locked[name]
                assert (entry.file_size, entry.compress_size, entry.CRC, entry.header_offset) == (
                    expected["bytes"],
                    expected["compressed_bytes"],
                    expected["crc32"],
                    expected["offset"],
                )
                assert entry.file_size <= 128 * 1024**2
                video = out / "videos" / f"{identifier}.mov"
                video.parent.mkdir(exist_ok=True)
                if video.exists():
                    data = video.read_bytes()
                else:
                    for attempt in range(3):
                        try:
                            data = archive.read(entry)
                            break
                        except (requests.RequestException, ValueError):
                            if attempt == 2:
                                raise
                    video.write_bytes(data)
                assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC
                row = {
                    **c,
                    "video_path": str(video),
                    "sha256": sha(video),
                    "bytes": len(data),
                    "crc32": entry.CRC,
                    "archive_path": name,
                }
                directory = out / "frames" / identifier
                try:
                    pixels = (
                        read(directory / "manifest.json")
                        if (directory / "manifest.json").exists()
                        else decode96(video, directory, ffmpeg, settings["frames_per_clip"])
                    )
                    assert sha(Path(pixels["frames_path"])) == pixels["frames_sha256"]
                    records.append({**row, **pixels})
                except ValueError as error:
                    failures.append({"clip_id": identifier, "reason": repr(error), "before_ssl": True})
                    atomic_json(directory / "failure.json", failures[-1])
                attempted.add(identifier)
                atomic_json(
                    out / "progress.json",
                    {
                        "clips": records,
                        "failures": failures,
                        "attempted": sorted(attempted),
                        "received": previous["received"] + remote.received,
                    },
                )
                print(
                    json.dumps(
                        {
                            "accepted": len(records),
                            "failed": len(failures),
                            "received_mib": (previous["received"] + remote.received) / 1024**2,
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    assert len(records) == 72
    by_light = {
        light: [r for r in records if r["keyframe_attributes"]["timeofday"] == light] for light in ("daytime", "night")
    }
    offsets, histories = {"daytime": 0, "night": 0}, {}
    for i, pattern in enumerate(settings["patterns"]):
        clips = []
        for symbol in pattern:
            light = "daytime" if symbol == "D" else "night"
            clips.append(by_light[light][offsets[light]])
            offsets[light] += 1
        histories[f"new_{i:02d}"] = {
            "pattern": pattern,
            "clip_ids": [c["clip_id"] for c in clips],
            "paths": [c["frames_path"] for c in clips],
        }
    assert not set(r["clip_id"] for r in records) & set(selection["excluded_clip_ids"])
    atomic_json(
        out / "intake.json",
        {
            "complete": True,
            "clips": records,
            "histories": histories,
            "failures": failures,
            "received": previous["received"] + remote.received,
            "selection_sha256": sha(out / "selection.json"),
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": sha(ffmpeg),
            "rate_hz": 16,
            "source_authenticity": "Pinned community mirror integrity; BAIR authenticity not independently established",
        },
    )


if __name__ == "__main__":
    main()
