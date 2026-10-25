"""Decode real ordered BDD frames, retaining source PTS and the existing PIL resize."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FRAME_COUNT = 128
FRAME_STRIDE = 4
SIZE = 64
WIDTH, HEIGHT = 1280, 720


def sha256(path: Path) -> str:
    """Hash an artifact without loading it all into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def decode(ffmpeg: Path, video: Path, directory: Path) -> dict[str, Any]:  # noqa: C901 - decoding integrity checks
    """Decode fixed source-frame positions and reject missing or invalid timestamps."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "decoder.log"
    frames = []
    # Logging before selection retains actual source frame indices and PTS.
    command = [str(ffmpeg), "-nostdin", "-hide_banner", "-threads", "1", "-i", str(video),
               "-map", "0:v:0", "-vf",
               f"showinfo,select=not(mod(n\\,{FRAME_STRIDE}))*(isnan(prev_selected_t)+gt(t\\,prev_selected_t))",
               "-frames:v", str(FRAME_COUNT), "-fps_mode", "passthrough", "-an",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with log.open("wb") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)  # noqa: S603 - explicit executable
        try:
            if process.stdout is None:
                raise RuntimeError("decoder stdout missing")
            for _ in range(FRAME_COUNT):
                raw = process.stdout.read(WIDTH * HEIGHT * 3)
                if len(raw) != WIDTH * HEIGHT * 3:
                    raise ValueError("clip shorter than the declared 128-frame intake horizon")
                image = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
                frames.append(np.array(image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))
            # Read to EOF so an unexpected extra frame cannot deadlock the child.
            if process.stdout.read(1):
                raise ValueError("decoder produced excess frames")
            if process.wait(timeout=60):
                raise ValueError("FFmpeg decoding failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    text = log.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)")
    selected: dict[int, dict[str, Any]] = {}
    rejected_pts = []
    for match in pattern.finditer(text):
        number, pts, seconds, width, height = match.groups()
        number = int(number)
        if number % FRAME_STRIDE == 0 and len(selected) < FRAME_COUNT:
            if (int(width), int(height)) != (WIDTH, HEIGHT):
                raise ValueError("unexpected source dimensions")
            if selected and float(seconds) <= next(reversed(selected.values()))["pts_seconds"]:
                rejected_pts.append({"source_frame_index": number, "pts": int(pts), "pts_seconds": float(seconds)})
                continue
            selected[number] = {"source_frame_index": number, "pts": int(pts), "pts_seconds": float(seconds)}
    positions = sorted(selected)
    if len(positions) != FRAME_COUNT:
        raise ValueError("could not verify every selected source frame's PTS")
    metadata = [selected[n] for n in positions]
    timestamps = [m["pts_seconds"] for m in metadata]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("source timestamps are not strictly increasing")
    array = np.stack(frames)
    np.save(directory / "frames.npy", array, allow_pickle=False)
    for record, frame in zip(metadata, array, strict=True):
        record["rgb64_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
    result = {"source_video": str(video.resolve()), "video_sha256": sha256(video),
              "frames_path": str((directory / "frames.npy").resolve()),
              "frames_sha256": sha256(directory / "frames.npy"),
              "decoder_log_sha256": sha256(log), "shape": list(array.shape), "dtype": str(array.dtype),
              "frames": metadata, "rejected_candidate_timestamps": rejected_pts}
    (directory / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Prepare a cache and chronological trace from a complete download manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-clips", type=int, help="Infrastructure smoke only; not a research collection")
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not source["complete"] and args.limit_clips is None:
        raise ValueError("download is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in source["clips"][:args.limit_clips]:
        directory = args.output / clip["clip_id"]
        result = decode(args.ffmpeg.resolve(), Path(clip["video_path"]), directory)
        if result["video_sha256"] != clip["sha256"]:
            raise ValueError("video changed since acquisition")
        records.append({**clip, **result})
        intake = {"stage": "decode", "complete": len(records) == len(source["clips"]) and source["complete"],
                  "infrastructure_smoke": args.limit_clips is not None,
                  "download_manifest_sha256": sha256(args.manifest), "ffmpeg_sha256": sha256(args.ffmpeg),
                  "frame_stride": FRAME_STRIDE, "frames_per_clip": FRAME_COUNT, "image_size": SIZE,
                  "sampling_rule": "first 128 candidates with source index divisible by 4 and strictly advancing PTS",
                  "resize": "PIL RGB bilinear; uint8 cache; training float32 / 255",
                  "source_group_level": "clip; ride/session unknown", "clips": records}
        (args.output / "intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
        print(f"decoded {len(records)}/{len(source['clips'])} clips", flush=True)


if __name__ == "__main__":
    main()
