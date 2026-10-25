"""Timestamp-based 8 Hz sampling over 32 seconds; infrastructure only until cleared."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from cafl4ds.artifacts import atomic_json

COUNT = 256
RATE = 8


def timestamps(text: str, count: int = COUNT) -> list[dict]:
    """Independently reconstruct the filter's timestamp-slot and monotonicity rule."""
    pattern = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)")
    selected = []
    for match in pattern.finditer(text):
        number, pts, seconds, width, height = match.groups()
        second = float(seconds)
        if len(selected) == count:
            break
        if second < len(selected) / RATE or (selected and second <= selected[-1]["pts_seconds"]):
            continue
        if (int(width), int(height)) != (1280, 720):
            raise ValueError("Unexpected native video dimensions")
        if second - len(selected) / RATE > 0.1:
            raise ValueError("Timestamp gap exceeds the frozen 100 ms slot tolerance")
        selected.append({"source_frame_index": int(number), "pts": int(pts), "pts_seconds": second})
    if len(selected) != count:
        raise ValueError("Cannot establish the full 32-second timestamp horizon")
    return selected


def decode(ffmpeg: Path, video: Path, output: Path) -> dict:
    """Keep source timestamps and use the established PIL RGB64 resize."""
    output.mkdir(parents=True, exist_ok=False)
    log = output / "decoder.log"
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
        "showinfo,select=gte(t\\,selected_n/8)*(isnan(prev_selected_t)+gt(t\\,prev_selected_t))",
        "-frames:v",
        str(COUNT),
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
        child = subprocess.Popen(  # noqa: S603 - explicit verified decoder and argument vector
            command,
            stdout=subprocess.PIPE,
            stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        try:
            if child.stdout is None:
                raise RuntimeError("No decoder pipe")
            for _ in range(COUNT):
                raw = child.stdout.read(1280 * 720 * 3)
                if len(raw) != 1280 * 720 * 3:
                    raise ValueError("Insufficient decoded frames; preserve decoder log")
                rgb = Image.frombytes("RGB", (1280, 720), raw)
                frames.append(np.array(rgb.resize((64, 64), Image.Resampling.BILINEAR)))
            if child.stdout.read(1) or child.wait(timeout=60):
                raise ValueError("Unexpected decoder output or exit status")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    trace = timestamps(log.read_text(encoding="utf-8", errors="replace"))
    array = np.stack(frames)
    np.save(output / "frames.npy", array, allow_pickle=False)
    for row, frame in zip(trace, array, strict=True):
        row["rgb64_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
    result = {
        "video_path": str(video.resolve()),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "frames_path": str((output / "frames.npy").resolve()),
        "frames_sha256": hashlib.sha256((output / "frames.npy").read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "frames": trace,
        "decoder_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    atomic_json(output / "manifest.json", result)
    return result


def main() -> None:
    """Decode a fully acquired smoke manifest without evaluating learners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    if not manifest["complete"]:
        raise ValueError("Download is not complete")
    records = []
    for row in manifest["clips"]:
        result = decode(args.ffmpeg.resolve(), Path(row["video_path"]), args.output / row["clip_id"])
        if result["video_sha256"] != row["sha256"]:
            raise ValueError("Video changed after intake")
        records.append({**row, **result})
        atomic_json(
            args.output / "intake.json",
            {
                "complete": len(records) == len(manifest["clips"]),
                "infrastructure_smoke": True,
                "rate_hz": RATE,
                "frames_per_clip": COUNT,
                "image_size": 64,
                "sampling_rule": (
                    "First real frame at or after k/8 seconds; strictly increasing PTS; slot delay <=100 ms"
                ),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "ffmpeg_sha256": hashlib.sha256(args.ffmpeg.read_bytes()).hexdigest(),
                "download_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "clips": records,
            },
        )
        print(
            f"Decoded {len(records)}/{len(manifest['clips'])}: {COUNT} frames "
            f"spanning {result['frames'][-1]['pts_seconds']:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
