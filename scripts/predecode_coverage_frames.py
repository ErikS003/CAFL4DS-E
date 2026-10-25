"""Decode completed downloads concurrently without evaluating or replacing sources."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

from cafl4ds.artifacts import atomic_json
from scripts.decode_history_video import decode
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "outputs/coverage-admission/20261013-intake/download_manifest.json"
OUT = ROOT / "outputs/coverage-admission/20261013-frames"


def decode_one(clip, ffmpeg):
    directory = OUT / clip["clip_id"]
    if (directory / "manifest.json").exists() or (directory / "predecode-failure.json").exists():
        return
    if digest(Path(clip["video_path"])) != clip["sha256"]:
        raise ValueError("Downloaded video integrity changed")
    try:
        decode(ffmpeg, Path(clip["video_path"]), directory)
    except ValueError as error:
        atomic_json(
            directory / "predecode-failure.json",
            {"clip_id": clip["clip_id"], "error": str(error), "before_any_ssl_outcomes": True},
        )
        print(json.dumps({"infrastructure_failure": clip["clip_id"], "error": str(error)}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    done = set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if not INTAKE.exists():
                time.sleep(5)
                continue
            snapshot = json.loads(INTAKE.read_text(encoding="utf-8"))
            pending = [c for c in snapshot["clips"] if c["clip_id"] not in done]
            for identifier in pool.map(lambda c: (decode_one(c, ffmpeg), c["clip_id"])[1], pending):
                done.add(identifier)
                print(json.dumps({"predecoded_or_failed": len(done), "clip": identifier}), flush=True)
            if snapshot["complete"]:
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
