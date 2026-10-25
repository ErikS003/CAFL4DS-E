"""Decode five fresh city/highway clip pairs per lighting condition."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg

from cafl4ds.artifacts import atomic_json
from scripts.decode_history_video import decode
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/history-order/20261013-intake/download_manifest.json"
    intake = json.loads(source.read_text())
    if not intake["complete"] or len(intake["clips"]) != 24:
        raise ValueError("Incomplete24-clip intake")
    groups = {
        scene: [c for c in intake["clips"] if c["keyframe_attributes"]["scene"] == scene]
        for scene in ("city street", "highway")
    }
    if any(len(g) != 12 for g in groups.values()):
        raise ValueError("Expected twelve clips per day scene")
    allocation = {
        "source_sha256": digest(source),
        "blocks": [
            {
                "context": "day",
                "block": k,
                "clip_ids": [
                    groups[s][k]["clip_id"]
                    for s in (("city street", "highway") if k % 2 == 0 else ("highway", "city street"))
                ],
            }
            for k in range(12)
        ],
    }
    if (out / "allocation.json").exists() and json.loads((out / "allocation.json").read_text()) != allocation:
        raise ValueError("Frozen order changed")
    atomic_json(out / "allocation.json", allocation)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    clips = {c["clip_id"]: c for c in intake["clips"]}
    blocks = []
    for item in allocation["blocks"]:
        records = []
        for identifier in item["clip_ids"]:
            clip = clips[identifier]
            video = Path(clip["video_path"])
            if digest(video) != clip["sha256"]:
                raise ValueError("Video hash changed")
            directory = out / identifier
            if directory.exists() and not (directory / "manifest.json").exists():
                attempt = 2
                while (directory / f"attempt_{attempt}").exists():
                    attempt += 1
                directory = directory / f"attempt_{attempt}"
            if (directory / "manifest.json").exists():
                decoded = json.loads((directory / "manifest.json").read_text())
                if digest(Path(decoded["frames_path"])) != decoded["frames_sha256"]:
                    raise ValueError("Decoded cache changed")
            else:
                decoded = decode(ffmpeg, video, directory)
            records.append({**clip, **decoded, "exposed_frames": 512})
            print(
                json.dumps({"context": item["context"], "block": item["block"], "decoded_clip": identifier}), flush=True
            )
        blocks.append({**item, "clips": records})
        atomic_json(
            out / "intake.json",
            {
                "complete": len(blocks) == 12,
                "blocks": blocks,
                "decoded_frames_per_clip": 512,
                "used_frames_per_clip": 512,
                "frames_per_block": 1024,
                "rate_hz": 16,
                "ffmpeg_path": str(ffmpeg),
                "ffmpeg_sha256": digest(ffmpeg),
                "decoder_sha256": digest(ROOT / "scripts/decode_history_video.py"),
                "preparation_sha256": digest(Path(__file__)),
                "source_group_level": "clip; sessions unknown",
            },
        )


if __name__ == "__main__":
    main()
