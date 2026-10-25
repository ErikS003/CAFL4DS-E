"""Decode the prespecified continued-night control, excluding all parent-history clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg

from cafl4ds.artifacts import atomic_json
from scripts.decode_regime_video import decode
from scripts.run_short_buffer_experiment import digest, remap

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json"
    parent_path = ROOT / "outputs/regime-windows/20261013-frames/intake.json"
    source = json.loads(source_path.read_text())
    prior = json.loads(parent_path.read_text())
    prior_ids = {c["clip_id"] for c in prior["clips"]}
    groups = []
    for scene in ("city street", "highway"):
        eligible = [
            c
            for c in source["clips"]
            if c["keyframe_attributes"]["timeofday"] == "night" and c["keyframe_attributes"]["scene"] == scene
        ]
        ranked = sorted(
            eligible, key=lambda c: hashlib.sha256(("regime-windows:20261013:" + c["clip_id"]).encode()).hexdigest()
        )
        groups.append(ranked[4:8])
    if any(len(g) != 4 for g in groups):
        raise ValueError("Insufficient continued-night control sources")
    clips = [c for pair in zip(*groups, strict=True) for c in pair]
    if {c["clip_id"] for c in clips} & prior_ids:
        raise ValueError("Continued-night clips overlap previous visits")
    allocation = {
        "clip_ids": [c["clip_id"] for c in clips],
        "original_intake_sha256": digest(source_path),
        "previous_regime_intake_sha256": digest(parent_path),
        "selection": "next four per shared night class under original outcome-independent hash order",
    }
    if (out / "allocation.json").exists() and json.loads((out / "allocation.json").read_text()) != allocation:
        raise ValueError("Allocation changed")
    atomic_json(out / "allocation.json", allocation)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    records = []
    for clip in clips:
        video = remap(clip["video_path"])
        if digest(video) != clip["video_sha256"]:
            raise ValueError("Source video changed")
        directory = out / clip["clip_id"]
        if (directory / "manifest.json").exists():
            row = json.loads((directory / "manifest.json").read_text())
            if digest(Path(row["frames_path"])) != row["frames_sha256"]:
                raise ValueError("Frame cache changed")
        else:
            row = decode(ffmpeg, video, directory)
        records.append({**clip, **row, "visit": "night2"})
        atomic_json(
            out / "intake.json",
            {
                "complete": len(records) == 8,
                "clips": records,
                "frames_per_clip": 528,
                "rate_hz": 16,
                "arrivals": 4224,
                "ffmpeg_sha256": digest(ffmpeg),
                "ffmpeg_path": str(ffmpeg),
                "decoder_sha256": digest(ROOT / "scripts/decode_regime_video.py"),
                "preparation_source_sha256": digest(Path(__file__)),
            },
        )
        print(
            json.dumps(
                {
                    "decoded_clips": len(records),
                    "total_clips": 8,
                    "clip_id": clip["clip_id"],
                    "last_pts_seconds": row["frames"][-1]["pts_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
