"""Prepare disjoint day/night/day videos without inspecting learner outcomes."""

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
    source = json.loads(source_path.read_text())
    groups = {}
    for day in ("daytime", "night"):
        for scene in ("city street", "highway"):
            eligible = [
                c
                for c in source["clips"]
                if c["keyframe_attributes"]["timeofday"] == day and c["keyframe_attributes"]["scene"] == scene
            ]
            groups[day, scene] = sorted(
                eligible, key=lambda c: hashlib.sha256(("regime-windows:20261013:" + c["clip_id"]).encode()).hexdigest()
            )
    visits = []
    for name, day, offset in (("day1", "daytime", 0), ("night", "night", 0), ("day2", "daytime", 4)):
        scenes = [groups[day, s][offset : offset + 4] for s in ("city street", "highway")]
        if any(len(g) != 4 for g in scenes):
            raise ValueError("Insufficient shared-class clips")
        visits.extend([{**clip, "visit": name} for pair in zip(*scenes, strict=True) for clip in pair])
    if len({c["clip_id"] for c in visits}) != 24:
        raise ValueError("Visits share a clip")
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    allocation = {
        "clips": [
            {"clip_id": c["clip_id"], "visit": c["visit"], "attributes": c["keyframe_attributes"]} for c in visits
        ],
        "source_sha256": digest(source_path),
    }
    if (out / "allocation.json").exists() and json.loads((out / "allocation.json").read_text()) != allocation:
        raise ValueError("Allocation changed")
    atomic_json(out / "allocation.json", allocation)
    records = []
    for clip in visits:
        video = remap(clip["video_path"])
        if digest(video) != clip["video_sha256"]:
            raise ValueError("Source video differs")
        directory = out / clip["clip_id"]
        if directory.exists() and not (directory / "manifest.json").exists():
            successful = sorted(directory.glob("attempt_*/manifest.json"))
            if successful:
                directory = successful[-1].parent
            else:
                attempt = 2
                while (directory / f"attempt_{attempt}").exists():
                    attempt += 1
                directory = directory / f"attempt_{attempt}"
        if (directory / "manifest.json").exists():
            row = json.loads((directory / "manifest.json").read_text())
            if digest(Path(row["frames_path"])) != row["frames_sha256"]:
                raise ValueError("Decoded cache changed")
        else:
            row = decode(ffmpeg, video, directory)
        records.append({**clip, **row})
        atomic_json(
            out / "intake.json",
            {
                "complete": len(records) == 24,
                "clips": records,
                "frames_per_clip": 528,
                "rate_hz": 16,
                "image_size": 64,
                "total_arrivals": 12672,
                "visit_boundaries": [4224, 8448, 12672],
                "sampling_rule": "first real frame at/after k/16 seconds, increasing PTS, real gaps retained and reported",
                "source_group_level": "clip; sessions unknown",
                "ffmpeg_path": str(ffmpeg),
                "ffmpeg_sha256": digest(ffmpeg),
                "decoder_source_sha256": digest(ROOT / "scripts/decode_regime_video.py"),
                "preparation_source_sha256": digest(Path(__file__)),
                "original_intake_sha256": digest(source_path),
            },
        )
        print(
            json.dumps(
                {
                    "decoded_clips": len(records),
                    "total_clips": 24,
                    "clip_id": clip["clip_id"],
                    "last_pts_seconds": row["frames"][-1]["pts_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
