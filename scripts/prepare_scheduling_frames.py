"""Freeze sources, decode real frames, and replace only infrastructure failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

from cafl4ds.artifacts import atomic_json
from cafl4ds.scheduling import allocation
from scripts.decode_history_video import decode
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "outputs/scheduling/20261013-intake"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    amendments = sorted(INTAKE.glob("amendment-*/reason.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    replacements = len(amendments)
    cumulative = json.loads(amendments[-1].read_text())["cumulative_download_bytes"] if amendments else 0
    while True:
        source = INTAKE / "download_manifest.json"
        acquired = json.loads(source.read_text(encoding="utf-8"))
        if not acquired["complete"] or len(acquired["clips"]) != 240:
            raise ValueError("Incomplete240-clip intake")
        selection_path = INTAKE / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        design = allocation(acquired["clips"])
        atomic_json(out / "allocation.json", {"streams": design, "selection_sha256": digest(selection_path)})
        records = []
        failed = None
        for clip in acquired["clips"]:
            directory = out / clip["clip_id"]
            video = Path(clip["video_path"])
            if digest(video) != clip["sha256"]:
                raise ValueError("Video integrity changed")
            if (directory / "manifest.json").exists():
                result = json.loads((directory / "manifest.json").read_text())
                if digest(Path(result["frames_path"])) != result["frames_sha256"]:
                    raise ValueError("Pixel cache changed")
            elif (directory / "predecode-failure.json").exists():
                failed = (clip, json.loads((directory / "predecode-failure.json").read_text())["error"])
                break
            else:
                try:
                    result = decode(ffmpeg, video, directory)
                except ValueError as error:
                    failed = (clip, str(error))
                    break
            records.append({**clip, **result})
            print(json.dumps({"decoded": len(records), "of": 240, "clip": clip["clip_id"]}), flush=True)
        if failed is None:
            cumulative += acquired["download_bytes"]
            atomic_json(
                out / "intake.json",
                {
                    "complete": True,
                    "clips": records,
                    "streams": design,
                    "frames_per_clip": 512,
                    "frames_per_period": 3072,
                    "ffmpeg_path": str(ffmpeg),
                    "ffmpeg_sha256": digest(ffmpeg),
                    "cumulative_download_bytes": cumulative,
                    "replacements": replacements,
                },
            )
            return
        replacements += 1
        if replacements > 24:
            raise ValueError("Replacement cap exceeded")
        clip, reason = failed
        cumulative += acquired["download_bytes"]
        amendment = INTAKE / f"amendment-{replacements}"
        amendment.mkdir(exist_ok=False)
        for path in (selection_path, source, out / "allocation.json"):
            (amendment / path.name).write_bytes(path.read_bytes())
        pool = json.loads((INTAKE / "selection.pool.json").read_text(encoding="utf-8"))
        key = f"{clip['keyframe_attributes']['timeofday']}/{clip['keyframe_attributes']['scene']}"
        used = {c["clip_id"] for c in selection["clips"]} | {p.stem for p in (INTAKE / "videos/train").glob("*.mov")}
        replacement = next(c for c in pool[key] if c["clip_id"] not in used)
        selection["clips"] = [replacement if c["clip_id"] == clip["clip_id"] else c for c in selection["clips"]]
        atomic_json(
            amendment / "reason.json",
            {
                "failed_clip": clip["clip_id"],
                "replacement_clip": replacement["clip_id"],
                "reason": reason,
                "cumulative_download_bytes": cumulative,
                "before_any_ssl_outcomes": True,
            },
        )
        atomic_json(selection_path, selection)
        remaining_mib = int((6144 * 1024**2 - cumulative) // 1024**2)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.fetch_cross_clips",
                "--archive-index",
                "outputs/temporal-bdd/20261011-intake/archive_index.json",
                "--selection",
                str(selection_path),
                "--output",
                str(INTAKE),
                "--max-download-mib",
                str(remaining_mib),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
