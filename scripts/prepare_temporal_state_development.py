"""Acquire and screen full temporal development inputs, then launch the scored run."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cafl4ds.artifacts import atomic_json
from scripts.clear_temporal_candidate_data import arrays, fingerprint
from scripts.decode_temporal_qualification import decode

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/temporal-benchmark"


def sha(path: Path) -> str:
    """Hash a local immutable input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_record(frame: np.ndarray) -> dict:
    """Use the frozen dHash/RGB8 screen on decoded RGB64 frames."""
    image = Image.fromarray(frame)
    grey = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    small = np.asarray(image.resize((8, 8), Image.Resampling.BILINEAR))
    return {
        "dhash": int.from_bytes(np.packbits((grey[:, 1:] > grey[:, :-1]).flatten()).tobytes(), "little"),
        "rgb8": base64.b64encode(small.tobytes()).decode(),
        "rgb64_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    }


def main() -> None:  # noqa: C901 - preserve every intake failure and fixed-order replacement
    """Preparation is outcome-free; local annotation provenance remains explicitly provisional."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    status = {"pid": os.getpid(), "stage": "waiting_for_intake", "scientific_run_started": False}
    atomic_json(out / "preparation-status.json", status)
    guard = None
    try:
        if os.name == "nt":
            guard = ctypes.windll.kernel32
            guard.SetThreadExecutionState(0x80000001)
        intake_path = out / "intake/download_manifest.json"
        deadline = time.monotonic() + 6 * 3600
        while not intake_path.exists() or not json.loads(intake_path.read_text())["complete"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Preserving partial intake after six-hour timeout")
            if intake_path.exists():
                partial = json.loads(intake_path.read_text())
                status.update(downloaded_clips=len(partial["clips"]), transfer_bytes=partial["download_bytes"])
                atomic_json(out / "preparation-status.json", status)
            time.sleep(5)
        allocation_path = BENCH / "20261011-role-repair-v1/allocation.json"
        allocation = json.loads(allocation_path.read_text())
        rows = allocation["allocations"]
        audit = json.loads(Path(allocation["audit_path"]).read_text())
        stream_list = json.loads((BENCH / "20261011-pending-stream-intake/all-streams.json").read_text())["clips"]
        downloaded = {r["clip_id"]: r for r in json.loads(intake_path.read_text())["clips"]}
        existing = {r["clip_id"]: r["local_qualified_smoke"] for r in stream_list if r["local_qualified_smoke"]}
        ffmpeg = ROOT / "outputs/temporal-bdd/tools/ffmpeg-win-x86_64-v7.1.exe"
        if sha(ffmpeg) != "2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3":
            raise ValueError("Decoder differs from qualified executable")
        historical = [
            json.loads(s)
            for s in (BENCH / "20261011-clearance-v1/historical_fingerprints.jsonl").read_text().splitlines()
        ]
        # Images/metadata are screened only; no predictions or accuracy outcomes are read.
        key_records = [fingerprint(Path(r["image_path"])) for r in rows]
        reader_records = [
            r for r, row in zip(key_records, rows, strict=True) if not row["role"].startswith("stream_cohort_")
        ]
        pool = historical + reader_records
        pool_hashes, pool_rgb = arrays(pool)
        pool_pixels = {r["rgb64_sha256"] for r in pool}
        old_intake = json.loads((ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json").read_text())
        old_hashes = {f["rgb64_sha256"] for c in old_intake["clips"] for f in c["frames"]}
        old_samples = [
            frame_record(f) for c in old_intake["clips"] for f in np.load(c["frames_path"], allow_pickle=False)[::32]
        ]
        old_h, old_rgb = arrays(old_samples)
        pool_hashes = np.concatenate((pool_hashes, old_h))
        pool_rgb = np.concatenate((pool_rgb, old_rgb))
        counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        used = {r["clip_id"] for r in rows}
        families = {r["clip_id"].split("-")[0]: r["role"] for r in rows}
        accepted_hashes = set()
        accepted_samples = []
        qualified = []
        failures = []
        replacements = []
        transfer_bytes = json.loads(intake_path.read_text())["download_bytes"]
        index_path = ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json"
        index = json.loads(index_path.read_text())
        members = {r["path"]: r for r in index["entries"]}

        def near_match(records: list[dict]) -> bool:
            """Screen sampled frames against historical/readers and accepted footage."""
            h, rgb = arrays(records)
            ph, pr = pool_hashes, pool_rgb
            if accepted_samples:
                ah, ar = arrays(accepted_samples)
                ph, pr = np.concatenate((ph, ah)), np.concatenate((pr, ar))
            for number in range(len(h)):
                distance = counts[np.bitwise_xor(ph, h[number]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
                selected = np.where(distance <= 4)[0]
                if len(selected) and np.any(np.sqrt(((pr[selected] - rgb[number]) ** 2).mean(axis=1)) <= 0.03):
                    return True
            return False

        streams = sorted(
            (r for r in rows if r["role"].startswith("stream_cohort_")), key=lambda r: (r["role"], r["clip_id"])
        )
        for slot, original in enumerate(streams):
            candidate = original
            for attempt in range(9):
                cid = candidate["clip_id"]
                status.update(stage="decoding_and_screening", qualified_clips=len(qualified), slot=slot, clip_id=cid)
                atomic_json(out / "preparation-status.json", status)
                try:
                    if cid in existing:
                        mp = Path(existing[cid]["decoded_manifest_path"])
                        if sha(mp) != existing[cid]["decoded_manifest_sha256"]:
                            raise RuntimeError("Existing decoded manifest changed")
                        decoded = json.loads(mp.read_text())
                    else:
                        directory = out / "decoded" / cid
                        mp = directory / "manifest.json"
                        if mp.exists():
                            decoded = json.loads(mp.read_text())
                        else:
                            decoded = decode(ffmpeg, Path(downloaded[cid]["video_path"]), directory)
                    if sha(Path(decoded["frames_path"])) != decoded["frames_sha256"]:
                        raise RuntimeError("Frame cache changed")
                    frames = np.load(decoded["frames_path"], allow_pickle=False)
                    frame_hashes = {hashlib.sha256(f.tobytes()).hexdigest() for f in frames}
                    samples = [frame_record(frames[i]) for i in (0, 64, 128, 192, 255)]
                    if frame_hashes & (old_hashes | pool_pixels | accepted_hashes) or near_match(samples):
                        raise ValueError("Video duplicate screen flagged this clip")
                    qualified.append({**candidate, **decoded, "decoded_manifest_path": str(mp.resolve())})
                    accepted_hashes.update(frame_hashes)
                    accepted_samples.extend(samples)
                    if cid != original["clip_id"]:
                        replacements.append(
                            {"old_clip_id": original["clip_id"], "new_clip_id": cid, "role": original["role"]}
                        )
                    break
                except ValueError as error:
                    failures.append({"clip_id": cid, "slot": slot, "attempt": attempt, "reason": str(error)})
                    atomic_json(out / "intake-failures.json", {"failures": failures, "replacements": replacements})
                    if attempt == 8:
                        raise RuntimeError("Nine outcome-free attempts exhausted for a stream slot") from error
                candidates = audit["pools"]["test"]["cells"][original["cell"]]["candidate_ids"]
                selected = None
                bdd = Path(audit["source_config"]["bdd_root"])
                for reserve in sorted(
                    candidates, key=lambda cid: hashlib.sha256(f"duration-reserve:{cid}".encode()).hexdigest()
                ):
                    family = reserve.split("-")[0]
                    if reserve in used or (family in families and families[family] != original["role"]):
                        continue
                    used.add(reserve)
                    image = bdd / "images/100k/test" / f"{reserve}.jpg"
                    key = fingerprint(image)
                    if key["rgb64_sha256"] in pool_pixels or near_match([key]):
                        continue
                    label = bdd / "labels/test" / f"{reserve}.json"
                    attrs = json.loads(label.read_text())["attributes"]
                    if attrs != original["keyframe_attributes"]:
                        raise RuntimeError("Reserve has wrong metadata cell")
                    member = members[f"bdd100k/videos/100k/test/{reserve}.mov"]
                    estimate = member["compressed_bytes"] + index["index_download_bytes"] + 4096
                    if transfer_bytes + estimate > 4096 * 1024**2:
                        raise RuntimeError("Preparation's 4 GiB total transfer budget exhausted")
                    directory = out / "reserves" / reserve
                    directory.mkdir(parents=True)
                    atomic_json(
                        directory / "selection.json",
                        {
                            "purpose": "Outcome-free duration/video-screen replacement",
                            "clips": [{"clip_id": reserve, "split": "test"}],
                        },
                    )
                    subprocess.run(  # noqa: S603 - fixed Python module and argument vector
                        [
                            sys.executable,
                            "-m",
                            "scripts.fetch_pinned_temporal_clips",
                            "--archive-index",
                            str(index_path),
                            "--selection",
                            str(directory / "selection.json"),
                            "--output",
                            str(directory),
                            "--max-download-mib",
                            "128",
                        ],
                        check=True,
                        cwd=ROOT,
                    )  # noqa: S603 - fixed module and explicit arguments
                    fetched = json.loads((directory / "download_manifest.json").read_text())
                    transfer_bytes += fetched["download_bytes"]
                    downloaded[reserve] = fetched["clips"][0]
                    families[family] = original["role"]
                    selected = {
                        **original,
                        "clip_id": reserve,
                        "image_path": str(image),
                        "image_sha256": sha(image),
                        "label_path": str(label),
                        "label_sha256": sha(label),
                        "replaces_clip_id": original["clip_id"],
                    }
                    break
                if selected is None:
                    raise RuntimeError("No unused same-cell reserve passed screening")
                candidate = selected
            atomic_json(
                out / "qualified-clips.json", {"clips": qualified, "failures": failures, "replacements": replacements}
            )
        replacement_by_old = {r["replaces_clip_id"]: r for r in qualified if "replaces_clip_id" in r}
        final_rows = [replacement_by_old.get(r["clip_id"], r) for r in rows]
        atomic_json(
            out / "prepared.json",
            {
                "mode": "provisional_local_annotation_development",
                "label_provenance_verified": False,
                "capture_sessions_known": False,
                "sealed_final_predictions_allowed": False,
                "allocation": final_rows,
                "clips": qualified,
                "failures": failures,
                "replacements": replacements,
                "parent_allocation_path": str(allocation_path),
                "parent_allocation_sha256": sha(allocation_path),
                "transfer_bytes": transfer_bytes,
                "maximum_transfer_bytes": 4096 * 1024**2,
                "video_screen": (
                    "All cached frames exact; five timestamp samples near vs "
                    "train/val/readers/old footage/accepted clips"
                ),
                "scientific_independence_established": False,
            },
        )
        status.update(stage="launching_scored_development", qualified_clips=len(qualified), scientific_run_started=True)
        atomic_json(out / "preparation-status.json", status)
        subprocess.run(  # noqa: S603 - fixed local experiment module
            [
                sys.executable,
                "-m",
                "scripts.run_temporal_state_development",
                "--prepared",
                str(out / "prepared.json"),
                "--protocol",
                str(ROOT / "docs/project-plan/slow-edge-temporal-state-development-frozen.md"),
                "--output",
                str(out / "experiment"),
            ],
            check=True,
            cwd=ROOT,
        )  # noqa: S603 - fixed local experiment module
        status.update(stage="complete")
        atomic_json(out / "preparation-status.json", status)
    except Exception as error:
        status.update(stage="failed", error=str(error))
        atomic_json(out / "preparation-status.json", status)
        raise
    finally:
        if guard is not None:
            guard.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
