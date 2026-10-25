"""Outcome-free fresh clip allocation and bounded sequential download/decode."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
import zipfile
import zlib
from pathlib import Path

import imageio_ffmpeg
from omegaconf import OmegaConf

from scripts.decode_history_video import decode
from scripts.prepare_bdd_temporal_archive import REPOSITORY, SplitHTTPFile
from scripts.run_preaction_stopping import ROOT, digest, read, write

CONFIG = ROOT / "cafl4ds/configs/dnd_health_transfer.yaml"


def exclusions(config):
    paths = sorted(set(ROOT.glob("outputs/**/intake.json")) | set(ROOT.glob("outputs/**/selection.json")))
    paths = [p for p in paths if "dnd-health-transfer" not in str(p)]
    excluded, hashes = set(), {}
    for path in paths:
        data = read(path)
        if not isinstance(data, dict):
            continue
        excluded.update(c["clip_id"] for c in data.get("clips", []) if isinstance(c, dict) and "clip_id" in c)
        excluded.update(data.get("excluded_clip_ids", []))
        hashes[str(path)] = digest(path)
    videos = sorted(ROOT.glob("outputs/**/*.mov"))
    excluded.update(p.stem for p in videos)
    monitor = read(ROOT / config.monitor / "input_lock.json")["reference"]["images"]
    excluded.update(Path(r["path"]).stem for r in monitor)
    evaluation = read(ROOT / config.evaluation / "data-role-manifest.json")
    excluded.update(Path(r["path"]).stem for r in evaluation["records"])
    return excluded, hashes, sorted({p.stem for p in videos})


def select(config):
    intake = ROOT / config.intake
    assert not intake.exists(), "Selection already exists; use acquire to resume"
    previous = read(ROOT / "outputs/scheduling/20261013-intake/selection.json")
    assert digest(ROOT / config.pool) == previous["eligible_pool_sha256"]
    excluded, prior_hashes, videos = exclusions(config)
    pool = read(ROOT / config.pool)
    ranked = {
        cell: sorted(
            [c for c in clips if c["clip_id"] not in excluded],
            key=lambda c: hashlib.sha256(("dnd-health-transfer:20261016:" + c["clip_id"]).encode()).digest(),
        )
        for cell, clips in pool.items()
    }
    used = {k: 0 for k in ranked}
    clips, histories = [], []
    for i, (day, night) in enumerate(zip(config.day_updates, config.night_updates, strict=True)):
        periods = []
        for visit, (light, updates) in enumerate((("daytime", day), ("night", night), ("daytime", 448 - day - night))):
            assert updates % 32 == 0
            ids = []
            for _ in range(updates // 32):
                for scene in ("city street", "highway"):
                    cell = f"{light}/{scene}"
                    record = {
                        **ranked[cell][used[cell]],
                        "cell": cell,
                        "history": i,
                        "visit": visit,
                        "slot": len(clips),
                    }
                    used[cell] += 1
                    clips.append(record)
                    ids.append(record["clip_id"])
            periods.append(ids)
        histories.append({"name": f"transfer_{i:02d}", "day_updates": day, "night_updates": night, "periods": periods})
    assert len(clips) == len({c["clip_id"] for c in clips}) == 224
    assert not excluded.intersection(c["clip_id"] for c in clips)
    index = read(ROOT / config.archive_index)
    entries = {Path(e["path"]).stem: e for e in index["entries"] if "/train/" in e["path"]}
    required = sum(entries[c["clip_id"]]["compressed_bytes"] for c in clips)
    assert required + index["index_download_bytes"] < config.network_budget_gib * 2**30
    intake.mkdir(parents=True)
    record = {
        "purpose": "fresh D/N/D frozen health-stop transfer",
        "clips": clips,
        "histories": histories,
        "excluded_clip_ids": sorted(excluded),
        "prior_manifests": prior_hashes,
        "local_video_ids": videos,
        "pool_sha256": digest(ROOT / config.pool),
        "archive_index_sha256": digest(ROOT / config.archive_index),
        "required_compressed_bytes": required,
        "selection_before_any_new_learner": True,
    }
    write(intake / "selection-initial.json", record)
    write(intake / "selection.json", record)
    write(intake / "reserves.json", {k: v[used[k] : used[k] + 16] for k, v in ranked.items()})
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/decode_history_video.py",
        ROOT / "scripts/prepare_bdd_temporal_archive.py",
        ROOT / "scripts/run_preaction_stopping.py",
        ROOT / "docs/project-plan/dnd-health-transfer-frozen.md",
        ROOT / config.pool,
        ROOT / config.archive_index,
        intake / "selection-initial.json",
        intake / "reserves.json",
    ]
    write(
        intake / "preparation-lock.json", {"hashes": {str(p): digest(p) for p in paths}, "prepared_unix": time.time()}
    )
    print({"selected": 224, "histories": 8, "excluded": len(excluded), "required_mib": required / 2**20}, flush=True)


def acquire(config):
    intake, frames = ROOT / config.intake, ROOT / config.frames
    from scripts.run_preaction_stopping import check

    check(read(intake / "preparation-lock.json")["hashes"])
    selection = read(intake / "selection.json")
    index = read(ROOT / config.archive_index)
    assert index["repository"] == REPOSITORY
    entries = {Path(e["path"]).stem: e for e in index["entries"] if "/train/" in e["path"]}
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    frames.mkdir(parents=True, exist_ok=True)
    temporary = intake / "temporary"
    temporary.mkdir(exist_ok=True)
    progress_path = intake / "progress.json"
    old_progress = read(progress_path) if progress_path.exists() else {}
    previous_bytes = old_progress.get("download_bytes", 0)
    records, failures = [], read(intake / "failures.json") if (intake / "failures.json").exists() else []
    remote = SplitHTTPFile(index["parts"], index["revision"], int(config.network_budget_gib * 2**30 - previous_bytes))

    def progress():
        write(
            progress_path,
            {
                "decoded": len(records),
                "total": 224,
                "download_bytes": previous_bytes + remote.received,
                "failures": len(failures),
                "free_bytes": shutil.disk_usage(frames).free,
            },
        )

    with zipfile.ZipFile(remote) as archive:
        for slot in range(224):
            while True:
                clip = selection["clips"][slot]
                directory = frames / clip["clip_id"]
                manifest = directory / "manifest.json"
                if manifest.exists():
                    decoded = read(manifest)
                    assert digest(decoded["frames_path"]) == decoded["frames_sha256"]
                    records.append({**clip, **decoded})
                    break
                assert shutil.disk_usage(frames).free > 768 * 2**20, "Intake disk reserve threatened"
                locked = entries[clip["clip_id"]]
                entry = archive.getinfo(locked["path"])
                assert (entry.file_size, entry.compress_size, entry.CRC, entry.header_offset) == (
                    locked["bytes"],
                    locked["compressed_bytes"],
                    locked["crc32"],
                    locked["offset"],
                )
                raw = temporary / f"{clip['clip_id']}.mov"
                if not raw.exists():
                    for attempt in range(3):
                        try:
                            payload = archive.read(entry)
                            assert len(payload) == entry.file_size and zlib.crc32(payload) == entry.CRC
                            raw.write_bytes(payload)
                            del payload
                            break
                        except Exception:
                            progress()
                            if attempt == 2:
                                raise
                            time.sleep(2)
                assert raw.stat().st_size == entry.file_size
                crc = zlib.crc32(raw.read_bytes())
                assert crc == entry.CRC
                video_hash = digest(raw)
                progress()
                try:
                    decoded = decode(ffmpeg, raw, directory)
                except Exception as error:
                    failures.append(
                        {
                            "clip_id": clip["clip_id"],
                            "slot": slot,
                            "cell": clip["cell"],
                            "reason": str(error),
                            "raw_sha256": video_hash,
                        }
                    )
                    write(intake / "failures.json", failures)
                    assert len(failures) <= 8, "Decode replacement reserve exhausted"
                    reserves = read(intake / "reserves.json")[clip["cell"]]
                    occupied = {c["clip_id"] for c in selection["clips"]} | {c["clip_id"] for c in failures}
                    next_clip = next(c for c in reserves if c["clip_id"] not in occupied)
                    replacement = {**next_clip, **{k: clip[k] for k in ("cell", "history", "visit", "slot")}}
                    selection["clips"][slot] = replacement
                    periods = selection["histories"][clip["history"]]["periods"][clip["visit"]]
                    periods[periods.index(clip["clip_id"])] = replacement["clip_id"]
                    write(intake / "selection.json", selection)
                    continue
                assert decoded["video_sha256"] == video_hash
                cache = Path(decoded["frames_path"])
                before = digest(cache)
                compressed = subprocess.run(
                    ["compact.exe", "/C", "/F", "/EXE:LZX", str(cache)],
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                assert compressed.returncode == 0 and digest(cache) == before
                decoded.update(
                    {
                        "archive_path": entry.filename,
                        "archive_crc32": crc,
                        "archive_bytes": entry.file_size,
                        "temporary_raw_removed": True,
                        "lossless_compression_preserved_hash": True,
                    }
                )
                write(manifest, decoded)
                assert raw.resolve().parent == temporary.resolve() and raw.suffix == ".mov"
                raw.unlink()  # Only this new verified temporary file; cached scientific pixels remain.
                records.append({**clip, **decoded})
                break
            progress()
            print(
                {
                    "decoded": len(records),
                    "total": 224,
                    "transfer_mib": round((previous_bytes + remote.received) / 2**20, 1),
                },
                flush=True,
            )
    write(
        frames / "intake.json",
        {
            "complete": True,
            "clips": records,
            "histories": selection["histories"],
            "frames_per_clip": 512,
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": digest(ffmpeg),
            "selection_sha256": digest(intake / "selection.json"),
            "download_bytes": previous_bytes + remote.received,
            "source_authenticity": "Pinned community mirror integrity; BAIR authenticity not established",
        },
    )
    print({"intake_complete": True, "clips": len(records), "failures": len(failures)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "acquire"))
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG)
    select(config) if args.stage == "select" else acquire(config)


if __name__ == "__main__":
    main()
