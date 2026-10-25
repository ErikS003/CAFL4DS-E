"""Study39 bounded intake, copied from the pinned natural intake before use.

Adds immediate lossless compression after committing decoded source metadata.
The original acquisition implementation remains unchanged.
"""

import os
import shutil
import subprocess
import time
import zipfile
import zlib
from pathlib import Path
import imageio_ffmpeg
from scripts.prepare_bdd_temporal_archive import SplitHTTPFile
from scripts.prepare_natural_health_rate import decode
from scripts.prepare_early_state_adaptation import ROOT, read, write, digest


def compress(path):
    before = digest(path)
    if os.name == "nt":
        subprocess.run(
            [
                str(Path(os.environ["SystemRoot"]) / "System32/compact.exe"),
                "/C",
                "/EXE:LZX",
                "/I",
                "/Q",
                str(path.resolve()),
            ],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
        )
    assert digest(path) == before


def acquire(config):
    intake, frames_root = ROOT / config.intake, ROOT / config.frames
    for name, expected in read(intake / "preparation-lock.json")["hashes"].items():
        assert digest(name) == expected, name
    frames_root.mkdir(parents=True, exist_ok=True)
    temporary = intake / "temporary"
    temporary.mkdir(exist_ok=True)
    index, selection = read(ROOT / config.archive_index), read(intake / "selection.json")
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    progress_path = intake / "progress.json"
    previous = read(progress_path).get("download_bytes", 0) if progress_path.exists() else 0
    remote = SplitHTTPFile(index["parts"], index["revision"], int(config.network_budget_gib * 2**30 - previous))
    records, total = [], 0

    def progress():
        write(
            progress_path,
            {
                "stage": "acquire",
                "clips": len(records),
                "frames": total,
                "target_frames": config.max_batches * 32,
                "download_bytes": previous + remote.received,
                "free_bytes": shutil.disk_usage(frames_root).free,
            },
        )

    try:
        with zipfile.ZipFile(remote) as archive:
            for selected in selection["clips"]:
                if total >= config.max_batches * 32:
                    break
                clip = selected["clip_id"]
                directory = frames_root / clip
                saved = directory / "source.json"
                if saved.exists():
                    record = read(saved)
                    assert digest(record["frames_path"]) == record["frames_sha256"]
                    records.append(record)
                    total += record["shape"][0]
                    progress()
                    continue
                assert shutil.disk_usage(frames_root).free > 768 * 2**20
                member = archive.getinfo(selected["path"])
                assert (member.file_size, member.compress_size, member.CRC, member.header_offset) == (
                    selected["bytes"],
                    selected["compressed_bytes"],
                    selected["crc32"],
                    selected["offset"],
                )
                raw_path = temporary / Path(member.filename).name
                if not raw_path.exists():
                    for attempt in range(3):
                        try:
                            payload = archive.read(member)
                            break
                        except Exception as error:
                            progress()
                            write(
                                intake / f"network-attempt-{time.time_ns()}.json",
                                {
                                    "clip_id": clip,
                                    "attempt": attempt,
                                    "error": repr(error),
                                    "download_bytes": previous + remote.received,
                                },
                            )
                            if attempt == 2:
                                raise
                            time.sleep(2)
                    assert len(payload) == member.file_size and zlib.crc32(payload) == member.CRC
                    raw_path.write_bytes(payload)
                    del payload
                assert raw_path.stat().st_size == member.file_size
                assert zlib.crc32(raw_path.read_bytes()) == member.CRC
                raw_hash = digest(raw_path)
                progress()
                decoded = decode(ffmpeg, raw_path, directory, config.max_batches * 32 - total)
                record = {**selected, **decoded, "video_sha256": raw_hash, "temporary_video_removed": True}
                write(saved, record)
                compress(Path(record["frames_path"]))
                assert raw_path.resolve().parent == temporary.resolve() and raw_path.suffix.lower() in (".mov", ".mp4")
                raw_path.unlink()
                records.append(record)
                total += record["shape"][0]
                progress()
                print({"clips": len(records), "frames": total, "target": config.max_batches * 32}, flush=True)
    except Exception as error:
        progress()
        write(
            intake / f"failure-{time.time_ns()}.json",
            {"error": repr(error), "download_bytes": previous + remote.received},
        )
        raise
    used = total // 32 * 32
    assert used // 32 >= config.minimum_batches, "Cohort too short; do not change it using learner outcomes"
    write(
        frames_root / "intake.json",
        {
            "complete": True,
            "clips": records,
            "frames_decoded": total,
            "frames_used": used,
            "terminal_frames_not_batched": total - used,
            "batches": used // 32,
            "ffmpeg_path": str(ffmpeg),
            "ffmpeg_sha256": digest(ffmpeg),
            "download_bytes": previous + remote.received,
            "selection_sha256": digest(intake / "selection.json"),
            "natural_capture_chronology_established": False,
        },
    )
    print({"intake_complete": True, "batches": used // 32, "clips": len(records)}, flush=True)
