"""Bounded clip intake from the already indexed, pinned community archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
import zlib
from pathlib import Path

from cafl4ds.artifacts import atomic_json
from scripts.prepare_bdd_temporal_archive import REPOSITORY, SplitHTTPFile


def main() -> None:
    """Verify frozen member locations, CRCs and strict HTTP range budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-download-mib", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.archive_index.read_text())
    selection = json.loads(args.selection.read_text())
    if index["repository"] != REPOSITORY:
        raise ValueError("Unexpected archive repository")
    expected = {e["path"]: e for e in index["entries"]}
    selected = [f"bdd100k/videos/100k/{c['split']}/{c['clip_id']}.mov" for c in selection["clips"]]
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate selected members")
    required = sum(expected[name]["compressed_bytes"] for name in selected) + index["index_download_bytes"]
    if required > args.max_download_mib * 1024**2:
        raise ValueError("Selection exceeds the declared intake budget")
    remote = SplitHTTPFile(index["parts"], index["revision"], args.max_download_mib * 1024**2)
    records = []
    with zipfile.ZipFile(remote) as archive:
        for clip, name in zip(selection["clips"], selected, strict=True):
            entry = archive.getinfo(name)
            locked = expected[name]
            if (entry.file_size, entry.compress_size, entry.CRC, entry.header_offset) != (
                locked["bytes"],
                locked["compressed_bytes"],
                locked["crc32"],
                locked["offset"],
            ):
                raise ValueError("Archive member differs from pinned index")
            if entry.file_size > 128 * 1024**2:
                raise ValueError("Member exceeds 128 MiB limit")
            path = args.output / "videos" / clip["split"] / Path(name).name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                data = path.read_bytes()
            else:
                data = archive.read(entry)
                temporary = path.with_suffix(".mov.partial")
                temporary.write_bytes(data)
                temporary.replace(path)
            if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                raise ValueError("Video CRC/length mismatch")
            records.append(
                {
                    **clip,
                    "video_path": str(path.resolve()),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "archive_path": name,
                    "crc32": entry.CRC,
                }
            )
            atomic_json(
                args.output / "download_manifest.json",
                {
                    "repository": REPOSITORY,
                    "revision": index["revision"],
                    "archive_index_sha256": hashlib.sha256(args.archive_index.read_bytes()).hexdigest(),
                    "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "download_bytes": remote.received,
                    "max_download_mib": args.max_download_mib,
                    "clips": records,
                    "complete": len(records) == len(selected),
                    "updated_unix": time.time(),
                    "scope": selection["purpose"],
                    "source_authenticity": "Pinned community mirror integrity; BAIR authenticity not established",
                },
            )
            print(
                f"Downloaded {len(records)}/{len(selected)}; transfer {remote.received / 1024**2:.1f} MiB", flush=True
            )


if __name__ == "__main__":
    main()
