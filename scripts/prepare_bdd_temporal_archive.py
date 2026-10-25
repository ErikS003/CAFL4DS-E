"""Read a split remote ZIP with bounded HTTP ranges; retrieve explicitly chosen clips.

Never downloads an entire archive part. ZIP member CRCs and independent SHA256
digests are retained, but these do not authenticate a community mirror to BAIR.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

import requests

REPOSITORY = "linxxx3/bdd100k_videos"
DIRECTORY = "bdd100k_videos"


class SplitHTTPFile(io.RawIOBase):
    """Seekable concatenation with strict range, response-length and byte budgets."""

    def __init__(self, parts: list[dict[str, Any]], revision: str, budget: int) -> None:
        """Pin archive part sizes, repository revision and transfer limit."""
        super().__init__()
        self.parts = parts
        self.revision = revision
        self.budget = budget
        self.received = 0
        self.position = 0
        self.edges = [0]
        for part in parts:
            self.edges.append(self.edges[-1] + part["size"])
        self.session = requests.Session()

    def seekable(self) -> bool:
        """Allow ZIP-directory seeks."""
        return True

    def readable(self) -> bool:
        """Expose a read-only stream."""
        return True

    def tell(self) -> int:
        """Return the concatenated byte offset."""
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move without fetching data."""
        position = offset + (0, self.position, self.edges[-1])[whence]
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        """Retrieve only the requested bounded range across part boundaries."""
        size = min(self.edges[-1] - self.position, size if size >= 0 else self.edges[-1])
        if size <= 0:
            return b""
        if self.received + size > self.budget:
            raise ValueError("remote byte budget exceeded")
        chunks = []
        while size:
            index = bisect.bisect_right(self.edges, self.position) - 1
            start = self.position - self.edges[index]
            count = min(size, self.edges[index + 1] - self.position, 16 * 1024 * 1024)
            end = start + count - 1
            part = self.parts[index]
            url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{self.revision}/{part['path']}"
            # Avoid a cached signed redirect issued for a different byte range.
            with self.session.get(
                url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                params={"download": "true", "range_key": f"{start}-{end}"}, stream=True, timeout=60,
            ) as response:
                expected = f"bytes {start}-{end}/{part['size']}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"server did not honor bounded range: {response.status_code}")
                self.received += count
                data = response.raw.read(count + 1)
                if len(data) != count:
                    raise ValueError("incorrect HTTP range body length")
            chunks.append(data)
            self.position += count
            size -= count
        return b"".join(chunks)


def main() -> None:
    """Write the archive index and optionally retrieve a frozen selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="JSON with clips list containing clip_id and split")
    parser.add_argument("--max-download-mib", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = requests.get(f"https://huggingface.co/api/datasets/{REPOSITORY}", timeout=30)
    info.raise_for_status()
    revision = info.json()["sha"]
    response = requests.get(
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/{DIRECTORY}", timeout=30,
    )
    response.raise_for_status()
    parts = sorted((p for p in response.json() if ".zip.part-" in p["path"]), key=lambda p: p["path"])
    if not parts:
        raise ValueError("archive parts not found")
    remote = SplitHTTPFile(parts, revision, args.max_download_mib * 1024 * 1024)
    with zipfile.ZipFile(remote) as archive:
        videos = [entry for entry in archive.infolist() if Path(entry.filename).suffix.lower() == ".mov"]
        videos += [entry for entry in archive.infolist() if Path(entry.filename).suffix.lower() == ".mp4"]
        index = {
            "repository": REPOSITORY, "revision": revision, "parts": parts,
            "archive_bytes": remote.edges[-1], "video_count": len(videos),
            "index_download_bytes": remote.received,
            "entries": [{"path": e.filename, "bytes": e.file_size, "compressed_bytes": e.compress_size,
                         "crc32": e.CRC, "offset": e.header_offset} for e in videos],
        }
        (args.output / "archive_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in index.items() if k not in {"parts", "entries"}}), flush=True)
        if args.selection:
            selection_bytes = args.selection.read_bytes()
            selection = json.loads(selection_bytes)
            records = []
            by_id: dict[str, list[zipfile.ZipInfo]] = {}
            for entry in videos:
                by_id.setdefault(Path(entry.filename).stem, []).append(entry)
            for clip in selection["clips"]:
                matches = [e for e in by_id.get(clip["clip_id"], []) if clip["split"] in Path(e.filename).parts]
                if len(matches) != 1:
                    raise ValueError(f"expected one matching video for {clip['clip_id']}, found {len(matches)}")
                entry = matches[0]
                if entry.file_size > 128 * 1024 * 1024:
                    raise ValueError("selected clip exceeds the 128 MiB extraction limit")
                destination = args.output / "videos" / clip["split"] / Path(entry.filename).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    data = destination.read_bytes()
                    if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                        raise ValueError(f"existing video fails archive integrity: {destination}")
                else:
                    data = archive.read(entry)  # zipfile verifies the decompressed member CRC.
                    destination.with_suffix(destination.suffix + ".partial").write_bytes(data)
                    destination.with_suffix(destination.suffix + ".partial").replace(destination)
                records.append({**clip, "video_path": str(destination.resolve()), "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(), "archive_path": entry.filename,
                                "crc32": entry.CRC})
                manifest = {"repository": REPOSITORY, "revision": revision,
                            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                            "download_bytes": remote.received, "clips": records,
                            "complete": len(records) == len(selection["clips"]), "updated_unix": time.time()}
                (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"clips {len(records)}/{len(selection['clips'])}; bytes {remote.received}", flush=True)


if __name__ == "__main__":
    main()
