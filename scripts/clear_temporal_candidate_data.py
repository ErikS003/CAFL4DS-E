"""CPU-only exposure and duplicate screening, without learner/query predictions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HAMMING_MAX = 4
RGB8_RMSE_MAX = 0.03


def atomic(path: Path, value: dict) -> None:
    """Persist readable progress through an atomic replacement."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fingerprint(path: Path) -> dict:
    """Use original bytes, training-size pixels and a label-free visual descriptor."""
    data = path.read_bytes()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        resized = np.asarray(rgb.resize((64, 64), Image.Resampling.BILINEAR))
        small = np.asarray(rgb.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.uint8)
        grey = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = np.packbits((grey[:, 1:] > grey[:, :-1]).flatten())
    return {
        "path": str(path),
        "clip_id": path.stem.casefold(),
        "bytes_sha256": hashlib.sha256(data).hexdigest(),
        "rgb64_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
        "dhash": int.from_bytes(bits.tobytes(), "little"),
        "rgb8": base64.b64encode(small.tobytes()).decode(),
        "size": len(data),
        "mtime_ns": path.stat().st_mtime_ns,
    }


def arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert descriptors without loading an encoder."""
    hashes = np.asarray([r["dhash"] for r in records], dtype=np.uint64)
    pixels = np.stack([np.frombuffer(base64.b64decode(r["rgb8"]), dtype=np.uint8) for r in records])
    return hashes, pixels.astype(np.float32) / 255


def main() -> None:  # noqa: C901 - staged immutable evidence and resumable CPU scan
    """Conservatively screen candidates against all historical train/val pixels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    completed = output / "clearance.json"
    if completed.exists():
        raise FileExistsError(completed)
    started = time.time()
    allocation = json.loads(args.allocation.read_text())
    audit = json.loads(Path(allocation["audit_path"]).read_text())
    dataset = Path(audit["source_config"]["bdd_root"])
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "allocation_sha256": hashlib.sha256(args.allocation.read_bytes()).hexdigest(),
        "source_sha256": source_hash,
        "hamming_max": HAMMING_MAX,
        "rgb8_rmse_max": RGB8_RMSE_MAX,
        "near_rule": "dHash Hamming <=4 AND pooled RGB8 RMSE <=.03; screen, not proof of independence",
        "historical_scope": "All local train/val images, including whole warm/stream partitions; old frame caches exact RGB64",
        "query_outcomes_read": False,
        "session_independence_established": False,
        "matching_policy": "Record every match; do not train, change roles or silently discard matches.",
    }
    frozen = output / "screening_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Screening source/protocol changed during resume")
    if not frozen.exists():
        atomic(frozen, protocol)
        (output / "screening_source.py").write_bytes(Path(__file__).read_bytes())

    def progress(stage: str, **fields: object) -> None:
        atomic(
            output / "status.json",
            {"stage": stage, "pid": os.getpid(), "started_unix": started, "updated_unix": time.time(), **fields},
        )

    try:
        candidates = []
        for row in allocation["allocations"]:
            record = fingerprint(Path(row["image_path"]))
            if record["bytes_sha256"] != row["image_sha256"]:
                raise ValueError(f"Candidate changed: {record['path']}")
            if hashlib.sha256(Path(row["label_path"]).read_bytes()).hexdigest() != row["label_sha256"]:
                raise ValueError("Candidate labels changed")
            candidates.append({**record, "role": row["role"], "cell": row["cell"]})
        atomic(output / "candidate_fingerprints.json", {"records": candidates})
        historical_paths = sorted(
            set((dataset / "images/100k/train").glob("*.jpg")) | set((dataset / "images/100k/val").glob("*.jpg"))
        )
        if len(historical_paths) != 80000:
            raise ValueError(f"Expected full 70k train + 10k val, found {len(historical_paths)}")
        historical = []
        cache = output / "historical_fingerprints.jsonl"
        if cache.exists():
            for line in cache.read_text().splitlines():
                record = json.loads(line)
                path = Path(record["path"])
                if path.stat().st_size != record["size"] or path.stat().st_mtime_ns != record["mtime_ns"]:
                    raise ValueError("Historical image changed since fingerprint")
                historical.append(record)
            if [r["path"] for r in historical] != [str(p) for p in historical_paths[: len(historical)]]:
                raise ValueError("Historical cache order differs")
        progress("fingerprinting_history", completed=len(historical), total=len(historical_paths))
        with cache.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=4) as pool:
            for record in pool.map(fingerprint, historical_paths[len(historical) :]):
                historical.append(record)
                stream.write(json.dumps(record) + "\n")
                if len(historical) % 1000 == 0:
                    stream.flush()
                    progress("fingerprinting_history", completed=len(historical), total=len(historical_paths))
                    print(f"Historical images {len(historical)}/{len(historical_paths)}", flush=True)
        by_bytes: dict[str, list[dict]] = {}
        by_rgb: dict[str, list[dict]] = {}
        for row in historical:
            by_bytes.setdefault(row["bytes_sha256"], []).append(row)
            by_rgb.setdefault(row["rgb64_sha256"], []).append(row)
        exact = []
        for row in candidates:
            for key, lookup in [("bytes_sha256", by_bytes), ("rgb64_sha256", by_rgb)]:
                for other in lookup.get(row[key], []):
                    exact.append(
                        {
                            "candidate": row["clip_id"],
                            "role": row["role"],
                            "historical_path": other["path"],
                            "kind": key,
                        }
                    )
        candidate_hashes, candidate_rgb = arrays(candidates)
        historical_hashes, historical_rgb = arrays(historical)
        counts = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
        near = []
        for start in range(0, len(historical), 128):
            xor = np.bitwise_xor(historical_hashes[start : start + 128, None], candidate_hashes[None, :])
            distance = counts[xor.view(np.uint8).reshape(len(xor), len(candidates), 8)].sum(axis=2)
            for h, c in np.argwhere(distance <= HAMMING_MAX):
                rmse = float(np.sqrt(np.mean((historical_rgb[start + h] - candidate_rgb[c]) ** 2)))
                if rmse <= RGB8_RMSE_MAX:
                    near.append(
                        {
                            "candidate": candidates[c]["clip_id"],
                            "role": candidates[c]["role"],
                            "historical_path": historical[start + h]["path"],
                            "hamming": int(distance[h, c]),
                            "rgb8_rmse": rmse,
                        }
                    )
            if start % 4096 == 0:
                progress("near_duplicate_history", completed=start, total=len(historical), matches=len(near))
        cross_roles = []
        for c, row in enumerate(candidates):
            xor = np.bitwise_xor(candidate_hashes[c + 1 :], candidate_hashes[c])
            distance = counts[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for j in np.where(distance <= HAMMING_MAX)[0]:
                other = candidates[c + 1 + j]
                rmse = float(np.sqrt(np.mean((candidate_rgb[c] - candidate_rgb[c + 1 + j]) ** 2)))
                if rmse <= RGB8_RMSE_MAX:
                    cross_roles.append(
                        {
                            "candidate": row["clip_id"],
                            "other": other["clip_id"],
                            "role": row["role"],
                            "other_role": other["role"],
                            "hamming": int(distance[j]),
                            "rgb8_rmse": rmse,
                        }
                    )
        old_intake = json.loads((ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json").read_text())
        old_frame_hashes = {frame["rgb64_sha256"] for clip in old_intake["clips"] for frame in clip["frames"]}
        frame_matches = [r["clip_id"] for r in candidates if r["rgb64_sha256"] in old_frame_hashes]
        passed = not (exact or near or cross_roles or frame_matches)
        atomic(
            completed,
            {
                "protocol": protocol,
                "historical_images": len(historical),
                "candidates": len(candidates),
                "exact_history_matches": exact,
                "near_history_matches": near,
                "within_allocation_near_matches": cross_roles,
                "old_stream_frame_exact_matches": frame_matches,
                "screen_passed_without_matches": passed,
                "training_ready": False,
                "elapsed_seconds": time.time() - started,
                "remaining": "Provenance, role/group limitations, video/keyframe correspondence and temporal regime gate",
            },
        )
        progress(
            "complete",
            screen_passed_without_matches=passed,
            exact_matches=len(exact),
            near_matches=len(near),
            cross_role_matches=len(cross_roles),
        )
    except Exception as error:
        progress("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
