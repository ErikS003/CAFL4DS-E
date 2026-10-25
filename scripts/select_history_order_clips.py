"""Select a bounded label-free SSL stream, excluding the established warm clip partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cafl4ds.data.attributes import (
    _driving_attributes,
    _filter_valid_records,
    _image_filename,
    _iter_bdd_records,
)

CELLS = {("daytime", scene): 12 for scene in ("city street", "highway")}
SELECTION_SEED = "history-order:20261013"


def main() -> None:
    """Write a deterministic selection before any video decoding or SSL outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_bytes = args.archive_index.read_bytes()
    index = json.loads(index_bytes)
    available = {Path(e["path"]).stem for e in index["entries"] if "train" in Path(e["path"]).parts}
    valid: list[tuple[Path, tuple[str, str], str, dict[str, int]]] = []
    metadata = {}
    prior_path = Path("outputs/temporal-bdd/20261011-intake/selection.json")
    prior = json.loads(prior_path.read_text())
    known_path = Path(".cache/history-known-videos.txt")
    excluded = {c["clip_id"] for c in prior["clips"]} | {Path(p).stem for p in known_path.read_text().splitlines()}
    for record in _iter_bdd_records(args.bdd_root / "labels/train"):
        attributes = _driving_attributes(record)
        if attributes is None:
            continue
        timeofday, weather, scene = attributes
        path = args.bdd_root / "images/100k/train" / _image_filename(record)
        if path.is_file():
            valid.append((path, (timeofday, weather), scene, {}))
            metadata[path.stem] = {
                "clip_id": path.stem,
                "split": "train",
                "role": "online_development",
                "keyframe_path": str(path),
                "keyframe_attributes": record["attributes"],
                "capture_group": path.stem,
                "session_id": None,
            }
    # Match the original source's filtering order and exact per-cell filename hash.
    warm = _filter_valid_records(
        valid, min_canary_count=50, partition_role="warm", warm_fraction=0.2, partition_seed=20260921
    )
    stream = _filter_valid_records(
        valid, min_canary_count=50, partition_role="stream", warm_fraction=0.2, partition_seed=20260921
    )
    warm_ids = {v[0].stem for v in warm}
    candidates: Counter[str] = Counter()
    clips: list[dict[str, Any]] = []
    pool = {}
    for (timeofday, scene), count in CELLS.items():
        eligible = [
            v[0]
            for v in stream
            if v[1] == (timeofday, "clear") and v[2] == scene and v[0].stem in available and v[0].stem not in excluded
        ]
        candidates[f"{timeofday}/{scene}"] = len(eligible)
        eligible.sort(key=lambda p: hashlib.sha256(f"{SELECTION_SEED}:{p.name}".encode()).digest())
        pool[f"{timeofday}/{scene}"] = [metadata[p.stem] for p in eligible]
        if len(eligible) < count:
            raise ValueError(f"not enough available stream clips for {timeofday}/{scene}")
        clips.extend(metadata[p.stem] for p in eligible[:count])
    if warm_ids.intersection(c["clip_id"] for c in clips):
        raise ValueError("warm clip overlap")
    val_ids = {p.stem for p in (args.bdd_root / "labels/val").glob("*.json")}
    if val_ids.intersection(c["clip_id"] for c in clips):
        raise ValueError("validation clip overlap")
    selection = {
        "purpose": "matched-exposure order effect; twelve new day city/highway blocks",
        "excluded_clip_ids": sorted(excluded),
        "prior_selection_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "local_video_inventory_sha256": hashlib.sha256(known_path.read_bytes()).hexdigest(),
        "selection_seed": SELECTION_SEED,
        "warm_partition_seed": 20260921,
        "warm_fraction": 0.2,
        "warm_overlap_count": 0,
        "validation_overlap_count": 0,
        "source_independence": "clip-level only; ride/session identities unavailable",
        "attributes_scope": "10-second keyframe; never propagated to training frames as labels",
        "archive_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "partition_metadata_sha256": hashlib.sha256(
            json.dumps(
                sorted((p.name, r, s) for p, r, s, _ in valid),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "candidate_counts": dict(candidates),
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool_path = args.output.with_suffix(".pool.json")
    pool_path.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    selection["eligible_pool_sha256"] = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(clips), "warm_overlap": 0, "candidate_counts": dict(candidates)}))


if __name__ == "__main__":
    main()
