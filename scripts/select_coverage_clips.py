"""Re-rank an audited warm-disjoint eligible pool, excluding all downloaded clips."""

import argparse
import hashlib
import json
from pathlib import Path

from cafl4ds.artifacts import atomic_json
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old = ROOT / "outputs/scheduling/20261013-intake"
    selection = json.loads((old / "selection.json").read_text())
    pool_file = old / "selection.pool.json"
    if (
        digest(pool_file) != selection["eligible_pool_sha256"]
        or not json.loads((ROOT / "outputs/scheduling/20261013-gpu/independent-verification.json").read_text())[
            "passed"
        ]
    ):
        raise ValueError("Eligible-pool provenance not audited")
    known = ROOT / ".cache/coverage-known-videos.txt"
    excluded = set(selection["excluded_clip_ids"]) | {
        Path(p).stem for p in known.read_text(encoding="utf-8").splitlines()
    }
    pools = json.loads(pool_file.read_text(encoding="utf-8"))
    fresh, clips = {}, []
    for key, pool in pools.items():
        candidates = [c for c in pool if c["clip_id"] not in excluded]
        candidates.sort(
            key=lambda c: hashlib.sha256(f"coverage-admission:20261013:{c['clip_id']}.jpg".encode()).digest()
        )
        if len(candidates) < 36:
            raise ValueError("Insufficient fresh pool")
        fresh[key] = candidates
        clips.extend(candidates[:36])
    for c in clips:
        metadata = Path(c["keyframe_path"]).parents[3] / "labels/train" / f"{c['clip_id']}.json"
        if json.loads(metadata.read_text())["attributes"] != c["keyframe_attributes"]:
            raise ValueError("Selected metadata changed")
    pool = args.output.with_suffix(".pool.json")
    atomic_json(pool, fresh)
    atomic_json(
        args.output,
        {
            "purpose": "accepted-batch coverage admission on fresh short/medium/long dwell streams",
            "clips": clips,
            "excluded_clip_ids": sorted(excluded),
            "eligible_pool_sha256": digest(pool),
            "original_pool_path": str(pool_file),
            "original_pool_sha256": digest(pool_file),
            "local_inventory_sha256": digest(known),
            "selection_seed": "coverage-admission:20261013",
            "warm_overlap_count": 0,
            "validation_overlap_count": 0,
            "warm_partition_seed": selection["warm_partition_seed"],
            "warm_fraction": selection["warm_fraction"],
            "archive_index_sha256": selection["archive_index_sha256"],
            "partition_metadata_sha256": selection["partition_metadata_sha256"],
        },
    )
    print(json.dumps({"selected": len(clips), "excluded": len(excluded), "warm_overlap": 0}), flush=True)


if __name__ == "__main__":
    main()
