"""Audit local temporal qualification inputs without scoring models or allocating data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import (
    BDD100KSource,
    _driving_attributes,
    _filter_valid_records,
    _image_filename,
    _iter_bdd_records,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAMES = {
    "inputs.json",
    "manifest.json",
    "data_lock.json",
    "input_lock.json",
    "parent_lock.json",
    "cache_manifest.json",
    "intake.json",
    "selection.json",
    "selection-recheck.json",
    "download_manifest.json",
    "frozen_selections.json",
    "selection_frozen.json",
}
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".mov", ".mp4"}


def digest(path: Path) -> str:
    """Hash a provenance file without loading media."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_ids(value: object, field: str = "") -> set[str]:
    """Conservatively collect media stems and explicit clip IDs, including map keys."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.suffix.lower() in MEDIA_SUFFIXES:
            return {path.stem.casefold()}
        if field in {"clip_id", "clip_ids", "capture_group"}:
            return {value.casefold()}
        return set()
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            result.update(recorded_ids(key))
            result.update(recorded_ids(item, str(key)))
        return result
    if isinstance(value, list):
        return set().union(*(recorded_ids(item, field) for item in value))
    return set()


def index_fields(value: object) -> set[str]:
    """Flag historical index-only exposure requiring source-order reconstruction."""
    if isinstance(value, dict):
        result = {str(k) for k in value if k in {"original_indices", "train_source_indices", "val_source_indices"}}
        for item in value.values():
            result.update(index_fields(item))
        return result
    if isinstance(value, list):
        return set().union(*(index_fields(item) for item in value if isinstance(item, (dict, list))))
    return set()


def audit() -> dict[str, Any]:  # noqa: C901 - explicit provenance and metadata stages
    """Inventory upper bounds on fresh candidates; never assert source independence."""
    original_path = ROOT / "outputs/image-utility/20261006-bdd-alpha-response/inputs.json"
    original = json.loads(original_path.read_text())
    domains = original["evaluation"]["domain_names"]
    classes = [original["evaluation"]["class_names"][str(i)] for i in range(3)]
    excluded: set[str] = set()
    history = {}
    index_only = {}
    for path in sorted(ROOT.glob("outputs/**/*.json")):
        if path.name not in MANIFEST_NAMES:
            continue
        data = json.loads(path.read_text())
        ids = recorded_ids(data)
        excluded.update(ids)
        relative = path.relative_to(ROOT).as_posix()
        history[relative] = {"sha256": digest(path), "recorded_media_ids": len(ids)}
        fields = index_fields(data)
        if fields:
            index_only[relative] = sorted(fields)
    print(f"Read {len(history)} allocation manifests; {len(excluded)} recorded media IDs", flush=True)
    archive_path = ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json"
    archive = json.loads(archive_path.read_text())
    archive_ids = {
        split: {Path(e["path"]).stem.casefold() for e in archive["entries"] if f"/{split}/" in e["path"]}
        for split in ("train", "val", "test")
    }
    local_videos = {p.stem.casefold() for p in (ROOT / "outputs/temporal-bdd").glob("**/*.mov")}
    pools = {}
    for split in ("val", "train", "test"):
        config = {k: v for k, v in original["config"]["data"].items() if not k.startswith("_")}
        config.update(split=split, partition_role="stream" if split == "train" else None)
        source = BDD100KSource(**config)
        image_dir, labels = source._paths()
        records = []
        scanned, missing_attributes = 0, 0
        label_hash = hashlib.sha256()
        # Commit to metadata as well as its path; no model outputs are read.
        label_files = sorted(labels.glob("*.json")) if labels.is_dir() else [labels]
        for label in label_files:
            label_hash.update(label.name.encode())
            label_hash.update(bytes.fromhex(digest(label)))
        for record in _iter_bdd_records(labels):
            scanned += 1
            attrs = _driving_attributes(record)
            if attrs is None:
                missing_attributes += 1
                continue
            path = image_dir / _image_filename(record)
            if path.is_file():
                records.append((path, (attrs[0], attrs[1]), attrs[2], {}))
        records = _filter_valid_records(
            records,
            min_canary_count=source.min_canary_count,
            partition_role=source.partition_role,
            warm_fraction=source.warm_fraction,
            partition_seed=source.partition_seed,
        )
        cells = {
            f"{domain}/{scene}": {
                "eligible": 0,
                "candidate_ids": [],
                "archive_candidates": 0,
                "local_video_candidates": 0,
            }
            for domain in domains
            for scene in classes
        }
        for path, domain, scene, _ in records:
            key = f"{'Â·'.join(domain)}/{scene}"
            if key not in cells:
                continue
            cell = cells[key]
            cell["eligible"] += 1
            clip_id = path.stem.casefold()
            if clip_id in excluded:
                continue
            cell["candidate_ids"].append(clip_id)
            cell["archive_candidates"] += int(clip_id in archive_ids[split])
            cell["local_video_candidates"] += int(clip_id in local_videos)
        for cell in cells.values():
            cell["candidate_ids"].sort()
            cell["candidate_count_upper_bound"] = len(cell["candidate_ids"])
        pools[split] = {
            "metadata_records": scanned,
            "missing_attributes": missing_attributes,
            "labels_path": str(labels),
            "label_files": len(label_files),
            "metadata_digest": label_hash.hexdigest(),
            "cells": cells,
        }
        print(f"Finished {split}: {scanned} metadata records", flush=True)
    intake_path = ROOT / "outputs/temporal-bdd/20261011-frames-v3/intake.json"
    intake = json.loads(intake_path.read_text())
    clips = intake["clips"]
    spans = [c["frames"][-1]["pts_seconds"] - c["frames"][0]["pts_seconds"] for c in clips]
    empty_cells = {
        split: [key for key, c in pool["cells"].items() if c["candidate_count_upper_bound"] == 0]
        for split, pool in pools.items()
    }
    return {
        "purpose": "CPU-only temporal benchmark readiness; metadata audit, no scientific run",
        "device": "cpu",
        "domain_names": domains,
        "class_names": classes,
        "source_config": original["config"]["data"],
        "history": history,
        "recorded_media_ids": len(excluded),
        "index_exposure_requiring_reconstruction": index_only,
        "candidate_interpretation": (
            "Upper bounds only: path/clip exclusions, no exact/near-duplicate or session clearance; "
            "index-only exposure unresolved."
        ),
        "pools": pools,
        "empty_candidate_cells": empty_cells,
        "archive": {
            "sha256": digest(archive_path),
            "repository": archive["repository"],
            "revision": archive["revision"],
            "split_video_counts": {k: len(v) for k, v in archive_ids.items()},
            "availability": "Remote index only; not downloaded or verified fresh media.",
        },
        "existing_temporal": {
            "intake_sha256": digest(intake_path),
            "clips": len(clips),
            "cached_frames": sum(len(c["frames"]) for c in clips),
            "span_seconds_min": min(spans),
            "span_seconds_max": max(spans),
            "session_ids_known": sum(c.get("session_id") is not None for c in clips),
            "new_local_clip_ids": sorted(local_videos - excluded),
            "labels_scope": "10-second keyframe only; no per-frame scene annotations established.",
        },
        "ready_for_independent_qualification": False,
        "next_gate": (
            "Validate candidate label provenance, historical exposure and duplicates; acquire new footage "
            "and clear role/group separation before freezing the outcome protocol."
        ),
    }


def main() -> None:
    """Persist immutable audit evidence separately from completed studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    atomic_json(args.output, result)
    print(
        json.dumps(
            {"empty_cells": result["empty_candidate_cells"], "ready": result["ready_for_independent_qualification"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
