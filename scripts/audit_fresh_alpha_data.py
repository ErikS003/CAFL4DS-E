"""Inventory unused BDD cells before allocating a new alpha-response experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import (
    BDD100KSource,
    _count_categories,
    _driving_attributes,
    _filter_valid_records,
    _image_filename,
    _iter_bdd_records,
)
from scripts.evaluate_checkpoint_robustness import digest
from scripts.run_slow_edge_pilot import ROOT


def image_paths(value: object) -> set[str]:
    """Collect recorded image paths recursively, including nested support and stream panels."""
    if isinstance(value, str) and Path(value).suffix.lower() in (".jpg", ".jpeg", ".png"):
        return {str(Path(value).resolve()).casefold()}
    if isinstance(value, dict):
        return set().union(*(image_paths(k) | image_paths(v) for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(image_paths(v) for v in value))
    return set()


def inventory() -> dict[str, Any]:
    """Read metadata only; reserve no images and read no historical response outcomes."""
    original = json.loads((ROOT / "outputs/image-utility/20261006-bdd-alpha-response/inputs.json").read_text())
    manifests = sorted(set(ROOT.glob("outputs/**/inputs.json")) | set(ROOT.glob("outputs/**/manifest.json")))
    excluded: set[str] = set()
    history = {}
    for path in manifests:
        recorded = image_paths(json.loads(path.read_text()))
        excluded.update(recorded)
        history[str(path)] = {"sha256": digest(path), "image_paths": len(recorded)}
    domains = original["evaluation"]["domain_names"]
    classes = [original["evaluation"]["class_names"][str(i)] for i in range(3)]
    pools = {}
    for split in ("val", "train"):
        config = {k: v for k, v in original["config"]["data"].items() if not k.startswith("_")}
        config.update(split=split, partition_role="stream" if split == "train" else None)
        source = BDD100KSource(**config)
        image_dir, label_file = source._paths()
        records = []
        for record in _iter_bdd_records(label_file):
            attrs = _driving_attributes(record)
            path = image_dir / _image_filename(record)
            if attrs is not None and path.is_file():
                records.append((path, (attrs[0], attrs[1]), attrs[2], _count_categories(record)))
        records = _filter_valid_records(
            records,
            min_canary_count=source.min_canary_count,
            partition_role=source.partition_role,
            warm_fraction=source.warm_fraction,
            partition_seed=source.partition_seed,
        )
        cells: dict[str, dict[str, Any]] = {
            f"{d}_{c}": {"total": 0, "unused_paths": []} for d in range(2) for c in range(3)
        }
        for path, domain, scene, _ in records:
            name = "·".join(domain)
            if name in domains and scene in classes:
                cell = cells[f"{domains.index(name)}_{classes.index(scene)}"]
                cell["total"] += 1
                if str(path.resolve()).casefold() not in excluded:
                    cell["unused_paths"].append(str(path.resolve()))
        for cell in cells.values():
            cell["unused_paths"].sort()
            cell["unused"] = len(cell["unused_paths"])
        pools[split] = cells
    return {
        "domain_names": domains,
        "class_names": classes,
        "config": original["config"],
        "historical_manifests": history,
        "excluded_image_count": len(excluded),
        "pools": pools,
        "limits": "Recorded paths excluded; unlisted historical exposure and sequence independence unknown.",
    }


def main() -> None:
    """Persist the allocation audit separately from any outcome study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = inventory()
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                split: {key: {"total": cell["total"], "unused": cell["unused"]} for key, cell in cells.items()}
                for split, cells in result["pools"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
