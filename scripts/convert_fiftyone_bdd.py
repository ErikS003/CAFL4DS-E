"""Convert the ETH/FiftyOne BDD100K validation export to the project's native layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _record(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one FiftyOne sample document to BDD's Detection 2020 JSON shape."""
    metadata = sample["metadata"]
    width, height = float(metadata["width"]), float(metadata["height"])
    labels = []
    for detection in sample.get("detections", {}).get("detections", []):
        x, y, w, h = (float(value) for value in detection["bounding_box"])
        labels.append(
            {
                "category": detection["label"],
                "attributes": {
                    "occluded": bool(detection.get("occluded", False)),
                    "truncated": bool(detection.get("truncated", False)),
                    "trafficLightColor": detection.get("trafficLightColor", "none"),
                },
                "box2d": {
                    "x1": x * width,
                    "y1": y * height,
                    "x2": (x + w) * width,
                    "y2": (y + h) * height,
                },
            }
        )
    return {
        "name": Path(sample["filepath"]).name,
        "attributes": {
            "weather": sample["weather"]["label"],
            "scene": sample["scene"]["label"],
            "timeofday": sample["timeofday"]["label"],
        },
        "labels": labels,
    }


def convert(source_root: Path, output_root: Path) -> tuple[int, Path]:
    """Hard-link images and write a native ``det_val.json`` without duplicating image bytes."""
    source_root = source_root.resolve()
    manifest = source_root / "samples.json"
    source_images = source_root / "data"
    if not manifest.is_file() or not source_images.is_dir():
        raise FileNotFoundError(f"expected samples.json and data/ under {source_root}")
    samples = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    images_dir = output_root / "images" / "100k" / "val"
    labels_file = output_root / "labels" / "det_20" / "det_val.json"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in samples:
        source = source_root / sample["filepath"]
        target = images_dir / source.name
        if not source.is_file():
            raise FileNotFoundError(f"missing source image {source}")
        if not target.exists():
            os.link(source, target)
        elif target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"existing target differs from source: {target}")
        records.append(_record(sample))

    labels_file.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    return len(records), labels_file


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path, help="Cloned dgural/bdd100k export")
    parser.add_argument("output_root", type=Path, help="Canonical BDD100K destination")
    args = parser.parse_args()
    count, labels_file = convert(args.source_root, args.output_root)
    print(f"converted {count} validation samples; labels: {labels_file}")


if __name__ == "__main__":
    main()
