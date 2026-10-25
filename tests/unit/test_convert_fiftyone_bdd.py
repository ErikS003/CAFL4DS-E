"""Tests for the lightweight FiftyOne-to-BDD conversion helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]


def test_converter_writes_native_labels_and_hardlinks(tmp_path: Path) -> None:
    """The mirror export becomes the exact canonical layout consumed by BDD100KSource."""
    spec = importlib.util.spec_from_file_location("_convert_fiftyone_bdd", _ROOT / "scripts/convert_fiftyone_bdd.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(source / "data/example.jpg")
    sample = {
        "filepath": "data/example.jpg",
        "metadata": {"width": 20, "height": 10},
        "weather": {"label": "clear"},
        "scene": {"label": "highway"},
        "timeofday": {"label": "daytime"},
        "detections": {
            "detections": [
                {
                    "label": "car",
                    "bounding_box": [0.1, 0.2, 0.3, 0.4],
                    "occluded": True,
                    "truncated": False,
                }
            ]
        },
    }
    (source / "samples.json").write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

    count, labels_file = script.convert(source, tmp_path / "output")
    records = json.loads(labels_file.read_text(encoding="utf-8"))
    image = tmp_path / "output/images/100k/val/example.jpg"
    assert count == 1 and image.samefile(source / "data/example.jpg")
    assert records[0]["attributes"] == {"weather": "clear", "scene": "highway", "timeofday": "daytime"}
    assert records[0]["labels"][0]["category"] == "car"
    assert records[0]["labels"][0]["box2d"] == pytest.approx({"x1": 2.0, "y1": 2.0, "x2": 8.0, "y2": 6.0})
