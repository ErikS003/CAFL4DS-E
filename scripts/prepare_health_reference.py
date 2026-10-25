"""Enroll a balanced fixed reference from the historical warm training partition."""

import argparse
import hashlib
import json
from pathlib import Path

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import _driving_attributes, _filter_valid_records, _image_filename, _iter_bdd_records
from scripts.run_short_buffer_experiment import digest


def main() -> None:
    """Select warm-only images before health scoring, without copying their pixels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    root = Path(__file__).resolve().parents[1]
    prior = json.loads((root / "outputs/retention-anchor/20261013-gpu/input_lock.json").read_text())
    excluded = {Path(p).stem for p in prior["hashes"] if Path(p).suffix.lower() in (".jpg", ".jpeg")}
    data = Path("C:/CAFL-E/datasets/bdd100k")
    valid, metadata = [], {}
    for record in _iter_bdd_records(data / "labels/train"):
        attributes = _driving_attributes(record)
        if attributes is None:
            continue
        light, weather, scene = attributes
        path = data / "images/100k/train" / _image_filename(record)
        if path.is_file():
            valid.append((path, (light, weather), scene, {}))
            metadata[path.name] = attributes
    warm = _filter_valid_records(
        valid, min_canary_count=50, partition_role="warm", warm_fraction=0.2, partition_seed=20260921
    )
    selected = []
    counts = {}
    for light in ("daytime", "night"):
        for scene in ("city street", "highway"):
            eligible = [
                v[0] for v in warm if metadata[v[0].name] == (light, "clear", scene) and v[0].stem not in excluded
            ]
            ordered = sorted(
                eligible, key=lambda p: hashlib.sha256(("health-reference:20261013:" + p.name).encode()).digest()
            )
            if len(ordered) < 64:
                raise ValueError(f"Insufficient warm images: {light}/{scene}/{len(ordered)}")
            counts[f"{light}/{scene}"] = len(ordered)
            selected.extend(
                {
                    "path": str(p),
                    "sha256": digest(p),
                    "cell": [light, scene],
                    "label_path": str(data / "labels/train" / (p.stem + ".json")),
                }
                for p in ordered[:64]
            )
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "reference.json",
        {
            "images": selected,
            "candidate_counts": counts,
            "warm_pool_ids_sha256": hashlib.sha256("\n".join(sorted(v[0].name for v in warm)).encode()).hexdigest(),
            "warm_count": len(warm),
            "metadata_enrollment": True,
            "runtime_labels_used": False,
            "warm_partition_seed": 20260921,
            "warm_fraction": 0.2,
            "query_support_and_stream_path_overlap": 0,
        },
    )
    print(json.dumps({"reference_images": len(selected), "candidate_counts": counts}))


if __name__ == "__main__":
    main()
