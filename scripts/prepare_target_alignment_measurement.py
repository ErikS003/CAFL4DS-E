"""Inventory documented evaluation use and reserve new task identities before fitting."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import _driving_attributes, _image_filename, _iter_bdd_records
from cafl4ds.functional_retention import plain
from cafl4ds.target_alignment_measurement import resampling_counts, support_draws
from scripts.run_collapse_risk_regression import digest, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_alignment_measurement.yaml"
OLD = ROOT / "outputs/functional-retention/20261014-cached"
OLD_IMAGES = ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt"


def inventory(out):
    """Conservative documented-ID exclusion, not a claim about every historical read."""
    command = ["rg", "--files", "outputs"]
    for pattern in (
        "data-role-manifest.json",
        "data_lock.json",
        "input_lock.json",
        "evaluation*manifest.json",
        "**/corpus/manifest.json",
        "!**/source_snapshot/**",
    ):
        command += ["-g", pattern]
    paths = sorted(ROOT / p for p in subprocess.check_output(command, cwd=ROOT, text=True).splitlines())
    paths = [p for p in paths if not p.is_relative_to(out)]
    expression = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{8}(?![0-9a-f])")
    used, metadata = set(), {}
    for p in paths:
        text = p.read_text(encoding="utf-8")
        used.update(expression.findall(text))
        metadata[str(p.resolve())] = digest(p)
    print(
        {"stage": "documented_identity_inventory", "metadata_files": len(paths), "excluded_ids": len(used)}, flush=True
    )
    return used, metadata


def prepare():
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "data-role-manifest.json"
    if destination.exists():
        previous = read(destination)
        assert all(digest(p) == h for p, h in previous["hashes"].items())
        return
    used, prior_metadata = inventory(out)
    base = Path("C:/CAFL-E/datasets/bdd100k")
    cells = {(d, c): [] for d in (0, 1) for c in (0, 1)}
    counts = {f"{d}/{c}": 0 for d, c in cells}
    for row in _iter_bdd_records(base / "labels/val"):
        attrs = _driving_attributes(row)
        if (
            not attrs
            or attrs[0] not in ("daytime", "night")
            or attrs[1] != "clear"
            or attrs[2] not in ("city street", "highway")
        ):
            continue
        path = base / "images/100k/val" / _image_filename(row)
        if not path.exists():
            continue
        d, c = int(attrs[0] == "night"), int(attrs[2] == "highway")
        counts[f"{d}/{c}"] += 1
        if path.stem not in used:
            cells[d, c].append(
                {
                    "clip_id": path.stem,
                    "path": str(path),
                    "label_path": str(base / "labels/val" / (path.stem + ".json")),
                    "domain": d,
                    "label": c,
                }
            )
    available = {f"{d}/{c}": len(values) for (d, c), values in cells.items()}
    print({"stage": "pool_inventory", "counts": counts, "available": available}, flush=True)
    support, queries = [], []
    for (d, c), values in cells.items():
        values.sort(key=lambda row: hashlib.sha256(f"{config.data_salt}:{row['clip_id']}".encode()).digest())
        assert len(values) >= config.support_pool_per_cell + config.queries_per_cell
        support += [{**r, "role": "support_pool"} for r in values[: config.support_pool_per_cell]]
        queries += [
            {**r, "role": "fresh_query"}
            for r in values[config.support_pool_per_cell : config.support_pool_per_cell + config.queries_per_cell]
        ]
    records = support + queries
    assert len({r["clip_id"] for r in records}) == len(records)
    pixels, hashes = [], {str(CONFIG): digest(CONFIG), str(Path(__file__)): digest(__file__)}
    hashes[str(ROOT / "docs/project-plan/target-alignment-measurement-frozen.md")] = digest(
        ROOT / "docs/project-plan/target-alignment-measurement-frozen.md"
    )
    for row in records:
        with Image.open(row["path"]) as image:
            pixels.append(np.array(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        for field in ("path", "label_path"):
            hashes[row[field]] = digest(row[field])
    images = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).float() / 255
    labels = np.array([r["label"] for r in records])
    domains = np.array([r["domain"] for r in records])
    pool = np.arange(len(support))
    query = np.arange(len(support), len(records))
    draws = support_draws(labels, domains, pool, config.support_seeds)
    old_images = torch.load(OLD_IMAGES, map_location="cpu", weights_only=True)
    with np.load(OLD / "evaluation_arrays.npz") as old:
        old_query = old["query"][old["labels"][old["query"]] < 2]
        legacy_y, legacy_d = old["labels"][old_query], old["domains"][old_query]
    assert len(old_query) == 400
    legacy = np.arange(len(records), len(records) + len(old_query))
    cached_hashes = {hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in old_images}
    new_hashes = [hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in images]
    assert len(set(new_hashes)) == len(new_hashes) and not set(new_hashes).intersection(cached_hashes)
    images = torch.cat([images, old_images[old_query]])
    labels, domains = np.concatenate([labels, legacy_y]), np.concatenate([domains, legacy_d])
    torch.save(images, out / "images.pt")
    np.savez_compressed(
        out / "data.npz",
        labels=labels,
        domains=domains,
        support_pool=pool,
        query=query,
        legacy_query=legacy,
        **{f"support_{s}": v for s, v in draws.items()},
    )
    panel, qcount = resampling_counts(
        labels[query], domains[query], len(draws), config.bootstrap_repetitions, config.bootstrap_seed
    )
    legacy_panel, legacy_count = resampling_counts(
        labels[legacy], domains[legacy], len(draws), config.bootstrap_repetitions, config.bootstrap_seed + 1
    )
    np.savez_compressed(
        out / "resamples.npz", panel=panel, query=qcount, legacy_panel=legacy_panel, legacy_query=legacy_count
    )
    for p in [OLD_IMAGES, OLD / "evaluation_arrays.npz", out / "images.pt", out / "data.npz", out / "resamples.npz"]:
        hashes[str(p)] = digest(p)
    atomic_json(
        destination,
        plain(
            {
                "records": records,
                "prior_metadata_hashes": prior_metadata,
                "excluded_ids": sorted(used),
                "target_counts": counts,
                "available_after_documented_exclusion": available,
                "hashes": hashes,
                "scope": "new_documented_image_identities_on_reused_training_histories",
                "capture_sessions_known": False,
                "historical_use_inventory_complete": False,
                "old_query_role": "reused_diagnostic_same_fitted_heads",
                "config": OmegaConf.to_container(config, resolve=True),
            }
        ),
    )
    print(
        json.dumps({"prepared_images": len(images), "new_support_pool": len(support), "new_queries": len(queries)}),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
