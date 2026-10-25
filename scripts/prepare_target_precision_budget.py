"""Expand held-out queries and record a pilot-based, fixed measurement budget."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import _driving_attributes, _image_filename, _iter_bdd_records
from cafl4ds.functional_retention import plain
from cafl4ds.target_alignment_measurement import resampling_counts, support_draws
from cafl4ds.target_precision_budget import planned_seeds
from scripts.run_collapse_risk_regression import digest, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/target_precision_budget.yaml"


def main():
    config = OmegaConf.load(CONFIG)
    source, out = ROOT / config.source, ROOT / config.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "data-role-manifest.json").exists(), "Prepared inputs are immutable"
    assert read(source / "independent-verification.json")["passed"]
    preflight = read(ROOT / config.preflight / "summary.json")
    assert preflight["passed"]
    old = read(source / "data-role-manifest.json")
    excluded = set(old["excluded_ids"])
    supports = [r for r in old["records"] if r["role"] == "support_pool"]
    pilot = [r for r in old["records"] if r["role"] == "fresh_query"]
    support_ids, pilot_ids = {r["clip_id"] for r in supports}, {r["clip_id"] for r in pilot}
    additional = []
    base = Path("C:/CAFL-E/datasets/bdd100k")
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
        if not path.exists() or path.stem in excluded | support_ids | pilot_ids:
            continue
        additional.append(
            {
                "clip_id": path.stem,
                "path": str(path),
                "label_path": str(base / "labels/val" / (path.stem + ".json")),
                "label": int(attrs[2] == "highway"),
                "domain": int(attrs[0] == "night"),
                "role": "additional_query",
            }
        )
    additional.sort(
        key=lambda r: hashlib.sha256(("target-precision-budget:20261015:" + r["clip_id"]).encode()).digest()
    )
    records = supports + pilot + additional
    cells = {
        f"{d}/{c}": sum(r["domain"] == d and r["label"] == c for r in pilot + additional)
        for d in (0, 1)
        for c in (0, 1)
    }
    assert cells == {"0/0": 733, "0/1": 314, "1/0": 1933, "1/1": 683}, cells
    print({"query_cells": cells, "additional_queries": len(additional)}, flush=True)
    old_images = torch.load(source / "images.pt", weights_only=True)
    with np.load(source / "data.npz") as data:
        old_arrays = {k: data[k] for k in data.files}
    rows = read(source / "collection.json")["records"]
    responses = []
    for row in rows:
        directory = source / "runs" / row["source"]
        probabilities = [
            np.load(directory / f"at_{step}/predictions.npz")["probabilities"][:, :1024]
            for step in (row["age"], row["age"] + 32)
        ]
        y = old_arrays["labels"][old_arrays["query"]]
        responses.append(100 * ((probabilities[1].argmax(-1) == y).astype(float) - (probabilities[0].argmax(-1) == y)))
    projection = instantiate(config.planner).project(
        responses, old_arrays["labels"][old_arrays["query"]], old_arrays["domains"][old_arrays["query"]], cells
    )
    assert projection["selected_panels"] == config.final_panels, "Budget selection differs; stop before new outcomes"
    projection["input_hashes"] = {
        str(p): digest(p)
        for p in (CONFIG, ROOT / "cafl4ds/target_precision_budget.py", source / "collection.json", source / "data.npz")
    }
    atomic_json(out / "planning.json", plain(projection))
    print({"selected_panels": projection["selected_panels"], "scope": projection["scope"]}, flush=True)
    pixels, hashes = [], {}
    for row in additional:
        with Image.open(row["path"]) as opened:
            pixels.append(np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
    extra = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).float() / 255
    images = torch.cat([old_images[:1224], extra, old_images[old_arrays["legacy_query"]]])
    labels = np.concatenate([np.array([r["label"] for r in records]), old_arrays["labels"][old_arrays["legacy_query"]]])
    domains = np.concatenate(
        [np.array([r["domain"] for r in records]), old_arrays["domains"][old_arrays["legacy_query"]]]
    )
    query = np.arange(200, len(records))
    legacy = np.arange(len(records), len(records) + 400)
    seeds = planned_seeds(
        read(source / "input_lock.json")["config"]["support_seeds"], config.final_panels, config.new_seed_start
    )
    draws = support_draws(labels, domains, np.arange(200), seeds)
    for seed in seeds[:12]:
        assert np.array_equal(draws[seed], old_arrays[f"support_{seed}"])
    ids = {r["clip_id"] for r in records}
    assert len(ids) == len(records) and not ids.intersection(excluded)
    image_hashes = [hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in images[: len(records)]]
    assert len(set(image_hashes)) == len(image_hashes)
    legacy_hashes = {hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in old_images}
    assert not {hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in extra}.intersection(legacy_hashes)
    torch.save(images, out / "images.pt")
    np.savez_compressed(
        out / "data.npz",
        labels=labels,
        domains=domains,
        query=query,
        pilot_query=np.arange(200, 1224),
        additional_query=np.arange(1224, len(records)),
        legacy_query=legacy,
        support_pool=np.arange(200),
        **{f"support_{s}": v for s, v in draws.items()},
    )
    for total in (config.interim_panels, config.final_panels):
        panel, counts = resampling_counts(
            labels[query], domains[query], total, config.bootstrap_repetitions, config.bootstrap_seed + total
        )
        np.savez_compressed(out / f"resamples_{total}.npz", panel=panel, query=counts)
    for row in records:
        for key in ("path", "label_path"):
            hashes[row[key]] = digest(row[key])
    for p in (
        CONFIG,
        Path(__file__),
        ROOT / "cafl4ds/target_precision_budget.py",
        ROOT / "docs/project-plan/target-precision-budget-frozen.md",
        ROOT / config.preflight / "summary.json",
        source / "data-role-manifest.json",
        out / "images.pt",
        out / "data.npz",
        out / "resamples_64.npz",
        out / "resamples_256.npz",
        out / "planning.json",
    ):
        hashes[str(p)] = digest(p)
    atomic_json(
        out / "data-role-manifest.json",
        plain(
            {
                "records": records,
                "excluded_ids": sorted(excluded),
                "query_cells": cells,
                "support_seeds": seeds,
                "hashes": hashes,
                "source": str(source),
                "query_population": "expanded_same_eligible_validation_population; pilot queries reused",
                "new_additional_queries": len(additional),
                "support_pool_changed": False,
                "capture_sessions_known": False,
                "historical_use_inventory_complete": False,
            }
        ),
    )
    print({"prepared_images": len(images), "query_count": len(query), "support_seeds": len(seeds)}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
