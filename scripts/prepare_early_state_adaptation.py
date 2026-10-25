"""Reserve unused identities and prepare the predeclared early-state population."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from scripts.prepare_natural_health_rate import digest, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/early_state_adaptation.yaml"
PROTOCOL = ROOT / "docs/project-plan/early-state-adaptation-frozen.md"
IDENTITY = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}$")


def write(path, value):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


def identities(value):
    """Read identities in nested manifest records, paths and explicit ID lists."""
    if isinstance(value, str):
        candidate = Path(value.replace("\\", "/")).stem
        return {candidate} if IDENTITY.fullmatch(candidate) else set()
    if isinstance(value, dict):
        return set().union(*(identities(v) for item in value.items() for v in item)) if value else set()
    if isinstance(value, list):
        return set().union(*(identities(v) for v in value)) if value else set()
    return set()


def context_config(study, context):
    result = OmegaConf.create(OmegaConf.to_container(study, resolve=True))
    result.intake = f"{study.output}/context{context}-intake"
    result.frames = f"{study.output}/context{context}-frames"
    return result


def select(study):
    out = ROOT / study.output
    assert not (out / "reservation.json").exists(), "Preserve the fixed reservation"
    excluded, manifests = set(), {}
    names = {
        "intake.json",
        "selection.json",
        "download_manifest.json",
        "allocation.json",
        "prepared.json",
        "plan.json",
        "data-role-manifest.json",
        "collection.json",
    }
    for path in sorted((ROOT / "outputs").rglob("*.json")):
        if out in path.parents or "source_snapshot" in path.parts:
            continue
        if path.name not in names and not any(s in path.name for s in ("allocation", "prepared")):
            continue
        data = read(path)
        found = identities(data)
        if found:
            excluded.update(found)
            manifests[str(path.resolve())] = digest(path)
    prior = read(ROOT / study.prior / "input_lock.json")
    excluded.update(identities(read(ROOT / study.selection_source)))
    for bank in ("monitor_reference", "audit_reference"):
        excluded.update(identities(prior[bank]))
    index = read(ROOT / study.archive_index)
    # Look up positions in the actual train inventory; test archive indices never
    # determine the next train position. This includes previously reserved IDs.
    # Warm IDs occur across the whole archive. Only actually selected/reserved
    # train members should advance the cursor; explicit exclusions still apply.
    actual_positions = []
    for p in manifests:

        def visit(v):
            if isinstance(v, dict):
                if "archive_position" in v and "/train/" in v.get("path", ""):
                    actual_positions.append(int(v["archive_position"]))
                for x in v.values():
                    visit(x)
            elif isinstance(v, list):
                for x in v:
                    visit(x)

        visit(read(p))
    assert actual_positions, "Need an authenticated previous train cursor"
    start = max(actual_positions)
    candidates = [
        (n, e)
        for n, e in enumerate(index["entries"])
        if n > start and "/train/" in e["path"] and Path(e["path"]).stem not in excluded
    ]
    selections = []
    for i, context in enumerate(study.contexts):
        clips = [
            {**e, "clip_id": Path(e["path"]).stem, "archive_position": n}
            for n, e in candidates[i * study.max_clips : (i + 1) * study.max_clips]
        ]
        assert len(clips) == study.max_clips
        assert max(c["bytes"] for c in clips) <= 128 * 2**20
        assert sum(c["compressed_bytes"] for c in clips) + 64 * 2**20 < study.network_budget_gib * 2**30
        selections.append({"context": context, "clips": clips})
    future_ids = {c["clip_id"] for s in selections for c in s["clips"]}
    assert len(future_ids) == len(study.contexts) * study.max_clips
    assert not excluded & future_ids
    # Fix candidate order before reading query pixels or learning outcomes.
    cells = {(d, c): [] for d in (0, 1) for c in (0, 1)}
    metadata = read(ROOT / study.metadata_source)["records"]
    for r in metadata:
        a = r["attributes"]
        if r["clip_id"] in excluded | future_ids or not isinstance(a, dict):
            continue
        if (
            a.get("weather") == "clear"
            and a.get("timeofday") in ("daytime", "night")
            and a.get("scene") in ("city street", "highway")
        ):
            cells[int(a["timeofday"] == "night"), int(a["scene"] == "highway")].append(r["clip_id"])
    for values in cells.values():
        values.sort(key=lambda c: hashlib.sha256(f"early-state-adaptation:query:20261019:{c}".encode()).digest())
        assert len(values) >= study.queries_per_cell, {str(k): len(v) for k, v in cells.items()}
    write(
        out / "reservation.json",
        {
            "contexts": selections,
            "excluded_ids": sorted(excluded),
            "prior_manifests": manifests,
            "start_after_train_archive_position": start,
            "query_candidates": {f"{d}/{c}": v for (d, c), v in cells.items()},
            "before_new_SSL_and_readout": True,
            "created_unix": time.time(),
        },
    )
    for selection in selections:
        config = context_config(study, selection["context"])
        target = ROOT / config.intake
        write(target / "selection.json", {"clips": selection["clips"], "before_learner_outcomes": True})
        paths = [
            CONFIG,
            PROTOCOL,
            Path(__file__),
            ROOT / "scripts/acquire_early_state_context.py",
            ROOT / "scripts/prepare_natural_health_rate.py",
            ROOT / "scripts/prepare_bdd_temporal_archive.py",
            ROOT / study.archive_index,
            ROOT / study.metadata_source,
            out / "reservation.json",
            target / "selection.json",
        ]
        write(
            target / "preparation-lock.json",
            {"hashes": {str(p.resolve()): digest(p) for p in paths}, "created_unix": time.time()},
        )
    print(
        {
            "reserved_contexts": len(selections),
            "query_candidates": {str(k): len(v) for k, v in cells.items()},
            "train_cursor": start,
        },
        flush=True,
    )


def queries(study):
    import torch
    from scripts.run_long_memory_replication import lossless_compress

    out, source = ROOT / study.evaluation, ROOT / study.support_source
    assert not (out / "data-role-manifest.json").exists(), "Preserve prepared queries"
    reservation = read(ROOT / study.output / "reservation.json")
    old_manifest = read(source / "data-role-manifest.json")
    old_images = torch.load(source / "images.pt", weights_only=True, map_location="cpu")
    with np.load(source / "data.npz") as z:
        old = {k: z[k] for k in z.files}
    hashes = {hashlib.sha256(im.numpy().tobytes()).hexdigest() for im in old_images}
    base = Path("C:/CAFL-E/datasets/bdd100k")
    records, pixels, rejected = [], [], []
    for cell, candidates in reservation["query_candidates"].items():
        d, c = map(int, cell.split("/"))
        accepted = 0
        for clip in candidates:
            path, label = base / "images/100k/train" / f"{clip}.jpg", base / "labels/train" / f"{clip}.json"
            attrs = read(label)["attributes"]
            assert attrs["weather"] == "clear" and attrs["timeofday"] == ("daytime", "night")[d]
            assert attrs["scene"] == ("city street", "highway")[c]
            with Image.open(path) as opened:
                raw = np.array(opened.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
            image = torch.from_numpy(raw).permute(2, 0, 1).float() / 255
            h = hashlib.sha256(image.numpy().tobytes()).hexdigest()
            if h in hashes:
                rejected.append(clip)
                continue
            hashes.add(h)
            pixels.append(image)
            records.append(
                {
                    "clip_id": clip,
                    "path": str(path),
                    "label_path": str(label),
                    "domain": d,
                    "label": c,
                    "role": "fresh_query",
                    "rgb64_float_sha256": h,
                    "image_sha256": digest(path),
                    "annotation_sha256": digest(label),
                }
            )
            accepted += 1
            if accepted == study.queries_per_cell:
                break
        assert accepted == study.queries_per_cell
    assert len({r["clip_id"] for r in records}) == 2048
    # Inference layout: pilot1024 + additional1024 + legacy400.
    images = torch.cat([old_images[:200], torch.stack(pixels), old_images[old["legacy_query"]]])
    supports = [r for r in old_manifest["records"] if r["role"] == "support_pool"]
    assert len(supports) == 200 and len(images) == 2648
    labels = np.r_[old["labels"][:200], [r["label"] for r in records], old["labels"][old["legacy_query"]]]
    domains = np.r_[old["domains"][:200], [r["domain"] for r in records], old["domains"][old["legacy_query"]]]
    seeds = old_manifest["support_seeds"][:12]
    out.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(out).free > study.minimum_free_mib * 2**20
    torch.save(images, out / "images.pt")
    lossless_compress(out / "images.pt")
    np.savez_compressed(
        out / "data.npz",
        labels=labels,
        domains=domains,
        query=np.arange(200, 2248),
        pilot_query=np.arange(200, 1224),
        additional_query=np.arange(1224, 2248),
        legacy_query=np.arange(2248, 2648),
        support_pool=np.arange(200),
        **{f"support_{s}": old[f"support_{s}"] for s in seeds},
    )
    write(
        out / "data-role-manifest.json",
        {
            "records": supports + records,
            "support_seeds": seeds,
            "exact_pixel_duplicate_rejections": rejected,
            "query_cells": {k: 512 for k in reservation["query_candidates"]},
            "capture_sessions_known": False,
            "source": str(source),
            "population": "unused_local_train_clip_identities",
            "hashes": {str(p.resolve()): digest(p) for p in (out / "images.pt", out / "data.npz")},
        },
    )
    print({"prepared_queries": 2048, "images": len(images), "pixel_duplicate_rejections": len(rejected)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "queries", "acquire"), required=True)
    parser.add_argument("--context", type=int)
    args = parser.parse_args()
    study = OmegaConf.load(CONFIG)
    if args.stage == "acquire":
        from scripts.acquire_early_state_context import acquire

        for context in study.contexts:
            if args.context is None or context == args.context:
                acquire(context_config(study, context))
    else:
        {"select": select, "queries": queries}[args.stage](study)


if __name__ == "__main__":
    main()
