"""Freeze fresh clip/keyframe membership and acquire the factorial pilot inputs."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import time
import zipfile
import zlib
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cafl4ds.artifacts import atomic_json
from scripts.prepare_bdd_temporal_archive import SplitHTTPFile
from scripts.prepare_single_image_transition import decode96
from scripts.run_short_buffer_experiment import digest
from scripts.run_single_image_forecast import local_image
from scripts.run_single_image_signal import read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/single_image_filter_factorial.yaml"
PROTOCOL = ROOT / "docs/project-plan/single-image-filter-factorial-frozen.md"
POOL = ROOT / "outputs/scheduling/20261013-intake/selection.pool.json"
INDEX = ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json"
SIGNAL = ROOT / "outputs/single-image-signal/20261021-followup"


def rank(namespace, row):
    return hashlib.sha256(f"{namespace}:{row['clip_id']}".encode()).digest()


def select(out, settings):
    pool = read(POOL)
    original = read(ROOT / "outputs/scheduling/20261013-intake/selection.json")
    assert digest(POOL) == original["eligible_pool_sha256"]
    inventory = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "outputs"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    texts = [inventory]
    files = sorted((ROOT / ".cache").glob("*known-videos.txt")) + [
        SIGNAL / "input_lock.json",
        ROOT / "outputs/single-image-transition/20261021-new-source/input_lock.json",
        ROOT / "outputs/temporal-state-opportunity/20261011-development-v1/experiment/reader-allocation.json",
        ROOT / "outputs/health-calibration/20261013-gpu/reference.json",
    ]
    texts += [p.read_text(encoding="utf-8") for p in files]
    known = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{8}", "\n".join(texts)))
    candidates = {}
    namespace = settings["selection_namespace"]
    for light in ("daytime", "night"):
        eligible = [r for r in pool[light + "/highway"] if r["clip_id"] not in known]
        candidates[light] = sorted(eligible, key=lambda r: rank(namespace, r))[: 30 + settings["reserves_per_light"]]
        assert len(candidates[light]) == 36
    excluded = known | {r["clip_id"] for rows in candidates.values() for r in rows}
    query = []
    for light in ("daytime", "night"):
        for scene in ("city street", "highway"):
            eligible = [r for r in pool[light + "/" + scene] if r["clip_id"] not in excluded]
            chosen = sorted(eligible, key=lambda r: rank(namespace + "-reader", r))[:150]
            assert len(chosen) == 150
            for i, row in enumerate(chosen):
                path = local_image(row["keyframe_path"])
                query.append(
                    {
                        **row,
                        "role": "development" if i < 50 else "evaluation",
                        "path": str(path),
                        "sha256": digest(path),
                        "label": ("city street", "highway").index(scene),
                        "domain": int(light == "night"),
                    }
                )
    selection = dict(
        excluded_clip_ids=sorted(known),
        candidates=candidates,
        query=query,
        inventory_sha256=hashlib.sha256(inventory.encode()).hexdigest(),
        hashes={str(p): digest(p) for p in [CONFIG, PROTOCOL, Path(__file__), POOL, INDEX, *files]},
        outcomes_inspected=False,
        warm_partition_seed=original["warm_partition_seed"],
        warm_fraction=original["warm_fraction"],
    )
    atomic_json(out / "selection.json", selection)
    return selection


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "intake.json").exists(), "Preserve completed intake"
    settings = OmegaConf.to_container(OmegaConf.load(CONFIG))
    selection = read(out / "selection.json") if (out / "selection.json").exists() else select(out, settings)
    for path, expected in selection["hashes"].items():
        assert digest(Path(path)) == expected, path
    original = torch.load(SIGNAL / "panel.pt", weights_only=True)
    pixel_sets = []
    for role in ("development", "evaluation"):
        rows = [r for r in selection["query"] if r["role"] == role]
        arrays = []
        for row in rows:
            assert digest(Path(row["path"])) == row["sha256"]
            with Image.open(row["path"]) as image:
                arrays.append(np.array(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)))
        pixels = np.stack(arrays)
        pixel_sets.append({hashlib.sha256(x.tobytes()).hexdigest() for x in pixels})
        panel = dict(
            support_images=original["support_images"],
            support_labels=original["support_labels"],
            query_images=torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255,
            query_labels=torch.tensor([r["label"] for r in rows]),
            query_domains=torch.tensor([r["domain"] for r in rows]),
        )
        torch.save(panel, out / f"{role}-panel.pt")
    assert not pixel_sets[0] & pixel_sets[1]
    index = read(INDEX)
    previous = (
        read(out / "progress.json") if (out / "progress.json").exists() else dict(clips=[], failures=[], received=0)
    )
    remote = SplitHTTPFile(index["parts"], index["revision"], settings["max_download_mib"] * 1024**2)
    remote.budget -= previous["received"]
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    locked = {r["path"]: r for r in index["entries"]}
    records, failures = previous["clips"], previous["failures"]
    attempted = {r["clip_id"] for r in records + failures}
    started = time.perf_counter()
    with zipfile.ZipFile(remote) as archive:
        for light in ("daytime", "night"):
            for row in selection["candidates"][light]:
                if sum(r["keyframe_attributes"]["timeofday"] == light for r in records) == 30:
                    break
                identifier = row["clip_id"]
                if identifier in attempted:
                    continue
                name = f"bdd100k/videos/100k/train/{identifier}.mov"
                entry, expected = archive.getinfo(name), locked[name]
                assert (entry.file_size, entry.compress_size, entry.CRC, entry.header_offset) == (
                    expected["bytes"],
                    expected["compressed_bytes"],
                    expected["crc32"],
                    expected["offset"],
                )
                assert entry.file_size <= 128 * 1024**2
                video = out / "videos" / f"{identifier}.mov"
                video.parent.mkdir(exist_ok=True)
                if not video.exists():
                    for attempt in range(3):
                        try:
                            data = archive.read(entry)
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                    video.write_bytes(data)
                data = video.read_bytes()
                assert len(data) == entry.file_size and zlib.crc32(data) == entry.CRC
                directory = out / "frames" / identifier
                try:
                    pixels = (
                        read(directory / "manifest.json")
                        if (directory / "manifest.json").exists()
                        else decode96(video, directory, ffmpeg, settings["frames_per_clip"])
                    )
                    assert digest(Path(pixels["frames_path"])) == pixels["frames_sha256"]
                    records.append(
                        dict(**row, **pixels, video_path=str(video), video_sha256=digest(video), crc32=entry.CRC)
                    )
                except ValueError as error:
                    failures.append(dict(clip_id=identifier, reason=repr(error), before_ssl=True))
                    atomic_json(directory / "failure.json", failures[-1])
                attempted.add(identifier)
                atomic_json(
                    out / "progress.json",
                    dict(clips=records, failures=failures, received=previous["received"] + remote.received),
                )
                print(
                    f"Intake {len(records)}/60 clips, {remote.received / 1024**2:.1f} MiB, {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    assert len(records) == 60
    pools = {
        light: [r for r in records if r["keyframe_attributes"]["timeofday"] == light] for light in ("daytime", "night")
    }
    offsets, histories = dict(daytime=0, night=0), {}
    for i, pattern in enumerate(settings["patterns"]):
        clips = []
        for symbol in pattern:
            light = "daytime" if symbol == "D" else "night"
            clips.append(pools[light][offsets[light]])
            offsets[light] += 1
        histories[f"fresh_{i:02d}"] = dict(
            pattern=pattern, clip_ids=[r["clip_id"] for r in clips], paths=[r["frames_path"] for r in clips]
        )
    assert not set(r["clip_id"] for r in records) & set(selection["excluded_clip_ids"])
    atomic_json(
        out / "intake.json",
        dict(
            complete=True,
            clips=records,
            histories=histories,
            failures=failures,
            received=previous["received"] + remote.received,
            selection_sha256=digest(out / "selection.json"),
            ffmpeg_sha256=digest(ffmpeg),
            development_evaluation_pixel_overlap=0,
            source_authenticity="Pinned community mirror CRC/SHA; recording-session independence not established",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
