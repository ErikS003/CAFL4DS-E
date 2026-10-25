"""Reserve all new long-stream blocks before acquisition and learner outcomes."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf

from scripts.natural_health_view import write
from scripts.prepare_natural_health_rate import acquire
from scripts.run_preaction_stopping import ROOT, digest, read

CONFIG = ROOT / "cafl4ds/configs/long_memory_replication.yaml"
PROTOCOL = ROOT / "docs/project-plan/long-memory-replication-frozen.md"


def block_config(study, block):
    result = OmegaConf.create(OmegaConf.to_container(study, resolve=True))
    result.intake = f"{study.study_root}/block{block}-intake"
    result.frames = f"{study.study_root}/block{block}-frames"
    return result


def select(study):
    old = read(ROOT / study.selection_source)
    prior = read(ROOT / study.prior / "input_lock.json")
    excluded = set(old["warm_clip_ids"]) | set(old["monitor_clip_ids"]) | set(old["evaluation_clip_ids"])
    excluded.update(Path(r["path"]).stem for r in prior["monitor_reference"]["images"])
    excluded.update(r["clip_id"] for r in prior["audit_reference"]["images"])
    manifests, positions = {}, []
    for name in ("intake.json", "selection.json", "download_manifest.json"):
        for path in sorted((ROOT / "outputs").rglob(name)):
            data = read(path)
            clips = data.get("clips", []) if isinstance(data, dict) else []
            if isinstance(clips, dict):
                excluded.update(clips)
                clips = list(clips.values())
            found = []
            for clip in clips:
                if isinstance(clip, str):
                    found.append(Path(clip).stem)
                elif isinstance(clip, dict):
                    identity = clip.get("clip_id") or clip.get("id") or Path(clip.get("path", "")).stem
                    if identity:
                        found.append(identity)
                    if "archive_position" in clip:
                        positions.append(int(clip["archive_position"]))
            if found:
                excluded.update(found)
                manifests[str(path)] = digest(path)
    start = max(positions)
    entries = read(ROOT / study.archive_index)["entries"]
    candidates = [
        (n, e)
        for n, e in enumerate(entries)
        if n > start and "/train/" in e["path"] and Path(e["path"]).stem not in excluded
    ]
    for index, block in enumerate(study.blocks):
        config = block_config(study, block)
        target = ROOT / config.intake
        assert not target.exists(), "Preserve old preparation"
        selected = [
            {**e, "clip_id": Path(e["path"]).stem, "archive_position": n}
            for n, e in candidates[index * study.max_clips : (index + 1) * study.max_clips]
        ]
        assert len(selected) == study.max_clips
        assert max(c["bytes"] for c in selected) <= 128 * 2**20
        assert sum(c["compressed_bytes"] for c in selected) + 64 * 2**20 < study.network_budget_gib * 2**30
        target.mkdir(parents=True)
        write(
            target / "prior-exclusions.json",
            {
                "ids": sorted(excluded),
                "manifests": manifests,
                "start_after_archive_position": start,
                "attributes_used_for_new_selection": False,
            },
        )
        write(
            target / "selection.json",
            {
                "clips": selected,
                "before_learner_outcomes": True,
                "ordering": "next eligible pinned archive members; no regime grouping",
            },
        )
        paths = [
            CONFIG,
            PROTOCOL,
            Path(__file__),
            ROOT / "scripts/prepare_natural_health_rate.py",
            ROOT / "scripts/prepare_bdd_temporal_archive.py",
            ROOT / study.archive_index,
            ROOT / study.selection_source,
            target / "prior-exclusions.json",
            target / "selection.json",
        ]
        write(
            target / "preparation-lock.json",
            {"hashes": {str(p): digest(p) for p in paths}, "created_unix": time.time()},
        )
        excluded.update(c["clip_id"] for c in selected)
        print(
            {
                "reserved_block": block,
                "first": selected[0]["archive_position"],
                "last": selected[-1]["archive_position"],
            },
            flush=True,
        )


def compress_frames(config):
    directory = (ROOT / config.frames).resolve()
    assert directory.is_relative_to((ROOT / config.study_root).resolve())
    paths = sorted(directory.rglob("frames.npy"))
    before = {str(p): digest(p) for p in paths}
    if os.name == "nt":
        result = subprocess.run(
            [
                str(Path(os.environ["SystemRoot"]) / "System32/compact.exe"),
                "/C",
                "/EXE:LZX",
                "/I",
                "/Q",
                f"/S:{directory}",
                "frames.npy",
            ],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert all(digest(p) == h for p, h in before.items())
    write(ROOT / config.intake / "compression-verification.json", {"passed": True, "hashes": before})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "acquire", "all"), default="all")
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    study = OmegaConf.load(CONFIG)
    if args.stage in ("select", "all"):
        select(study)
    if args.stage in ("acquire", "all"):
        assert all((ROOT / block_config(study, b).intake / "preparation-lock.json").exists() for b in study.blocks)
        for block in study.blocks:
            if args.block is not None and block != args.block:
                continue
            config = block_config(study, block)
            acquire(config)
            compress_frames(config)


if __name__ == "__main__":
    main()
