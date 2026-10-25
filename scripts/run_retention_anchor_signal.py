"""Measure one-step anchor risk on cached, audited train/hold parents."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.anchor_signal_statistics import report
from cafl4ds.artifacts import atomic_json
from cafl4ds.retention_anchor import private_loss, reservoir_indices
from cafl4ds.short_buffer import capture, restore, synchronize
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/budget-reservation/20261013-gpu"


def read(path: Path) -> Any:
    """Read an existing artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def gather(stream: dict[str, Any], clips: dict[str, Any], indices: list[int]) -> torch.Tensor:
    """Load only requested image slots, avoiding whole-stream float caches."""
    identifiers = [c for group in stream["periods"] for c in group]
    arrays = {
        k: np.load(clips[identifiers[k]]["frames_path"], mmap_mode="r", allow_pickle=False)
        for k in {i // 512 for i in indices}
    }
    images = np.stack([arrays[i // 512][i % 512] for i in indices])
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255


def setup() -> None:
    """Use the established deterministic GPU settings."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load(out: Path) -> tuple[Any, Any, Any, Any, Any]:
    """Verify inherited evidence and freeze source and outcome inputs."""
    previous = read(PRIOR / "input_lock.json")
    for name in (
        "verification.json",
        "independent-verification.json",
        "parent-verification.json",
        "statistics-verification.json",
    ):
        if not read(PRIOR / name)["passed"]:
            raise ValueError("Previous audit failed")
    hashes = dict(previous["hashes"])
    rows = read(PRIOR / "collection.json")["records"]
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    paths = [
        Path(__file__),
        ROOT / "scripts/verify_retention_anchor_signal.py",
        ROOT / "cafl4ds/retention_anchor.py",
        ROOT / "cafl4ds/anchor_signal_statistics.py",
        ROOT / "docs/project-plan/slow-edge-retention-anchor-frozen.md",
        PRIOR / "collection.json",
    ]
    paths += [
        PRIOR / n
        for n in (
            "verification.json",
            "independent-verification.json",
            "parent-verification.json",
            "statistics-verification.json",
        )
    ]
    parents = [r for r in rows if r["arm"] == "spend"]
    for row in parents:
        paths.append(PRIOR / "parents" / row["stream"] / row["checkpoint"] / f"seed_{row['seed']}/state.pt")
        paths.append(PRIOR / row["directory"] / "training.json")
    for row in rows:
        if row["arm"] in ("spend", "hold"):
            paths += [PRIOR / row[h] / "predictions.npz" for h in ("immediate", "terminal")]
    for path in paths:
        hashes[str(path)] = digest(path)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Locked evidence changed")
    lock = {
        "hashes": hashes,
        "config": previous["config"],
        "streams": intake["streams"],
        "role": "retrospective_reused_development",
    }
    atomic_json(out / "input_lock.json", lock)
    for path in paths:
        if path.suffix in (".py", ".md"):
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return OmegaConf.create(previous["config"]), intake, parents, rows, lock


def measure(
    initial: Any, config: Any, parent: dict[str, Any], anchors: torch.Tensor, incoming: torch.Tensor, mask_seed: int
) -> dict[str, Any]:
    """Pay for the preview and prove all private readings preserve complete state."""
    device = next(initial.parameters()).device
    synchronize(device)
    tick = time.perf_counter()
    model, optimizer = restore(initial, config.optim, parent)
    synchronize(device)
    clone_seconds = time.perf_counter() - tick
    marker = fingerprint(capture(model, optimizer))
    tick = time.perf_counter()
    incoming_loss = private_loss(model, incoming, None)
    synchronize(device)
    incoming_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    before = private_loss(model, anchors, mask_seed)
    synchronize(device)
    anchor_before_seconds = time.perf_counter() - tick
    if fingerprint(capture(model, optimizer)) != marker:
        raise ValueError("Pre-action readings changed complete parent state")
    tick = time.perf_counter()
    outcome = optimize_current(model, optimizer, incoming.to(device), True)
    synchronize(device)
    update_seconds = time.perf_counter() - tick
    virtual_marker = fingerprint(capture(model, optimizer))
    tick = time.perf_counter()
    after = private_loss(model, anchors, mask_seed)
    synchronize(device)
    anchor_after_seconds = time.perf_counter() - tick
    if fingerprint(capture(model, optimizer)) != virtual_marker or abs(incoming_loss - outcome[0]) > 1e-7:
        raise ValueError("Post-action measurement or incoming-view agreement failed")
    return {
        "incoming_loss": incoming_loss,
        "anchor_before": before,
        "anchor_after": after,
        "relative_anchor_change": (after - before) / max(before, 1e-12),
        "preview_loss_gradient": list(outcome),
        "parent_fingerprint": marker,
        "virtual_fingerprint": virtual_marker,
        "clone_seconds": clone_seconds,
        "incoming_loss_seconds": incoming_seconds,
        "anchor_before_seconds": anchor_before_seconds,
        "anchor_after_seconds": anchor_after_seconds,
        "preview_update_seconds": update_seconds,
        "charged_preview_seconds": clone_seconds + anchor_before_seconds + update_seconds + anchor_after_seconds,
        "comparison_forward_seconds": incoming_seconds,
    }


def points_from(observations: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute source means from predictions; outcome labels never enter signals."""
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    outcomes = {(r["stream"], r["seed"], r["checkpoint"], r["arm"]): r for r in rows if r["arm"] in ("spend", "hold")}
    points = []
    for stream in sorted({o["stream"] for o in observations}):
        for checkpoint in ("before_shift", "stable_second", "before_return"):
            selected = [o for o in observations if o["stream"] == stream and o["checkpoint"] == checkpoint]
            point = {
                "stream": stream,
                "checkpoint": checkpoint,
                "updates": selected[0]["point"] // 64,
                **{f: statistics.fmean(o[f] for o in selected) for f in ("incoming_loss", "relative_anchor_change")},
            }
            for horizon in ("immediate", "terminal"):
                effects = {}
                for metric, mask in (("balanced", np.ones(400, bool)), ("A", domains == 0), ("B", domains == 1)):
                    changes = []
                    for o in selected:
                        scores = {}
                        for arm in ("spend", "hold"):
                            row = outcomes[stream, o["seed"], checkpoint, arm]
                            with np.load(PRIOR / row[horizon] / "predictions.npz", allow_pickle=False) as p:
                                scores[arm] = statistics.fmean(
                                    100 * float((p[f"{d}_finetune"][mask] == labels[mask]).mean())
                                    for d in (51011, 51017, 51023)
                                )
                        changes.append(scores["spend"] - scores["hold"])
                    effects[metric] = statistics.fmean(changes)
                point[f"{horizon}_damage"] = -min(effects["A"], effects["B"])
                point[f"{horizon}_balanced_effect"] = effects["balanced"]
                point[f"{horizon}_domain_effects"] = effects
            points.append(point)
    return points


def main() -> None:
    """Collect once, replay all virtual updates and publish the fixed assay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable run; no automatic retry")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 512 * 1024**2:
        raise ValueError("Storage reserve insufficient")
    start = time.perf_counter()

    def progress(**fields: Any) -> None:
        status = {**fields, "elapsed_seconds": time.perf_counter() - start}
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    setup()
    progress(stage="locking")
    config, intake, parents, rows, lock = load(out)
    torch.manual_seed(11)
    initial = _method(config).cuda()
    clips = {c["clip_id"]: c for c in intake["clips"]}
    streams = {s["name"]: s for s in intake["streams"]}
    observations = []
    fields = (
        "incoming_loss",
        "anchor_before",
        "anchor_after",
        "relative_anchor_change",
        "preview_loss_gradient",
        "parent_fingerprint",
        "virtual_fingerprint",
    )
    for row in parents:
        if shutil.disk_usage(out).free < 256 * 1024**2:
            raise ValueError("Storage reserve threatened")
        point, seed = row["point"], row["seed"]
        parent = torch.load(
            PRIOR / "parents" / row["stream"] / row["checkpoint"] / f"seed_{seed}/state.pt",
            weights_only=True,
            map_location="cpu",
        )
        tick = time.perf_counter()
        indices = reservoir_indices(point, 64, seed + 1117)
        reservoir_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        anchors = gather(streams[row["stream"]], clips, indices)
        incoming = gather(streams[row["stream"]], clips, list(range(point, point + 32)))
        acquisition_seconds = time.perf_counter() - tick
        mask_seed = seed + 81203 + point
        if not observations:
            preflight = [measure(initial, config, parent, anchors, incoming, mask_seed) for _ in range(2)]
            if any(preflight[0][f] != preflight[1][f] for f in fields):
                raise ValueError("Nondeterministic preflight")
            atomic_json(out / "preflight.json", {"passed": True, "discarded_virtual_updates": 2})
        measured = measure(initial, config, parent, anchors, incoming, mask_seed)
        repeated = measure(initial, config, parent, anchors, incoming, mask_seed)
        if any(measured[f] != repeated[f] for f in fields):
            raise ValueError("Independent virtual-step replay differs")
        actual = read(PRIOR / row["directory"] / "training.json")["records"][0]
        if measured["preview_loss_gradient"] != actual["loss_gradient"]:
            raise ValueError("Preview differs from audited real first update")
        observation = {
            "stream": row["stream"],
            "checkpoint": row["checkpoint"],
            "seed": seed,
            "point": point,
            "anchor_indices": indices,
            "incoming_indices": list(range(point, point + 32)),
            "mask_seed": mask_seed,
            "reservoir_construction_seconds": reservoir_seconds,
            "retrospective_acquisition_seconds": acquisition_seconds,
            **measured,
        }
        observations.append(observation)
        atomic_json(out / "observations.json", {"complete": len(observations) == 54, "records": observations})
        progress(stage="collection_and_replay", completed=len(observations), total=54)
    summary = report(points_from(observations, rows))
    summary["costs"] = {
        key: statistics.fmean(o[key] for o in observations)
        for key in (
            "charged_preview_seconds",
            "preview_update_seconds",
            "clone_seconds",
            "incoming_loss_seconds",
            "anchor_before_seconds",
            "anchor_after_seconds",
            "retrospective_acquisition_seconds",
            "reservoir_construction_seconds",
        )
    }
    summary["costs"]["preview_over_16_update_estimate"] = summary["costs"]["charged_preview_seconds"] / (
        16 * summary["costs"]["preview_update_seconds"]
    )
    atomic_json(out / "summary.json", summary)
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Final input hashes differ")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 64 * 1024**2:
        raise ValueError("Storage output cap exceeded")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "virtual_previews": 54,
            "replayed_previews": 54,
            "actual_first_step_matches": 54,
            "new_downstream_fits": 0,
            "output_mib": size / 1024**2,
            "free_gib": shutil.disk_usage(out).free / 1024**3,
            "locked_inputs": len(lock["hashes"]),
            "parent_observer_invariance": True,
        },
    )
    progress(stage="complete", signal_qualified=summary["signal_qualified"])


if __name__ == "__main__":
    main()
