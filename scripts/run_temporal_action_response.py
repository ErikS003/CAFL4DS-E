"""Reconstruct exact temporal histories, then collect paired fixed-horizon actions."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.temporal_action_response import restore, run_action, snapshot
from cafl4ds.temporal_anchored_filter import optimize_window
from scripts.decode_bdd_temporal import sha256
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_recurrence_retention import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/content-filter-bdd/20261011-anchored-change-half"
PROTOCOL = ROOT / "docs/project-plan/slow-edge-temporal-action-response-frozen.md"
HISTORIES = ("full", "uniform_half")
OFFSETS = (8, 32, 56, 80)
SEEDS = (5003, 5009, 5011, 5021, 5023)
HORIZON = 8


def main() -> None:  # noqa: C901 - explicit integrity and restoration gates
    """Preserve failures and seal parents before any branch query outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    completed = 0

    def progress(**fields: Any) -> None:  # noqa: ANN401 - JSON progress fields
        atomic_json(
            out / "status.json",
            {
                "started_unix": started,
                "updated_unix": time.time(),
                "completed_branches": completed,
                "total_branches": 80,
                **fields,
            },
        )

    try:
        progress(stage="validating_prior")
        audit_path = PRIOR / "independent_verification.json"
        audit = json.loads(audit_path.read_text())
        if not audit["passed"] or audit["arms_checked"] != 25 or audit["optimizer_updates"] != 2880:
            raise ValueError("anchored comparison is not completely audited")
        old_inputs = json.loads((PRIOR / "input_lock.json").read_text())
        hashes = {
            **old_inputs["hashes"],
            **{str(PRIOR / name): expected for name, expected in audit["artifact_hashes"].items()},
        }
        for name, expected in hashes.items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"prior evidence changed: {name}")
        additional = [
            Path(__file__),
            PROTOCOL,
            audit_path,
            ROOT / "cafl4ds/temporal_action_response.py",
            ROOT / "scripts/report_paired_update_response.py",
            ROOT / "scripts/audit_recurrence_retention.py",
            ROOT / "scripts/report_temporal_action_response.py",
            ROOT / "scripts/run_temporal_action_background.py",
        ]
        hashes.update({str(p): sha256(p) for p in additional})
        old_data = json.loads((PRIOR / "data_lock.json").read_text())
        intake = json.loads(Path(old_data["intake_path"]).read_text())
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        images = (
            torch.from_numpy(
                np.concatenate([np.load(clip["frames_path"], allow_pickle=False) for clip in intake["clips"]])
            )
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )
        if images.shape != (6144, 3, 64, 64):
            raise ValueError("temporal arrivals differ")
        config = OmegaConf.create(old_inputs["config"])
        with np.load(PRIOR / "evaluation_arrays.npz", allow_pickle=False) as arrays:
            evaluation = Evaluation(
                images=torch.load(PRIOR / "evaluation_images.pt", map_location="cpu", weights_only=True),
                labels=torch.from_numpy(arrays["labels"].copy()),
                domains=torch.from_numpy(arrays["domains"].copy()),
                query=torch.from_numpy(arrays["query"].copy()),
                supports={int(draw): torch.tensor(ids) for draw, ids in old_data["evaluation"]["supports"].items()},
                manifest=old_data["evaluation"],
            )
        atomic_json(
            out / "input_lock.json",
            {
                "hashes": hashes,
                "config": old_inputs["config"],
                "prior": str(PRIOR),
                "seeds": SEEDS,
                "histories": HISTORIES,
                "offsets": OFFSETS,
                "horizon": HORIZON,
                "protocol_sha256": sha256(PROTOCOL),
            },
        )
        atomic_json(out / "data_lock.json", old_data)
        for name in hashes:
            source = Path(name)
            if source.suffix in {".py", ".md", ".yaml"} and source.is_relative_to(ROOT):
                target = out / "source_snapshot" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        torch.manual_seed(11)
        initial = _method(config)
        progress(stage="reconstructing_parents")
        parent_hashes = {}
        parent_updates = 0
        for seed in SEEDS:
            for history in HISTORIES:
                directory = out / "parents" / history / f"seed_{seed}"
                directory.mkdir(parents=True)
                torch.manual_seed(seed)
                model = copy.deepcopy(initial)
                optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
                initial_state = snapshot(model, optimizer, 0, 0)
                expected_initial = torch.load(
                    PRIOR / history / f"seed_{seed}" / "initial_state.pt", map_location="cpu", weights_only=False
                )
                if fingerprint(initial_state) != fingerprint(expected_initial):
                    raise ValueError("parent initialization differs from audited history")
                updates = 0
                windows = []
                for window in range(96):
                    if window in OFFSETS or (seed == SEEDS[0] and history == "full" and window == 16):
                        path = directory / f"window_{window:03d}.pt"
                        torch.save(snapshot(model, optimizer, updates, window * 64), path)
                        parent_hashes[str(path)] = sha256(path)
                    row = optimize_window(
                        model, optimizer, history, images, window * 64, torch.Generator().manual_seed(0)
                    )
                    windows.append(row)
                    updates += row["optimizer_updates"]
                    if (window + 1) % 16 == 0:
                        progress(stage="reconstructing_parents", seed=seed, history=history, window=window + 1)
                final = snapshot(model, optimizer, updates, 6144)
                expected = torch.load(
                    PRIOR / history / f"seed_{seed}" / "final_state.pt", map_location="cpu", weights_only=False
                )
                old_windows = json.loads((PRIOR / history / f"seed_{seed}" / "windows.json").read_text())["windows"]
                if fingerprint(final) != fingerprint(expected) or any(
                    row["loss_gradients"] != old["loss_gradients"]
                    for row, old in zip(windows, old_windows, strict=True)
                ):
                    raise ValueError("reconstructed complete history or losses differ")
                torch.save(final, directory / "final_state.pt")
                atomic_json(
                    directory / "reproduction.json",
                    {
                        "passed": True,
                        "exact_final_and_all_losses": True,
                        "optimizer_updates": updates,
                        "windows": windows,
                    },
                )
                parent_updates += updates
        atomic_json(out / "parent_lock.json", {"hashes": parent_hashes, "reconstruction_updates": parent_updates})
        progress(stage="restored_continuation_preflight")
        first = out / "parents/full/seed_5003"
        parent = torch.load(first / "window_008.pt", map_location="cpu", weights_only=False)
        marker = fingerprint(parent)
        model, optimizer = restore(initial, config, parent)
        if fingerprint(snapshot(model, optimizer, 16, 512)) != marker:
            raise ValueError("preflight restoration differs")
        continued = run_action(model, optimizer, images, 8, "full")
        expected = torch.load(first / "window_016.pt", map_location="cpu", weights_only=False)
        if fingerprint(snapshot(model, optimizer, 32, 1024)) != fingerprint(expected):
            raise ValueError("restored full continuation differs")
        torch.save(snapshot(model, optimizer, 32, 1024), out / "compatibility_final_state.pt")
        parent_trace = json.loads((first / "reproduction.json").read_text())["windows"][8:16]
        if [r["loss_gradients"] for r in continued] != [r["loss_gradients"] for r in parent_trace]:
            raise ValueError("restored continuation losses differ")
        model, optimizer = restore(initial, config, parent)
        smoke = run_action(model, optimizer, images, 8, "half", horizon=1)
        if fingerprint(parent) != marker:
            raise ValueError("preflight mutated parent")
        atomic_json(
            out / "preflight.json",
            {
                "passed": True,
                "restored_continuation_exact": True,
                "continuation_updates": 16,
                "discarded_half_updates": 1,
                "continuation": continued,
                "half_smoke": smoke,
            },
        )
        progress(stage="reproducing_frozen")
        frozen_dir = out / "frozen"
        frozen_dir.mkdir()
        frozen, reference = evaluate(initial, evaluation, None, 0, True, 100, frozen_dir)
        if frozen != json.loads((PRIOR / "frozen/result.json").read_text()):
            raise ValueError("cached fixed readout does not reproduce frozen")
        atomic_json(frozen_dir / "result.json", frozen)
        for history_index, history in enumerate(HISTORIES):
            for seed_index, seed in enumerate(SEEDS):
                for offset_index, offset in enumerate(OFFSETS):
                    parent_file = out / "parents" / history / f"seed_{seed}" / f"window_{offset:03d}.pt"
                    parent = torch.load(parent_file, map_location="cpu", weights_only=False)
                    parent_marker = fingerprint(parent)
                    actions = (
                        ("full", "half") if (history_index + seed_index + offset_index) % 2 == 0 else ("half", "full")
                    )
                    for action in actions:
                        directory = out / "branches" / history / f"seed_{seed}" / f"window_{offset:03d}" / action
                        directory.mkdir(parents=True, exist_ok=False)
                        progress(stage="branch_training", history=history, seed=seed, offset=offset, action=action)
                        tick = time.perf_counter()
                        model, optimizer = restore(initial, config, parent)
                        restoration_seconds = time.perf_counter() - tick
                        before = snapshot(model, optimizer, parent["optimizer_updates"], offset * 64)
                        restored_marker = fingerprint(before)
                        if restored_marker != parent_marker:
                            raise ValueError("branch initial restoration differs")
                        torch.save(before, directory / "initial_state.pt")
                        rows = run_action(model, optimizer, images, offset, action)
                        updates = sum(r["optimizer_updates"] for r in rows)
                        final = snapshot(
                            model, optimizer, parent["optimizer_updates"] + updates, (offset + HORIZON) * 64
                        )
                        final_marker = fingerprint(final)
                        torch.save(final, directory / "final_state.pt")
                        progress(stage="branch_readout", history=history, seed=seed, offset=offset, action=action)
                        outcome, _ = evaluate(
                            model, evaluation, reference, (offset + HORIZON) * 2, True, 100, directory
                        )
                        if final_marker != fingerprint(
                            snapshot(model, optimizer, final["optimizer_updates"], final["arrivals"])
                        ):
                            raise ValueError("branch readout mutated learning state")
                        if parent_marker != fingerprint(parent) or restored_marker != fingerprint(before):
                            raise ValueError("branch mutated initial snapshot or parent")
                        atomic_json(
                            directory / "result.json",
                            {
                                "history": history,
                                "seed": seed,
                                "offset": offset,
                                "action": action,
                                "parent_sha256": sha256(parent_file),
                                "restored_fingerprint": restored_marker,
                                "final_fingerprint": final_marker,
                                "windows": rows,
                                "outcome": outcome,
                                "optimizer_updates": updates,
                                "online_seconds": sum(r["online_seconds"] for r in rows),
                                "restoration_seconds": restoration_seconds,
                                "parent_and_readout_unchanged": True,
                            },
                        )
                        completed += 1
        if any(sha256(Path(name)) != expected for name, expected in {**hashes, **parent_hashes}.items()):
            raise ValueError("locked evidence changed during collection")
        progress(stage="collection_complete", parent_reconstruction_updates=parent_updates, branch_updates=960)
    except Exception as error:
        progress(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
