"""Matched exposure with reversed day/night histories; fixed primary day interaction."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import Evaluation
from cafl4ds.return_actions import run_branch
from cafl4ds.short_buffer import capture, restore
from cafl4ds.state_horizon import observe_before_action
from cafl4ds.history_order import ACTIONS, CHECKPOINTS, STATES, CASES, PARENTS, report_history, history_indices
from hydra.utils import instantiate
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_regime_windows_experiment import evaluate_gpu
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/regime-windows/20261013-gpu"
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
METRICS = ("balanced", "A", "B")
READERS = ("finetune", "linear", "knn")


def load_inputs(out):
    previous = json.loads((PRIOR / "input_lock.json").read_text())
    verification = json.loads((PRIOR / "independent-verification.json").read_text())
    if not verification["passed"] or any(digest(Path(p)) != v for p, v in previous["hashes"].items()):
        raise ValueError("Prior audited inputs changed")
    role = json.loads((PRIOR / "data-role-manifest.json").read_text())
    arrays = []
    for clip in role["clips"]:
        frames = np.load(clip["frames_path"], allow_pickle=False)
        if digest(Path(clip["frames_path"])) != clip["frames_sha256"]:
            raise ValueError("Source cache changed")
        for metadata, image in zip(clip["frames"], frames, strict=True):
            if hashlib.sha256(image.tobytes()).hexdigest() != metadata["rgb64_sha256"]:
                raise ValueError("Source pixels changed")
        arrays.append(frames)
    arrivals = torch.from_numpy(np.concatenate(arrays)).permute(0, 3, 1, 2).float() / 255
    intake_path = ROOT / "outputs/history-order/20261013-frames/intake.json"
    intake = json.loads(intake_path.read_text())
    if not intake["complete"] or len(intake["blocks"]) != 12:
        raise ValueError("Twelve fresh day blocks required")
    secondary_path = ROOT / "outputs/cross-state/20261013-frames/intake.json"
    secondary = json.loads(secondary_path.read_text())
    blocks = intake["blocks"] + [b for b in secondary["blocks"] if b["context"] == "night"]
    streams = {}
    source_ids = {c["clip_id"] for c in role["clips"]}
    for block in blocks:
        values = []
        for clip in block["clips"]:
            if clip["clip_id"] in source_ids or digest(Path(clip["frames_path"])) != clip["frames_sha256"]:
                raise ValueError("Continuation overlaps historical clips or cache changed")
            values.append(np.load(clip["frames_path"], allow_pickle=False)[:512])
        streams[block["context"], block["block"]] = (
            torch.from_numpy(np.concatenate(values)).permute(0, 3, 1, 2).float() / 255
        )
    parents, traces = {}, {}
    paths = [
        PRIOR / "input_lock.json",
        PRIOR / "independent-verification.json",
        PRIOR / "data-role-manifest.json",
        PRIOR / "evaluation_arrays.npz",
        intake_path,
        secondary_path,
        ROOT / "outputs/cross-state/20261013-gpu/independent-verification.json",
        ROOT / "outputs/history-order/20261013-intake/selection.json",
        ROOT / "outputs/history-order/20261013-intake/selection.pool.json",
        ROOT / "outputs/history-order/20261013-intake/download_manifest.json",
        ROOT / "scripts/select_history_order_clips.py",
        ROOT / "scripts/prepare_history_order_frames.py",
        ROOT / "scripts/fetch_cross_clips.py",
        Path(__file__),
        ROOT / "scripts/verify_history_order_experiment.py",
        ROOT / "docs/project-plan/slow-edge-history-order-frozen.md",
        ROOT / "scripts/run_regime_windows_experiment.py",
        ROOT / "scripts/run_adaptation_experiment.py",
        ROOT / "scripts/run_paired_update_response.py",
        ROOT / "scripts/run_short_buffer_experiment.py",
    ]
    for seed in SEEDS:
        directory = PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448"
        paths.extend((directory / "state.pt", directory / "readout.json", directory / "predictions.npz"))
    image_path = ROOT / "outputs/content-filter-bdd/20261011-coverage-half/evaluation_images.pt"
    images = torch.load(image_path, map_location="cpu", weights_only=True)
    paths.append(image_path)
    with np.load(PRIOR / "evaluation_arrays.npz", allow_pickle=False) as a:
        data = Evaluation(
            images,
            torch.from_numpy(a["labels"].copy()),
            torch.from_numpy(a["domains"].copy()),
            torch.from_numpy(a["query"].copy()),
            {d: torch.from_numpy(a[f"support_{d}"].copy()) for d in DRAWS},
            role,
        )
    for block in blocks:
        for clip in block["clips"]:
            cache = Path(clip["frames_path"])
            paths.extend(
                (
                    cache,
                    Path(clip["video_path"]),
                    cache.parent / "manifest.json",
                    cache.parent / "decoder.log",
                    Path(clip["keyframe_path"]).parents[3] / "labels/train" / f"{clip['clip_id']}.json",
                )
            )
    paths.extend((ROOT / "outputs/history-order/20261013-intake").glob("amendment-*/*.json"))
    paths.extend(
        (
            ROOT / "scripts/decode_history_video.py",
            ROOT / "scripts/decode_regime_video.py",
            ROOT / "outputs/history-order/20261013-frames/allocation.json",
            ROOT / ".cache/history-known-videos.txt",
        )
    )
    paths.extend(
        (
            ROOT / "outputs/temporal-bdd/20261011-intake/archive_index.json",
            ROOT / "outputs/history-order/20261013-intake/protocol-sampling-original.md",
        )
    )
    paths.extend((ROOT / "cafl4ds").rglob("*.py"))
    hashes = {**previous["hashes"], **{str(p): digest(p) for p in paths}}
    lock = json.loads(
        json.dumps(
            {
                "hashes": hashes,
                "config": previous["config"],
                "states": STATES,
                "seeds": SEEDS,
                "checkpoints": CHECKPOINTS,
                "actions": ACTIONS,
                "scientific_ssl_updates": 4848,
                "primary_fits": 324,
                "parent_history_frames": 8448,
                "parent_updates": 264,
            }
        )
    )
    if (out / "input_lock.json").exists() and json.loads((out / "input_lock.json").read_text()) != lock:
        raise ValueError("Resume lock changed")
    atomic_json(out / "input_lock.json", lock)
    atomic_json(
        out / "data-role-manifest.json",
        {
            "states": STATES,
            "parent_source": str(PRIOR),
            "query": data.query.tolist(),
            "supports": {str(k): v.tolist() for k, v in data.supports.items()},
            "development_only": True,
            "overlapping_suffixes": False,
            "cases": CASES,
            "new_blocks": blocks,
            "controller_fit": False,
        },
    )
    for path in paths:
        if path.suffix in (".py", ".md") and path.is_relative_to(ROOT):
            destination = out / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    return OmegaConf.create(previous["config"]), arrivals, streams, parents, traces, data, lock


def stamp(state, parent_clock):
    state.update(
        parent_arrivals=parent_clock,
        parent_updates=parent_clock // 32,
        total_updates=parent_clock // 32 + state["updates"],
    )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = {"pid": os.getpid(), "total_branches": 204}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - start)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        progress(stage="locking_inputs")
        config, arrivals, streams, parents, traces, data, lock = load_inputs(out)
        torch.manual_seed(11)
        initial = _method(config).cuda()
        warm_states = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            optimizer = instantiate(config.optim, params=list(initial.parameters()))
            state = capture(initial, optimizer)
            directory = out / "warm-parents" / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "state.pt").exists():
                stored = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
                if fingerprint(state) != fingerprint(stored):
                    raise ValueError("Warm initial state changed")
            else:
                torch.save(state, directory / "state.pt")
            warm_states[seed] = state
        histories = {"DN": arrivals[:8448], "ND": torch.cat((arrivals[4224:8448], arrivals[:4224]))}
        if histories["DN"].stride() != histories["ND"].stride():
            raise ValueError("History tensor layouts differ")
        if not (out / "preflight.json").exists():
            signatures = []
            for _ in range(2):
                model, optim = restore(initial, config.optim, warm_states[SEEDS[0]])
                before = fingerprint(capture(model, optim))
                observe_before_action(model, arrivals[8320:8448], arrivals[8448:8480], 0.5, SEEDS[0])
                if fingerprint(capture(model, optim)) != before:
                    raise ValueError("Observer mutated parent")
                run_branch(model, optim, arrivals[8448:8512], "full", SEEDS[0], (64,))
                signatures.append(fingerprint(capture(model, optim)))
            if signatures[0] != signatures[1]:
                raise ValueError("Preflight continuation differs")
            run_branch(model, optim, arrivals[8448:8512], "hold", SEEDS[0], (64,))
            if fingerprint(capture(model, optim)) != signatures[-1]:
                raise ValueError("Preflight hold mutated learner")
            evaluate_gpu(model, optim, data, out / "preflight")
            atomic_json(
                out / "preflight.json", {"passed": True, "discarded_ssl_updates": 4, "discarded_primary_fits": 3}
            )
        atomic_json(out / "history-indices.json", {order: history_indices(order) for order in PARENTS})
        parent_records = []
        for seed in SEEDS:
            for order in PARENTS:
                progress(stage="parent_history", order=order, seed=seed)
                directory = out / "parents" / order / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                model, optimizer = restore(initial, config.optim, warm_states[seed])
                states = {}

                def collect_parent(clock, state):
                    states[clock] = state
                    return {"state_fingerprint": fingerprint(state), "updates": state["updates"]}

                training = run_branch(model, optimizer, histories[order], "full", seed, (8448,), collect_parent)
                state = states[8448]
                if (directory / "state.pt").exists():
                    stored = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
                    if fingerprint(state) != fingerprint(stored):
                        raise ValueError("Cached parent replay changed")
                else:
                    torch.save(state, directory / "state.pt")
                atomic_json(directory / "training.json", training)
                if order == "DN":
                    old = torch.load(
                        PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448/state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    if fingerprint(state["learner"]) != fingerprint(old["learner"]):
                        raise ValueError("Same-order parent not exactly reproduced")
                evaluate_gpu(model, optimizer, data, directory)
                if order == "DN":
                    with (
                        np.load(directory / "predictions.npz") as a,
                        np.load(PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448/predictions.npz") as b,
                    ):
                        if not all(np.array_equal(a[k], b[k]) for k in a.files):
                            raise ValueError("Same-order parent predictions differ")
                parents[order, seed] = state["learner"]
                traces[order, seed] = training["blocks"]
                parent_records.append(
                    {
                        "order": order,
                        "seed": seed,
                        "updates": 264,
                        "directory": str(directory.relative_to(out)),
                        "online_seconds": training["online_seconds"],
                    }
                )
        atomic_json(
            out / "parent-collection.json",
            {"records": parent_records, "scientific_ssl_updates": 1584, "primary_fits": 18},
        )
        rows = []
        for seed in SEEDS:
            for name, parent_clock in STATES.items():
                case = CASES[name]
                continuation = streams[case["context"], case["block"]]
                parent = parents[case["parent"], seed]
                model, optim = restore(initial, config.optim, parent)
                parent_directory = out / "parents" / case["parent"] / f"seed_{seed}"
                recent_losses = [
                    b["loss_gradients"][0][0] for b in traces[case["parent"], seed] if b["end"] <= parent_clock
                ][-4:]
                if len(recent_losses) != 4:
                    raise ValueError("Historical loss trace incomplete")
                before = fingerprint(capture(model, optim))
                observation = observe_before_action(
                    model,
                    histories[case["parent"]][-128:],
                    continuation[:32],
                    statistics.fmean(recent_losses),
                    seed,
                )
                if fingerprint(capture(model, optim)) != before:
                    raise ValueError("Observation mutated parent")
                (out / "observations" / name).mkdir(parents=True, exist_ok=True)
                atomic_json(out / "observations" / name / f"seed_{seed}.json", observation)
                for action in ACTIONS:
                    directory = out / "runs" / name / action / f"seed_{seed}"
                    directory.mkdir(parents=True, exist_ok=True)
                    progress(stage="branch", state=name, seed=seed, action=action, completed_branches=len(rows))
                    model, optim = restore(initial, config.optim, parent)

                    def observe(clock, state):
                        state = stamp(state, parent_clock)
                        checkpoint = directory / f"at_{clock}"
                        checkpoint.mkdir(exist_ok=True)
                        signature = fingerprint(state)
                        if (checkpoint / "state.pt").exists():
                            if (
                                fingerprint(torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True))
                                != signature
                            ):
                                raise ValueError("Partial branch state changed")
                        else:
                            torch.save(state, checkpoint / "state.pt")
                        if action == "full" and not (checkpoint / "readout.json").exists():
                            evaluate_gpu(model, optim, data, checkpoint)
                        if action == "hold":
                            atomic_json(
                                checkpoint / "readout-alias.json", {"directory": str(parent_directory.relative_to(out))}
                            )
                        return {"fingerprint": signature, "updates": state["updates"]}

                    training = run_branch(model, optim, continuation, action, seed, CHECKPOINTS, observe)
                    if training["updates"] != (32 if action == "full" else 0):
                        raise ValueError("Scientific budget mismatch")
                    atomic_json(directory / "training.json", training)
                    row = {
                        "state": name,
                        "seed": seed,
                        "action": action,
                        "directory": str(directory.relative_to(out)),
                        "updates": training["updates"],
                        "online_seconds": training["online_seconds"],
                    }
                    rows.append(row)
        atomic_json(
            out / "collection.json",
            {"records": rows, "scientific_ssl_updates": 1584 + sum(r["updates"] for r in rows), "primary_fits": 324},
        )
        for row in parent_records:
            order, seed = row["order"], row["seed"]
            progress(stage="parent_replay", order=order, seed=seed)
            model, optimizer = restore(initial, config.optim, warm_states[seed])
            states = {}
            repeated = run_branch(
                model,
                optimizer,
                histories[order],
                "full",
                seed,
                (8448,),
                lambda clock, state: states.setdefault(clock, state),
            )
            stored = torch.load(out / row["directory"] / "state.pt", map_location="cpu", weights_only=True)
            if fingerprint(states[8448]) != fingerprint(stored):
                raise ValueError("Parent history replay differs")
            original = json.loads((out / row["directory"] / "training.json").read_text())
            for a, b in zip(original["blocks"], repeated["blocks"], strict=True):
                for key in ("start", "end", "selected_indices", "exposure_indices", "updates", "loss_gradients"):
                    if json.dumps(a[key]) != json.dumps(b[key]):
                        raise ValueError("Parent trace replay differs")
        for index, row in enumerate(rows):
            name, seed, action = row["state"], row["seed"], row["action"]
            parent_clock = STATES[name]
            case = CASES[name]
            continuation = streams[case["context"], case["block"]]
            progress(stage="replay_audit", audited_branches=index, state=name, seed=seed, action=action)
            model, optim = restore(initial, config.optim, parents[case["parent"], seed])
            directory = out / row["directory"]

            def check(clock, state):
                state = stamp(state, parent_clock)
                stored = torch.load(directory / f"at_{clock}/state.pt", map_location="cpu", weights_only=True)
                if fingerprint(state) != fingerprint(stored):
                    raise ValueError("Complete checkpoint replay differs")
                if action == "hold" and fingerprint(state["learner"]) != fingerprint(parents[case["parent"], seed]):
                    raise ValueError("Hold parent invariant failed")
                return {"fingerprint": fingerprint(state), "updates": state["updates"]}

            replay = run_branch(model, optim, continuation, action, seed, CHECKPOINTS, check)
            original = json.loads((directory / "training.json").read_text())
            for a, b in zip(original["blocks"], replay["blocks"], strict=True):
                for key in ("start", "end", "selected_indices", "exposure_indices", "updates", "loss_gradients"):
                    if json.dumps(a[key]) != json.dumps(b[key]):
                        raise ValueError("Exposure/optimization replay differs")
        atomic_json(
            out / "training-replay-audit.json",
            {"passed": True, "branches": 204, "checkpoints": 204, "audit_ssl_updates": 4848},
        )
        if any(digest(Path(p)) != v for p, v in lock["hashes"].items()):
            raise ValueError("Locked source changed")
        result = report_history(out, data)
        progress(
            stage="complete",
            completed_branches=204,
            audited_branches=204,
            decision=result["decision"],
            scientific_ssl_updates=4848,
            primary_fits=324,
        )
    except Exception as error:
        progress(stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
