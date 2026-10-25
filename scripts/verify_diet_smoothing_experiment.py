"""Independent admission, parent-state and integer-statistics checks for return actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from scripts.run_paired_update_response import fingerprint

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/regime-windows/20261013-gpu"
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
CONTEXTS = ("day2", "night2")
ACTIONS = ("hold", "full", "smooth", "abrupt")
CLOCKS = (256, 4224)
METRICS = ("balanced", "A", "B")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    lock = json.loads((out / "input_lock.json").read_text())
    assert all(digest(Path(p)) == v for p, v in lock["hashes"].items())
    role = json.loads((out / "data-role-manifest.json").read_text())
    prior_role = json.loads((PRIOR / "data-role-manifest.json").read_text())
    assert not {c["clip_id"] for c in role["contexts"]["night2"]} & {c["clip_id"] for c in prior_role["clips"]}
    caches = {}
    timestamps_checked = 0
    pattern = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)")
    for context, clips in role["contexts"].items():
        assert len(clips) == 8
        assert sum(c["keyframe_attributes"]["scene"] == "city street" for c in clips) == 4
        assert sum(c["keyframe_attributes"]["scene"] == "highway" for c in clips) == 4
        caches[context] = []
        for clip in clips:
            assert clip["keyframe_attributes"]["timeofday"] == ("daytime" if context == "day2" else "night")
            cache = Path(clip["frames_path"])
            arrays = np.load(cache, allow_pickle=False)
            assert arrays.shape == (528, 64, 64, 3)
            reconstructed = []
            for match in pattern.finditer((cache.parent / "decoder.log").read_text(errors="replace")):
                n, pts, sec, w, h = match.groups()
                sec = float(sec)
                if (
                    len(reconstructed) < 528
                    and sec >= len(reconstructed) / 16
                    and (not reconstructed or sec > reconstructed[-1][2])
                ):
                    assert (int(w), int(h)) == (1280, 720)
                    reconstructed.append((int(n), int(pts), sec))
            assert len(reconstructed) == 528 and len({r[0] for r in reconstructed}) == 528
            for actual, saved, image in zip(reconstructed, clip["frames"], arrays, strict=True):
                assert actual == (saved["source_frame_index"], saved["pts"], saved["pts_seconds"])
                assert hashlib.sha256(image.tobytes()).hexdigest() == saved["rgb64_sha256"]
                timestamps_checked += 1
            caches[context].append(arrays)
    history_clips = [c for c in prior_role["clips"] if c["visit"] == "night"]
    memory = (
        torch.from_numpy(np.concatenate([np.load(c["frames_path"])[-64:] for c in history_clips]))
        .permute(0, 3, 1, 2)
        .float()
        / 255
    )
    assert memory.shape == (512, 3, 64, 64)
    rows = json.loads((out / "collection.json").read_text())["records"]
    assert len(rows) == 24 and sum(r["updates"] for r in rows) == 2376
    parents = {
        s: torch.load(
            PRIOR / "runs/w32/full" / f"seed_{s}" / "at_8448/state.pt", map_location="cpu", weights_only=True
        )["learner"]
        for s in SEEDS
    }
    hashes = {}
    costs = {}
    for row in rows:
        directory = out / row["directory"]
        trace = json.loads((directory / "training.json").read_text())
        action, context, seed = row["action"], row["context"], row["seed"]
        expected_updates = {"hold": 0, "smooth": 132, "abrupt": 132, "full": 132}[action]
        assert trace["updates"] == row["updates"] == expected_updates
        assert trace["accepted_frames"] == (3952 if action in ("smooth", "abrupt") else expected_updates * 32)
        assert len(trace["blocks"]) == 132
        exposed = []
        historical_exposed = []
        pointer = 0
        for index, block in enumerate(trace["blocks"]):
            start = index * 32
            assert block["start"] == start and block["end"] == start + 32
            if action in ("smooth", "abrupt"):
                h = max(0, 16 - index // 2) if action == "smooth" else (32 if index < 8 else 16 if index == 8 else 0)
                old = [(pointer + j * 17) % 512 for j in range(h)]
                pointer += h * 17
                new = [start + (j + 1) * 32 // (32 - h) - 1 for j in range(32 - h)] if h < 32 else []
                assert block["history_indices"] == old and block["historical_count"] == h
                assert block["exposure_indices"] == [[-j - 1 for j in old] + new]
                historical_exposed.extend(old)
            else:
                new = [] if action == "hold" else list(range(start, start + 32))
                assert block["exposure_indices"] == ([] if action == "hold" else [new])
            assert block["selected_indices"] == new and not block["queued_indices"]
            assert block["updates"] == (0 if action == "hold" else 1)
            exposed.extend(new)
        expected_new = 3952 if action in ("smooth", "abrupt") else expected_updates * 32
        assert len(set(exposed)) == len(exposed) == expected_new
        if action in ("smooth", "abrupt"):
            assert len(historical_exposed) == 272
            assert len(exposed) + len(historical_exposed) == 4224
        for clock in CLOCKS:
            checkpoint = directory / f"at_{clock}"
            state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
            count = sum(b["updates"] for b in trace["blocks"] if b["end"] <= clock)
            assert (
                state["updates"] == count and state["total_updates"] == 264 + count and state["parent_updates"] == 264
            )
            assert {int(v["step"]) for v in state["learner"]["optimizer"]["state"].values()} == {264 + count}
            assert not state["pending_indices"] and not state["queued_indices"]
            if action in ("smooth", "abrupt"):
                assert torch.equal(state["history_images"], memory)
                assert (
                    state["history_pointer"]
                    == sum(b["historical_count"] for b in trace["blocks"] if b["end"] <= clock) * 17
                )
            if action == "hold":
                assert fingerprint(state["learner"]) == fingerprint(parents[seed])
                alias = json.loads((checkpoint / "readout-alias.json").read_text())
                assert alias["directory"] == f"parents/seed_{seed}"
            if context == "day2" and action == "full":
                previous = PRIOR / "runs/w32/full" / f"seed_{seed}" / f"at_{8448 + clock}"
                old = torch.load(previous / "state.pt", map_location="cpu", weights_only=True)
                assert fingerprint(old["learner"]) == fingerprint(state["learner"])
                with np.load(previous / "predictions.npz") as a, np.load(checkpoint / "predictions.npz") as b:
                    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
            hashes[row["directory"] + f"/at_{clock}"] = {"state": digest(checkpoint / "state.pt")}
            if action != "hold":
                hashes[row["directory"] + f"/at_{clock}"]["predictions"] = digest(checkpoint / "predictions.npz")
        costs[row["directory"]] = {
            k: sum(b[k] for b in trace["blocks"]) for k in ("selection_seconds", "training_seconds", "online_seconds")
        }
    with np.load(PRIOR / "evaluation_arrays.npz") as arrays:
        labels, domains = arrays["labels"][arrays["query"]], arrays["domains"][arrays["query"]]
        assert len(labels) == 400 and all(
            int(((labels == c) & (domains == d)).sum()) == 100 for c in (0, 1) for d in (0, 1)
        )
        assert all(len(arrays[f"support_{d}"]) == 50 for d in DRAWS)
    counts = {}
    directories = (
        [(f"parents/seed_{s}", out / "parents" / f"seed_{s}") for s in SEEDS]
        + [
            (r["directory"] + f"/at_{c}", out / r["directory"] / f"at_{c}")
            for r in rows
            if r["action"] != "hold"
            for c in CLOCKS
        ]
        + [("warm_frozen", PRIOR / "frozen")]
    )
    for name, directory in directories:
        with np.load(directory / "predictions.npz") as pred:
            for draw in DRAWS:
                for reader in ("finetune", "linear", "knn"):
                    correct = pred[f"{draw}_{reader}"] == labels
                    counts[name, draw, reader] = {
                        "balanced": int(correct.sum()),
                        "A": int(correct[domains == 0].sum()),
                        "B": int(correct[domains == 1].sum()),
                    }
    for seed in SEEDS:
        with (
            np.load(out / "parents" / f"seed_{seed}" / "predictions.npz") as a,
            np.load(PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448/predictions.npz") as b,
        ):
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    by = {(r["context"], r["action"], r["seed"]): r for r in rows}

    def total(context, action, seed, clock, metric, reader="finetune"):
        name = (
            "warm_frozen"
            if action == "warm_frozen"
            else f"parents/seed_{seed}"
            if action == "hold"
            else by[context, action, seed]["directory"] + f"/at_{clock}"
        )
        return sum(counts[name, d, reader][metric] for d in DRAWS)

    def check_interval(saved, values, denominator):
        pp = [100 * v / denominator for v in values]
        mean = statistics.fmean(pp)
        half = float(t.ppf(0.975, 2)) * statistics.stdev(pp) / 3**0.5
        assert abs(saved["mean_pp"] - mean) < 1e-10
        assert np.allclose(saved["ci95_pp"], [mean - half, mean + half], atol=1e-10)
        assert saved["wins"] == sum(v > 0 for v in values) and saved["ties"] == sum(v == 0 for v in values)

    summary = json.loads((out / "summary.json").read_text())
    checks = 0
    for context in CONTEXTS:
        result = summary["contexts"][context]
        for action in ACTIONS:
            for clock in CLOCKS:
                for reader in ("finetune", "linear", "knn"):
                    for metric in METRICS:
                        denom = 1200 if metric == "balanced" else 600
                        actual = (
                            100 * sum(total(context, action, s, clock, metric, reader) for s in SEEDS) / (denom * 3)
                        )
                        recorded = (
                            result["means_accuracy_percent"][action][str(clock)][metric]
                            if reader == "finetune"
                            else result["secondary_accuracy_percent"][reader][action][str(clock)][metric]
                        )
                        assert abs(actual - recorded) < 1e-10
        for name, contrast in result["contrasts"].items():
            first, second = name.split("_minus_")
            for clock in CLOCKS:
                for metric in METRICS:
                    check_interval(
                        contrast[str(clock)][metric],
                        [
                            total(context, first, s, clock, metric) - total(context, second, s, clock, metric)
                            for s in SEEDS
                        ],
                        1200 if metric == "balanced" else 600,
                    )
                    checks += 1
        check_interval(
            result["smooth_useful_vs_warm_frozen"],
            [
                total(context, "smooth", s, 4224, "balanced") - total(context, "warm_frozen", s, 4224, "balanced")
                for s in SEEDS
            ],
            1200,
        )
        checks += 1
        for action in ACTIONS:
            preferred = all(
                result["contrasts"][action + "_minus_" + other]["4224"]["balanced"]["ci95_pp"][0] > 0
                and result["contrasts"][action + "_minus_" + other]["4224"]["balanced"]["mean_pp"] >= 0.1
                and all(
                    result["contrasts"][action + "_minus_" + other]["4224"][metric]["ci95_pp"][0] > -1
                    for metric in ("A", "B")
                )
                for other in ACTIONS
                if other != action
            )
            assert preferred == result["quality_preferred"][action]
        savings = [
            1 - by[context, "smooth", s]["online_seconds"] / by[context, "full", s]["online_seconds"] for s in SEEDS
        ]
        assert abs(statistics.fmean(savings) - result["mean_paired_time_saving"]) < 1e-12
        resource = False
        assert resource == result["smooth_resource_lead_descriptive_only"]
    for action in ("full", "smooth", "abrupt"):
        for clock in CLOCKS:
            for metric in METRICS:
                vals = [
                    total("night2", action, s, clock, metric)
                    - total("night2", "hold", s, clock, metric)
                    - total("day2", action, s, clock, metric)
                    + total("day2", "hold", s, clock, metric)
                    for s in SEEDS
                ]
                check_interval(
                    summary["context_action_interaction_night_minus_day"][action][str(clock)][metric],
                    vals,
                    1200 if metric == "balanced" else 600,
                )
                checks += 1
    replay = json.loads((out / "training-replay-audit.json").read_text())
    assert (
        replay["passed"]
        and replay["branches"] == 24
        and replay["checkpoints"] == 48
        and replay["audit_ssl_updates"] == 2376
    )
    payload = {
        "passed": True,
        "locked_hashes": len(lock["hashes"]),
        "source_timestamp_frames_checked": timestamps_checked,
        "branches": 24,
        "checkpoints": 48,
        "new_prediction_sets": 351,
        "independent_primary_intervals": checks,
        "all_hold_learners_rng_adam_unchanged": True,
        "prior_full_day_states_predictions_and_parent_readouts_exact": True,
        "all_quality_and_resource_gates_checked": True,
        "verification_source_sha256": digest(Path(__file__)),
        "artifact_hashes": hashes,
    }
    atomic = lambda name, value: (out / name).write_text(json.dumps(value, indent=2))
    atomic("independent-verification.json", payload)
    atomic("cost-breakdown.json", costs)
    print(
        json.dumps(payload | {"artifact_hashes": "48 checkpoint hashes plus36 new prediction artifact hashes archived"})
    )


if __name__ == "__main__":
    main()
