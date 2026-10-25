"""Independent source-time, selection, transient-buffer and readout pilot checks."""

from __future__ import annotations

import argparse
import ast
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
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
CLOCKS = (4224, 4480, 8448, 8704, 12672)
POLICIES = ("full", "uniform_half", "random_half", "live_loss_half")
WIDTHS = (32, 128, 512)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    lock = json.loads((out / "input_lock.json").read_text())
    assert all(digest(Path(p)) == v for p, v in lock["hashes"].items())
    repair = json.loads((out / "audit-metadata-repair.json").read_text())
    original = json.loads((out / "collection-input-lock.json").read_text())
    old_source = out / "collection-source/run_regime_windows_experiment.py"
    current = ROOT / "scripts/run_regime_windows_experiment.py"
    assert digest(old_source) == repair["original_collection_source_sha256"] == original["hashes"][str(current)]
    assert digest(current) == repair["audit_source_sha256"]
    assert digest(ROOT / "scripts/close_regime_windows_experiment.py") == repair["closure_source_sha256"]
    before = ast.parse(old_source.read_text())
    after = ast.parse(current.read_text())
    assert [
        ast.dump(node) for node in before.body if not isinstance(node, ast.FunctionDef) or node.name != "audit"
    ] == [ast.dump(node) for node in after.body if not isinstance(node, ast.FunctionDef) or node.name != "audit"]
    manifest = json.loads((out / "data-role-manifest.json").read_text())
    clips = manifest["clips"]
    assert len({c["clip_id"] for c in clips}) == 24
    decoder = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+).*?s:(\d+)x(\d+)")
    max_delay = 0
    frame_caches = []
    for index, clip in enumerate(clips):
        assert clip["visit"] == ("day1" if index < 8 else "night" if index < 16 else "day2")
        assert clip["keyframe_attributes"]["scene"] in ("city street", "highway")
        assert clip["keyframe_attributes"]["timeofday"] == ("night" if 8 <= index < 16 else "daytime")
        cache = Path(clip["frames_path"])
        assert digest(cache) == clip["frames_sha256"]
        frame_caches.append(np.load(cache, mmap_mode="r", allow_pickle=False))
        selected = []
        for match in decoder.finditer((cache.parent / "decoder.log").read_text(errors="replace")):
            number, pts, seconds, width, height = match.groups()
            seconds = float(seconds)
            if len(selected) < 528 and seconds >= len(selected) / 16 and (not selected or seconds > selected[-1][2]):
                assert (int(width), int(height)) == (1280, 720)
                selected.append((int(number), int(pts), seconds))
        assert len(selected) == 528 and len({row[0] for row in selected}) == 528
        for k, (actual, saved) in enumerate(zip(selected, clip["frames"], strict=True)):
            assert actual == (saved["source_frame_index"], saved["pts"], saved["pts_seconds"])
            max_delay = max(max_delay, actual[2] - k / 16)
    for start in (0, 8, 16):
        assert [c["keyframe_attributes"]["scene"] for c in clips[start : start + 8]].count("city street") == 4
    records = json.loads((out / "collection.json").read_text())["records"]
    assert len(records) == 36 and sum(r["updates"] for r in records) == 8910
    hashes = {}
    costs = {}
    crossings = []
    regime_admissions = []
    full_final = {}
    fixed_predictions = {}
    observations = 0
    for row in records:
        directory = out / row["directory"]
        trace = json.loads((directory / "training.json").read_text())
        width, policy, seed = row["width"], row["policy"], row["seed"]
        generator = torch.Generator().manual_seed(seed + 62003)
        queued = []
        all_selected = []
        all_exposed = []
        for block in trace["blocks"]:
            start, end = block["start"], block["end"]
            n = end - start
            assert n == width or end == 12672 and n == 384
            if policy == "full":
                indices = list(range(n))
            elif policy == "uniform_half":
                indices = list(range(1, n, 2))
            elif policy == "random_half":
                indices = sorted(torch.randperm(n, generator=generator)[: n // 2].tolist())
            else:
                losses = block["scores"]
                assert len(losses) == n and np.isfinite(losses).all()
                indices = sorted(sorted(range(n), key=lambda i: (-losses[i], i))[: n // 2])
                assert block["scoring_seed"] == seed + 72003 + start
            expected = [i + start for i in indices]
            assert block["selected_indices"] == expected
            queued.extend(expected)
            batches = []
            while len(queued) >= 32:
                batches.append(queued[:32])
                del queued[:32]
            assert block["exposure_indices"] == batches and block["queued_indices"] == queued
            assert block["updates"] == len(batches)
            all_selected.extend(expected)
            all_exposed.extend(i for batch in batches for i in batch)
            for boundary in (4224, 8448):
                if start < boundary < end:
                    crossings.append(
                        {
                            "width": width,
                            "policy": policy,
                            "seed": seed,
                            "boundary": boundary,
                            "window": [start, end],
                            "selected_before": sum(i < boundary for i in expected),
                            "selected_after": sum(i >= boundary for i in expected),
                        }
                    )
        assert not queued and all_selected == all_exposed and len(set(all_exposed)) == len(all_exposed)
        assert len(all_exposed) == (12672 if policy == "full" else 6336)
        regime_admissions.append(
            {
                "width": width,
                "policy": policy,
                "seed": seed,
                "day1": sum(i < 4224 for i in all_selected),
                "night": sum(4224 <= i < 8448 for i in all_selected),
                "day2": sum(i >= 8448 for i in all_selected),
            }
        )
        for clock in CLOCKS:
            checkpoint = directory / f"at_{clock}"
            state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
            assert state["arrivals"] == clock
            closed = [b for b in trace["blocks"] if b["end"] <= clock]
            expected_updates = sum(b["updates"] for b in closed)
            assert state["updates"] == expected_updates
            steps = {int(v["step"]) for v in state["learner"]["optimizer"]["state"].values()}
            assert steps == {expected_updates}
            last_end = closed[-1]["end"] if closed else 0
            assert state["pending_indices"] == list(range(last_end, clock))
            assert state["queued_indices"] == (closed[-1]["queued_indices"] if closed else [])
            assert state["pending_images"].shape[0] == len(state["pending_indices"])
            assert state["queued_images"].shape[0] == len(state["queued_indices"])
            for field in ("pending", "queued"):
                for image, index in zip(state[field + "_images"], state[field + "_indices"], strict=True):
                    raw = frame_caches[index // 528][index % 528].copy()
                    expected = torch.from_numpy(raw).permute(2, 0, 1).float() / 255
                    assert torch.equal(image, expected)
            if clock == 12672:
                assert not state["pending_indices"] and not state["queued_indices"]
                if policy in ("full", "uniform_half"):
                    key = (policy, seed)
                    signature = fingerprint(state["learner"])
                    assert key not in full_final or full_final[key] == signature
                    full_final[key] = signature
                    with np.load(checkpoint / "predictions.npz") as pred:
                        values = {name: pred[name].copy() for name in pred.files}
                    assert key not in fixed_predictions or all(
                        np.array_equal(values[k], fixed_predictions[key][k]) for k in values
                    )
                    fixed_predictions[key] = values
            hashes[row["directory"] + f"/at_{clock}"] = {
                "state": digest(checkpoint / "state.pt"),
                "predictions": digest(checkpoint / "predictions.npz"),
            }
            observations += 1
        costs[row["directory"]] = {
            k: sum(b[k] for b in trace["blocks"])
            for k in ("gather_seconds", "score_seconds", "selection_seconds", "training_seconds", "online_seconds")
        }
    with np.load(out / "evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
        assert len(labels) == 400 and all(
            int(((labels == c) & (domains == d)).sum()) == 100 for c in (0, 1) for d in (0, 1)
        )
        assert all(len(a[f"support_{draw}"]) == 50 for draw in DRAWS)
    counts = {}
    reader_directories = [("frozen", out / "frozen")] + [
        (r["directory"] + f"/at_{c}", out / r["directory"] / f"at_{c}") for r in records for c in CLOCKS
    ]
    for name, directory in reader_directories:
        with np.load(directory / "predictions.npz") as predictions:
            for draw in DRAWS:
                for reader in ("finetune", "linear", "knn"):
                    correct = predictions[f"{draw}_{reader}"] == labels
                    counts[name, draw, reader] = {
                        "balanced": int(correct.sum()),
                        "micro": int(correct.sum()),
                        "A": int(correct[domains == 0].sum()),
                        "B": int(correct[domains == 1].sum()),
                    }
    by = {(r["width"], r["policy"], r["seed"]): r for r in records}
    summary = json.loads((out / "summary.json").read_text())

    def total(width, policy, seed, clock, metric, reader="finetune"):
        name = "frozen" if policy == "frozen" else by[width, policy, seed]["directory"] + f"/at_{clock}"
        return sum(counts[name, d, reader][metric] for d in DRAWS)

    def check_interval(saved, values, denominator):
        values = [100 * v / denominator for v in values]
        mean = statistics.fmean(values)
        half = float(t.ppf(0.975, 2)) * statistics.stdev(values) / 3**0.5
        assert abs(saved["mean_pp"] - mean) < 1e-10
        assert np.allclose(saved["ci95_pp"], [mean - half, mean + half], atol=1e-10)
        assert saved["wins"] == sum(v > 0 for v in values) and saved["ties"] == sum(v == 0 for v in values)

    checks = 0
    secondary = {}
    for width in WIDTHS:
        for policy in POLICIES:
            arm = summary["arms"][f"w{width}/{policy}"]
            for clock in CLOCKS:
                for metric in ("balanced", "A", "B", "micro"):
                    denom = 600 if metric in ("A", "B") else 1200
                    actual = 100 * sum(total(width, policy, s, clock, metric) for s in SEEDS) / (denom * 3)
                    assert abs(actual - arm["trajectory_accuracy_percent"][str(clock)][metric]) < 1e-10
                    check_interval(
                        arm["matched_full"][str(clock)][metric],
                        [
                            total(width, policy, s, clock, metric) - total(width, "full", s, clock, metric)
                            for s in SEEDS
                        ],
                        denom,
                    )
                    checks += 1
            pairs = {
                "initial_day_learning": (4224, 0, "A"),
                "day_retention_change": (8448, 4224, "A"),
                "night_adaptation": (8448, 4224, "B"),
                "early_night_response": (4480, 4224, "B"),
                "day_recovery": (12672, 8448, "A"),
                "early_day_recovery": (8704, 8448, "A"),
                "final_day_vs_first_day": (12672, 4224, "A"),
                "night_retention_on_return": (12672, 8448, "B"),
            }
            for key, (first, second, metric) in pairs.items():
                values = [
                    total(width, policy, s, first, metric)
                    - total(width, "frozen" if second == 0 else policy, s, second, metric)
                    for s in SEEDS
                ]
                check_interval(arm["contrasts"][key], values, 600)
                checks += 1
            check_interval(
                arm["contrasts"]["final_balanced_vs_frozen"],
                [total(width, policy, s, 12672, "balanced") - total(width, "frozen", s, 0, "balanced") for s in SEEDS],
                1200,
            )
            values = [
                total(width, policy, s, 8448, "A")
                - total(width, policy, s, 4224, "A")
                - total(width, "full", s, 8448, "A")
                + total(width, "full", s, 4224, "A")
                for s in SEEDS
            ]
            check_interval(arm["retention_change_minus_full"], values, 600)
            checks += 2
            for other in ("uniform_half", "random_half"):
                for clock in CLOCKS:
                    values = [
                        total(width, policy, s, clock, "balanced") - total(width, other, s, clock, "balanced")
                        for s in SEEDS
                    ]
                    check_interval(arm["content_vs_uniform_random"][other][str(clock)], values, 1200)
                    checks += 1
            useful = arm["contrasts"]["final_balanced_vs_frozen"]["ci95_pp"][0] > 0
            absence = (
                arm["contrasts"]["initial_day_learning"]["ci95_pp"][0] > 0
                and arm["contrasts"]["day_retention_change"]["ci95_pp"][1] < 0
            )
            guard = (
                all(arm["matched_full"]["12672"][m]["ci95_pp"][0] > -1 for m in ("balanced", "A", "B"))
                and arm["retention_change_minus_full"]["ci95_pp"][0] > -1
            )
            assert useful == arm["useful_final_ssl"] and absence == arm["absence_decline_provisionally_supported"]
            assert arm["efficiency_pilot_lead"] == (
                policy != "full" and useful and guard and arm["mean_processing_saving_fraction"] >= 0.25
            )
            secondary[f"w{width}/{policy}"] = {
                reader: {
                    str(clock): {
                        m: 100
                        * sum(total(width, policy, s, clock, m, reader) for s in SEEDS)
                        / ((600 if m in ("A", "B") else 1200) * 3)
                        for m in ("balanced", "A", "B")
                    }
                    for clock in CLOCKS
                }
                for reader in ("linear", "knn")
            }
    replay = json.loads((out / "training-replay-audit.json").read_text())
    assert (
        replay["passed"]
        and replay["trajectories"] == 36
        and replay["checkpoints"] == 180
        and replay["audit_ssl_updates"] == 8910
    )
    result = {
        "passed": True,
        "locked_hashes": len(lock["hashes"]),
        "source_timestamp_frames_checked": 12672,
        "max_slot_delay_seconds": max_delay,
        "trajectories": 36,
        "complete_checkpoint_budget_checks": observations,
        "readout_prediction_sets": len(reader_directories) * 9,
        "independent_primary_intervals_checked": checks,
        "full_terminal_learner_identical_across_widths": True,
        "uniform_terminal_learner_and_fixed_policy_predictions_identical_across_widths": True,
        "crossing_windows": len(crossings),
        "verification_source_sha256": digest(Path(__file__)),
        "artifact_hashes": hashes,
    }
    atomic = lambda name, value: (out / name).write_text(json.dumps(value, indent=2))
    atomic("independent-verification.json", result)
    atomic("cost-breakdown.json", costs)
    atomic("crossing-window-diagnostics.json", crossings)
    atomic("secondary-readouts.json", secondary)
    atomic("regime-admission-counts.json", regime_admissions)
    print(json.dumps(result | {"artifact_hashes": "180 checkpoint/prediction pairs archived"}))


if __name__ == "__main__":
    main()
