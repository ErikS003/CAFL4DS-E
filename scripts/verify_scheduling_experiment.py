"""Independent source, budget, frozen-choice and prediction-derived schedule audit."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from scripts.decode_history_video import timestamps
from scripts.run_paired_update_response import fingerprint

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5003, 5009, 5011)
DRAWS = (51011, 51017, 51023)
HALF = ("0011", "0101", "0110", "1001", "1010", "1100")
SCHEDULES = (*HALF, "1111", "0000")
READERS = ("finetune", "linear", "knn")
METRICS = ("balanced", "A", "B")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    lock = read(out / "input_lock.json")
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    intake = read(ROOT / "outputs/scheduling/20261013-frames/intake.json")
    source = read(ROOT / "outputs/scheduling/20261013-intake/selection.json")
    assert len(intake["clips"]) == len(source["clips"]) == 240
    ids = {c["clip_id"] for c in intake["clips"]}
    assert len(ids) == 240 and not ids & set(source["excluded_clip_ids"])
    assert source["warm_overlap_count"] == source["validation_overlap_count"] == 0
    pool_path = ROOT / "outputs/scheduling/20261013-intake/selection.pool.json"
    assert digest(pool_path) == source["eligible_pool_sha256"]
    pool = read(pool_path)
    assert ids <= {c["clip_id"] for group in pool.values() for c in group}
    assert intake["cumulative_download_bytes"] <= 6144 * 1024**2 and intake["replacements"] <= 24
    frame_checks = 0
    for clip in intake["clips"]:
        cache = Path(clip["frames_path"])
        pixels = np.load(cache, allow_pickle=False)
        assert pixels.shape == (512, 64, 64, 3) and pixels.dtype == np.uint8
        metadata = timestamps((cache.parent / "decoder.log").read_text(encoding="utf-8", errors="replace"))
        assert len({r["source_frame_index"] for r in metadata}) == 512
        for a, b, image in zip(metadata, clip["frames"], pixels, strict=True):
            assert all(a[k] == b[k] for k in a)
            assert hashlib.sha256(image.tobytes()).hexdigest() == b["rgb64_sha256"]
            frame_checks += 1
        assert (
            read(Path(clip["keyframe_path"]).parents[3] / "labels/train" / f"{clip['clip_id']}.json")["attributes"]
            == clip["keyframe_attributes"]
        )
    clips = {c["clip_id"]: c for c in intake["clips"]}
    assigned = []
    expected_patterns = ("DNDN", "NDND", "DDND", "NNDN", "DNDN", "NDND", "DDND", "DDND", "NNDN", "NNDN")
    for k, stream in enumerate(intake["streams"]):
        assert stream["pattern"] == expected_patterns[k]
        assert stream["role"] == ("development" if k < 4 else "heldout")
        for domain, group in zip(stream["pattern"], stream["periods"], strict=True):
            assert len(group) == 6
            for j, identifier in enumerate(group):
                attributes = clips[identifier]["keyframe_attributes"]
                assert attributes["timeofday"] == ("daytime" if domain == "D" else "night")
                assert attributes["scene"] == ("city street" if j % 2 == 0 else "highway")
                assert attributes["weather"] == "clear"
                assigned.append(identifier)
    assert len(assigned) == len(set(assigned)) == 240 and set(assigned) == ids
    rows = read(out / "collection.json")["records"]
    assert len(rows) == len({(r["stream"], r["seed"], r["schedule"]) for r in rows}) == 240
    assert sum(r["updates"] for r in rows) == 46080
    audit = read(out / "training-replay-audit.json")
    assert audit["passed"] and audit["audit_ssl_updates"] == 46080 and audit["complete_states"] == 960
    chosen = read(out / "development-selection.json")
    assert chosen["locked_inputs_sha256"] == digest(out / "input_lock.json")
    assert all(r["started_unix"] >= chosen["frozen_unix"] for r in rows if r["role"] == "heldout")
    arrays = ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz"
    with np.load(arrays, allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
        assert len(labels) == 400
        assert all(((labels == c) & (domains == d)).sum() == 100 for c in (0, 1) for d in (0, 1))
        assert not set(a["query"]) & {int(k) for draw in DRAWS for k in a[f"support_{draw}"]}
    seen_predictions = set()

    def scores(directory):
        p = out / directory
        if (p / "readout-alias.json").exists():
            p = out / read(p / "readout-alias.json")["directory"]
        seen_predictions.add(str(p))
        with np.load(p / "predictions.npz", allow_pickle=False) as predictions:
            assert set(predictions.files) == {f"{d}_{r}" for d in DRAWS for r in READERS}
            values = {}
            for r in READERS:
                values[r] = {}
                for m in METRICS:
                    mask = np.ones(400, bool) if m == "balanced" else domains == (0 if m == "A" else 1)
                    count = sum(int(((predictions[f"{d}_{r}"] == labels) & mask).sum()) for d in DRAWS)
                    values[r][m] = 100 * count / (3 * int(mask.sum()))
            readout = read(p / "readout.json")
            assert readout["ssl_state_unchanged"]
            assert all(v["steps"] == 100 for v in readout["fine_tuning_audits"].values())
            return values

    values = {}
    state_checks = 0
    for row in rows:
        trace = read(out / row["directory"] / "training.json")
        assert row["updates"] == row["schedule"].count("1") * 96
        assert len(trace["periods"]) == len(row["endpoints"]) == 4
        previous = torch.load(out / "warm" / f"seed_{row['seed']}" / "state.pt", map_location="cpu", weights_only=True)
        total = 0
        previous_prediction = f"warm/seed_{row['seed']}"
        for p, bit in enumerate(row["schedule"]):
            training = trace["periods"][p]
            assert training["updates"] == (96 if bit == "1" else 0)
            expected = list(range(p * 3072, (p + 1) * 3072)) if bit == "1" else []
            assert [k for b in training["blocks"] for k in b["selected_indices"]] == expected
            assert [k for b in training["blocks"] for batch in b["exposure_indices"] for k in batch] == expected
            total += training["updates"]
            state = torch.load(out / row["endpoints"][p] / "state.pt", map_location="cpu", weights_only=True)
            assert state["arrivals"] == (p + 1) * 3072 and state["updates"] == total
            assert state["schedule"] == row["schedule"] and state["period"] == p
            assert all(int(v["step"]) == total for v in state["learner"]["optimizer"]["state"].values())
            if bit == "0":
                assert fingerprint(state["learner"]) == fingerprint(previous)
                assert read(out / row["endpoints"][p] / "readout-alias.json")["directory"] == previous_prediction
            else:
                previous_prediction = row["endpoints"][p]
            previous = state["learner"]
            values[row["stream"], row["seed"], row["schedule"], p] = scores(row["endpoints"][p])
            state_checks += 1
    assert len(seen_predictions) == 483
    summary = read(out / "summary.json")
    dev = sorted({r["stream"] for r in rows if r["role"] == "development"})
    test = sorted({r["stream"] for r in rows if r["role"] == "heldout"})

    def metric(n, s, r, m, p=3):
        return statistics.fmean(values[n, seed, s, p][r][m] for seed in SEEDS)

    means = {s: statistics.fmean(metric(n, s, "finetune", "balanced") for n in dev) for s in HALF}
    independent_choice = min(HALF, key=lambda s: (-round(means[s], 12), s))
    assert independent_choice == chosen["schedule"] == summary["selected_fixed_schedule"]
    assert all(abs(means[s] - chosen["development_terminal_means"][s]) < 1e-10 for s in HALF)
    intervals = 0

    def check(effect, delta, family):
        nonlocal intervals
        mean = statistics.fmean(delta)
        half = float(t.ppf(1 - 0.05 / family / 2, 5)) * statistics.stdev(delta) / 6**0.5
        assert np.allclose(effect["stream_values_pp"], delta, atol=1e-10)
        assert abs(effect["mean_pp"] - mean) < 1e-10
        assert np.allclose(effect["ci95_pp"], [mean - half, mean + half], atol=1e-10)
        intervals += 1

    fixed = chosen["schedule"]
    for s in HALF:
        if s == fixed:
            continue
        for r in READERS:
            for level, family in (("ordinary", 1), ("adjusted", 15)):
                for m in METRICS:
                    delta = [metric(n, s, r, m) - metric(n, fixed, r, m) for n in test]
                    check(summary["half_vs_frozen_fixed"][s][r][level][m], delta, family)
    for ref in ("1111", "0000"):
        for r in READERS:
            for m in METRICS:
                check(
                    summary["fixed_vs_controls_secondary"][ref][r][m],
                    [metric(n, fixed, r, m) - metric(n, ref, r, m) for n in test],
                    1,
                )
    for s in SCHEDULES:
        for r in READERS:
            for m in METRICS:
                assert (
                    abs(summary["heldout_terminal_means"][s][r][m] - statistics.fmean(metric(n, s, r, m) for n in test))
                    < 1e-10
                )
                assert np.allclose(
                    summary["trajectory_means"][s][r][m],
                    [statistics.fmean(metric(n, s, r, m, p) for n in test) for p in range(4)],
                    atol=1e-10,
                )
    winners = {n: min(HALF, key=lambda s: (-round(metric(n, s, "finetune", "balanced"), 12), s)) for n in test}
    assert winners == summary["optimistic_six_schedule_reference"]["winners"]
    for n in test:
        gap = metric(n, winners[n], "finetune", "balanced") - metric(n, fixed, "finetune", "balanced")
        assert abs(gap - summary["optimistic_six_schedule_reference"]["stream_gaps_pp"][n]) < 1e-10
    patterns = {r["stream"]: r["pattern"] for r in rows}
    pattern_choices = {}
    for pattern in set(patterns.values()):
        training = [n for n in dev if patterns[n] == pattern]
        pattern_means = {s: statistics.fmean(metric(n, s, "finetune", "balanced") for n in training) for s in HALF}
        pattern_choices[pattern] = min(HALF, key=lambda s: (-round(pattern_means[s], 12), s))
    assert pattern_choices == chosen["pattern_specific_schedules"]
    for n in test:
        gap = metric(n, pattern_choices[patterns[n]], "finetune", "balanced") - metric(n, fixed, "finetune", "balanced")
        assert abs(gap - summary["development_pattern_specific_secondary_gaps_pp"][n]) < 1e-10
        other = [k for k in test if k != n and patterns[k] == patterns[n]]
        if other:
            means = {s: statistics.fmean(metric(k, s, "finetune", "balanced") for k in other) for s in HALF}
            s = min(HALF, key=lambda s: (-round(means[s], 12), s))
            result = summary["other_replicate_selected_secondary"][n]
            assert result["schedule_selected_on_other_replicate"] == s
            assert (
                abs(result["gap_pp"] - metric(n, s, "finetune", "balanced") + metric(n, fixed, "finetune", "balanced"))
                < 1e-10
            )
    for s in SCHEDULES:
        seconds = [r["online_seconds"] for r in rows if r["role"] == "heldout" and r["schedule"] == s]
        assert np.allclose(seconds, summary["timing"][s]["stream_seed_seconds"])
        assert abs(statistics.fmean(seconds) - summary["timing"][s]["mean_online_seconds"]) < 1e-10
    qualified = []
    for s, result in summary["half_vs_frozen_fixed"].items():
        effects = result["finetune"]["adjusted"]
        if (
            effects["balanced"]["mean_pp"] >= 0.5
            and effects["balanced"]["ci95_pp"][0] > 0
            and all(effects[m]["ci95_pp"][0] > -1 for m in ("A", "B"))
        ):
            qualified.append(s)
    assert sorted(qualified) == sorted(summary["qualified_static_alternatives"])
    assert not summary["adaptive_controller_qualified"]
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "locked_hashes": len(lock["hashes"]),
            "source_frames": frame_checks,
            "complete_states": state_checks,
            "prediction_sets": 483 * 9,
            "reader_metric_intervals": intervals,
            "selected_schedule": fixed,
            "scientific_ssl_updates": 46080,
            "audit_ssl_updates": 46080,
            "primary_fits": 1449,
        },
    )
    print(json.dumps(read(out / "independent-verification.json")), flush=True)


if __name__ == "__main__":
    main()
