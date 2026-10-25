"""Independent stopping-curve, source integrity, selection and transfer-gate audit."""

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
PRIOR = ROOT / "outputs/scheduling/20261013-gpu"
SEEDS = (5003, 5009, 5011)
BUDGETS = (0, 48, 96, 192, 288, 384)
DRAWS = (51011, 51017, 51023)
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
    assert read(PRIOR / "independent-verification.json")["passed"]
    intake = read(ROOT / "outputs/stopping-calibration/20261013-frames/intake.json")
    selected = read(ROOT / "outputs/stopping-calibration/20261013-intake/selection.json")
    assert len(intake["clips"]) == len(selected["clips"]) == 144
    assert selected["warm_overlap_count"] == selected["validation_overlap_count"] == 0
    fresh_ids = {c["clip_id"] for c in intake["clips"]}
    assert len(fresh_ids) == 144 and not fresh_ids & set(selected["excluded_clip_ids"])
    original_pool = Path(selected["original_pool_path"])
    assert digest(original_pool) == selected["original_pool_sha256"]
    original = read(original_pool)
    eligible = {c["clip_id"] for group in original.values() for c in group}
    assert fresh_ids <= eligible
    assert (
        digest(ROOT / "outputs/stopping-calibration/20261013-intake/selection.pool.json")
        == selected["eligible_pool_sha256"]
    )
    assert intake["cumulative_download_bytes"] <= 4096 * 1024**2
    assert intake["replacements"] <= 24
    old_intake = read(ROOT / "outputs/scheduling/20261013-frames/intake.json")
    old_ids = {c["clip_id"] for c in old_intake["clips"]}
    assert not fresh_ids & old_ids
    clips = {c["clip_id"]: c for c in old_intake["clips"] + intake["clips"]}
    patterns = ("DNDNDNDNDNDN", "NDNDNDNDNDND", "DDDNNNDDDNNN", "NNNDDDNNNDDD", "DDDDDDNNNDDD", "NNNNNNDDDNNN")
    assigned = []
    for i, stream in enumerate(intake["streams"]):
        assert stream["pattern"] == patterns[i]
        assert stream["dwell"] == ("short", "medium", "long")[i // 2]
        assert len(stream["periods"]) == 12
        for domain, pair in zip(stream["pattern"], stream["periods"], strict=True):
            assert len(pair) == 2
            for j, identifier in enumerate(pair):
                attrs = clips[identifier]["keyframe_attributes"]
                assert attrs["timeofday"] == ("daytime" if domain == "D" else "night")
                assert attrs["scene"] == ("city street" if j == 0 else "highway")
                assert attrs["weather"] == "clear"
                assigned.append(identifier)
    assert len(assigned) == len(set(assigned)) == 144
    dev = [s for s in lock["streams"] if s["role"] == "development"]
    assert len(dev) == 4 and all(s["old_name"] in ("stream_00", "stream_01", "stream_02", "stream_03") for s in dev)
    used_ids = {c for s in lock["streams"] for group in s["periods"] for c in group}
    assert len(used_ids) == 240
    pixel_checks = 0
    for identifier in sorted(used_ids):
        c = clips[identifier]
        cache = Path(c["frames_path"])
        array = np.load(cache, allow_pickle=False)
        assert array.shape == (512, 64, 64, 3) and array.dtype == np.uint8
        reconstructed = timestamps((cache.parent / "decoder.log").read_text(encoding="utf-8", errors="replace"))
        for a, b, image in zip(reconstructed, c["frames"], array, strict=True):
            assert all(a[k] == b[k] for k in a)
            assert hashlib.sha256(image.tobytes()).hexdigest() == b["rgb64_sha256"]
            pixel_checks += 1
        attrs = read(Path(c["keyframe_path"]).parents[3] / "labels/train" / f"{identifier}.json")["attributes"]
        assert attrs == c["keyframe_attributes"]
    collection = read(out / "collection.json")
    rows, trainings = collection["records"], collection["trainings"]
    assert collection["complete"] and len(rows) == 180 and len(trainings) == 30
    assert len({(r["stream"], r["seed"], r["budget"]) for r in rows}) == 180
    assert sum(r["updates"] for r in trainings) == 7488
    replay = read(out / "training-replay-audit.json")
    assert replay["passed"] and replay["audit_ssl_updates"] == 8064 and replay["hold_counterfactuals"] == 180
    assert replay["new_checkpoints_replayed"] == 102 and replay["original_development96_states_reproduced"] == 12
    choice = read(out / "fixed-budget-selection.json")
    assert choice["locked_inputs_sha256"] == digest(out / "input_lock.json")
    assert all(r["started_unix"] >= choice["frozen_unix"] for r in rows if r["role"] == "heldout")
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
        assert len(labels) == 400 and all(((labels == c) & (domains == d)).sum() == 100 for c in (0, 1) for d in (0, 1))
        assert not set(a["query"]) & {int(v) for d in DRAWS for v in a[f"support_{d}"]}
    values = {}
    for row in rows:
        directory = out / row["directory"]
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
        assert (
            state["stop_updates"] == row["budget"]
            and state["stop_arrivals"] == row["budget"] * 32
            and state["terminal_arrivals"] == 12288
        )
        assert all(int(v["step"]) == row["budget"] for v in state["learner"]["optimizer"]["state"].values())
        if row["budget"] == 0:
            old = PRIOR / "warm" / f"seed_{row['seed']}"
            assert fingerprint(state["learner"]) == fingerprint(
                torch.load(old / "state.pt", map_location="cpu", weights_only=True)
            )
            assert digest(directory / "predictions.npz") == digest(old / "predictions.npz")
        elif row["role"] == "development" and row["budget"] != 48:
            old_name = row["stream"].removeprefix("dev_")
            old = PRIOR / "runs" / old_name / "1111" / f"seed_{row['seed']}" / f"at_{32 * row['budget']}"
            assert fingerprint(state["learner"]) == fingerprint(
                torch.load(old / "state.pt", map_location="cpu", weights_only=True)["learner"]
            )
            assert digest(directory / "predictions.npz") == digest(old / "predictions.npz")
        readout = read(directory / "readout.json")
        assert readout["ssl_state_unchanged"] and all(v["steps"] == 100 for v in readout["fine_tuning_audits"].values())
        with np.load(directory / "predictions.npz", allow_pickle=False) as p:
            assert set(p.files) == {f"{d}_{r}" for d in DRAWS for r in READERS}
            values[row["stream"], row["seed"], row["budget"]] = {
                r: {
                    m: 100 * sum(int(((p[f"{d}_{r}"] == labels) & mask).sum()) for d in DRAWS) / (3 * int(mask.sum()))
                    for m, mask in {"balanced": np.ones(400, bool), "A": domains == 0, "B": domains == 1}.items()
                }
                for r in READERS
            }
    dev_names = sorted({r["stream"] for r in rows if r["role"] == "development"})
    names = sorted({r["stream"] for r in rows if r["role"] == "heldout"})

    def metric(n, k, reader, m):
        return statistics.fmean(values[n, seed, k][reader][m] for seed in SEEDS)

    dev_means = {k: statistics.fmean(metric(n, k, "finetune", "balanced") for n in dev_names) for k in BUDGETS}
    chosen = min(BUDGETS, key=lambda k: (-round(dev_means[k], 12), k))
    assert chosen == choice["budget"]
    assert all(abs(dev_means[k] - choice["development_terminal_means"][str(k)]) < 1e-10 for k in BUDGETS)
    summary = read(out / "summary.json")
    assert summary["selected_budget"] == chosen
    checks = 0
    for ref in (0, 192, 384):
        for reader in READERS:
            for level, family in (("ordinary", 1), ("adjusted", 9)):
                for m in METRICS:
                    delta = [metric(n, chosen, reader, m) - metric(n, ref, reader, m) for n in names]
                    mean = statistics.fmean(delta)
                    width = float(t.ppf(1 - 0.05 / family / 2, 5)) * statistics.stdev(delta) / 6**0.5
                    stored = summary["selected_vs_references"][str(ref)][reader][level][m]
                    assert abs(stored["mean_pp"] - mean) < 1e-10
                    assert np.allclose(stored["ci95_pp"], [mean - width, mean + width], atol=1e-10)
                    assert np.allclose(stored["source_stream_values_pp"], delta, atol=1e-10)
                    checks += 1
    for k in BUDGETS:
        for reader in READERS:
            for m in METRICS:
                assert (
                    abs(
                        summary["heldout_means"][str(k)][reader][m]
                        - statistics.fmean(metric(n, k, reader, m) for n in names)
                    )
                    < 1e-10
                )
        seconds = statistics.fmean(r["processing_seconds"] for r in rows if r["role"] == "heldout" and r["budget"] == k)
        assert abs(summary["prepared_arrival_seconds"][str(k)] - seconds) < 1e-10
    for n in names:
        winner = min(BUDGETS, key=lambda k: (-round(metric(n, k, "finetune", "balanced"), 12), k))
        assert winner == summary["optimistic_budget_reference"]["winners"][n]
        gap = metric(n, winner, "finetune", "balanced") - metric(n, chosen, "finetune", "balanced")
        assert abs(summary["optimistic_budget_reference"]["gaps_pp"][n] - gap) < 1e-10
        shortfall = {
            m: metric(n, chosen, "finetune", m) - metric(n, 384, "finetune", m)
            for m in METRICS
            if round(metric(n, chosen, "finetune", m) - metric(n, 384, "finetune", m), 12) < -1
        }
        assert set(shortfall) == set(summary["diagnostic_shortfalls_vs_full"][n])
        for k in BUDGETS:
            for m in METRICS:
                assert abs(summary["per_source_primary_curves"][n][str(k)][m] - metric(n, k, "finetune", m)) < 1e-10
    effects = {ref: summary["selected_vs_references"][str(ref)]["finetune"]["adjusted"] for ref in (0, 192, 384)}
    saving = 1 - summary["prepared_arrival_seconds"][str(chosen)] / summary["prepared_arrival_seconds"]["384"]
    assert abs(saving - summary["observed_selected_savings_fraction"]) < 1e-10
    positive = effects[0]["balanced"]["mean_pp"] >= 0.5 and effects[0]["balanced"]["ci95_pp"][0] > 0
    full_ni = all(effects[384][m]["ci95_pp"][0] > -1 for m in METRICS)
    prior = (
        all(effects[192][m]["ci95_pp"][0] > -1 for m in METRICS)
        if chosen <= 192
        else effects[192]["balanced"]["mean_pp"] >= 0.5 and effects[192]["balanced"]["ci95_pp"][0] > 0
    )
    assert summary["transfer_gates"] == {
        "positive_ssl": positive,
        "full_noninferiority": full_ni,
        "prior_budget_comparison": prior,
        "work_saving": chosen < 384 and saving >= 0.25,
        "combined": bool(positive and full_ni and prior and chosen < 384 and saving >= 0.25),
    }
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "locked_hashes": len(lock["hashes"]),
            "source_frame_checks": pixel_checks,
            "stop_policy_states": 180,
            "prediction_sets_checked": 1620,
            "reader_metric_intervals": checks,
            "development_reused_states": 48,
            "new_prefix_states": 102,
            "new_ssl_updates": 7488,
            "verification_updates": 8064,
            "new_primary_fits": 306,
            "selected_budget": chosen,
        },
    )
    print(json.dumps(read(out / "independent-verification.json")), flush=True)


if __name__ == "__main__":
    main()
