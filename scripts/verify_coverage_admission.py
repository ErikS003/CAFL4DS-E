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
ARMS = ("early_stop", "uniform", "random", "coverage", "full")
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
    intake = read(ROOT / "outputs/coverage-admission/20261013-frames/intake.json")
    selected = read(ROOT / "outputs/coverage-admission/20261013-intake/selection.json")
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
        digest(ROOT / "outputs/coverage-admission/20261013-intake/selection.pool.json")
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
    rows = read(out / "collection.json")["records"]
    assert len(rows) == len({(r["stream"], r["seed"], r["arm"]) for r in rows}) == 150
    assert sum(r["updates"] for r in rows if not r["reused"]) == 29952
    audit = read(out / "training-replay-audit.json")
    assert audit["passed"] and audit["endpoint_states"] == 276 and audit["audit_ssl_updates"] == 29952
    with np.load(ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz", allow_pickle=False) as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
        assert len(labels) == 400 and all(((labels == c) & (domains == d)).sum() == 100 for c in (0, 1) for d in (0, 1))
    values, prediction_dirs = {}, set()
    coverage_windows = 0
    for row in rows:
        base = out / row["directory"]
        if row["reused"]:
            old_name = row["stream"].removeprefix("dev_")
            old = PRIOR / "runs" / old_name / "1111" / f"seed_{row['seed']}"
            for clock in (6144, 12288):
                assert digest(base / f"at_{clock}/state.pt") == digest(old / f"at_{clock}/state.pt")
        else:
            training = read(base / "training.json")
            assert training["updates"] == (384 if row["arm"] == "full" else 192)
            if row["arm"] != "full":
                assert len(training["blocks"]) == 96
                random = torch.Generator().manual_seed(row["seed"] + 62003)
                history = []
                recent = []
                for k, block in enumerate(training["blocks"]):
                    assert block["start"] == 128 * k and block["end"] == 128 * (k + 1)
                    if row["arm"] == "uniform":
                        selected_batches = [0, 2]
                    elif row["arm"] == "random":
                        selected_batches = sorted(torch.randperm(4, generator=random)[:2].tolist())
                    elif row["arm"] == "early_stop":
                        selected_batches = list(range(4)) if k < 48 else []
                    else:
                        features = np.array(block["centroids"], dtype=np.float64)
                        features /= np.sqrt(np.sum(features**2, axis=1))[:, None]
                        temporary = list(history)
                        chosen = []
                        for i in range(2):
                            distances = [
                                min(max(0.0, min(2.0, 1 - float(np.sum(v * h)))) for h in temporary)
                                if temporary
                                else 0.0
                                for v in features
                            ]
                            assert np.allclose(distances, block["novelty_rounds"][i], atol=1e-12, rtol=0)
                            selected = min((i for i in range(4) if i not in chosen), key=lambda i: (-distances[i], i))
                            chosen.append(selected)
                            temporary = (temporary + [features[selected]])[-32:]
                        selected_batches = sorted(chosen)
                        history = (history + [features[i] for i in selected_batches])[-32:]
                        coverage_windows += 1
                    assert block["selected_batches"] == selected_batches
                    assert block["exposure_indices"] == [
                        list(range(128 * k + 32 * i, 128 * k + 32 * (i + 1))) for i in selected_batches
                    ]
                    expected_prior = statistics.fmean(recent[-4:]) if recent else None
                    assert block["prior_training_loss"] == expected_prior
                    if expected_prior is not None:
                        assert np.allclose(
                            block["relative_losses"], [v - expected_prior for v in block["incoming_losses"]], atol=1e-12
                        )
                    recent.extend(v[0] for v in block["loss_gradients"])
                    if row["arm"] == "coverage" and k in (47, 95):
                        stored = torch.load(
                            base / f"at_{128 * (k + 1)}/state.pt", map_location="cpu", weights_only=True
                        )
                        assert np.allclose(stored["memory"], history, atol=1e-12, rtol=0)
        previous = None
        for clock, endpoint in zip((6144, 12288), row["endpoints"], strict=True):
            path = out / endpoint
            state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
            expected = (
                clock // 32
                if row["arm"] == "full"
                else min(clock // 32, 192)
                if row["arm"] == "early_stop"
                else clock // 64
            )
            assert state["updates"] == expected
            assert all(int(v["step"]) == expected for v in state["learner"]["optimizer"]["state"].values())
            if previous is not None and row["arm"] == "early_stop":
                assert fingerprint(previous["learner"]) == fingerprint(state["learner"])
            previous = state
            if (path / "readout-alias.json").exists():
                path = out / read(path / "readout-alias.json")["directory"]
            prediction_dirs.add(str(path))
            with np.load(path / "predictions.npz", allow_pickle=False) as p:
                assert set(p.files) == {f"{d}_{r}" for d in DRAWS for r in READERS}
                if clock == 12288:
                    values[row["stream"], row["seed"], row["arm"]] = {
                        r: {
                            m: 100
                            * sum(int(((p[f"{d}_{r}"] == labels) & mask).sum()) for d in DRAWS)
                            / (3 * int(mask.sum()))
                            for m, mask in {
                                "balanced": np.ones(400, bool),
                                "A": domains == 0,
                                "B": domains == 1,
                            }.items()
                        }
                        for r in READERS
                    }
            assert read(path / "readout.json")["ssl_state_unchanged"]
    assert len(prediction_dirs) == 270 and coverage_windows == 2880
    summary = read(out / "summary.json")
    warm_values = []
    for seed in SEEDS:
        path = out / "warm" / f"seed_{seed}"
        with np.load(path / "predictions.npz", allow_pickle=False) as p:
            warm_values.append(100 * sum(int((p[f"{d}_finetune"] == labels).sum()) for d in DRAWS) / 1200)
    names = sorted({r["stream"] for r in rows if r["role"] == "heldout"})

    def metric(n, arm, reader, m):
        return statistics.fmean(values[n, seed, arm][reader][m] for seed in SEEDS)

    checks = 0
    for ref in ("uniform", "random", "early_stop", "full"):
        for reader in READERS:
            for level, family in (("ordinary", 1), ("adjusted", 12)):
                for m in METRICS:
                    delta = [metric(n, "coverage", reader, m) - metric(n, ref, reader, m) for n in names]
                    mean = statistics.fmean(delta)
                    width = float(t.ppf(1 - 0.05 / family / 2, 5)) * statistics.stdev(delta) / 6**0.5
                    stored = summary["coverage_vs_references"][ref][reader][level][m]
                    assert abs(stored["mean_pp"] - mean) < 1e-10
                    assert np.allclose(stored["ci95_pp"], [mean - width, mean + width], atol=1e-10)
                    assert np.allclose(stored["source_stream_values_pp"], delta, atol=1e-10)
                    checks += 1
    for arm in ARMS:
        for reader in READERS:
            for m in METRICS:
                assert (
                    abs(
                        summary["heldout_means"][arm][reader][m]
                        - statistics.fmean(metric(n, arm, reader, m) for n in names)
                    )
                    < 1e-10
                )
        seconds = statistics.fmean(r["processing_seconds"] for r in rows if r["role"] == "heldout" and r["arm"] == arm)
        assert abs(seconds - summary["processing_seconds"][arm]) < 1e-10
    superiority = all(
        summary["coverage_vs_references"][r]["finetune"]["adjusted"]["balanced"]["mean_pp"] >= 0.5
        and summary["coverage_vs_references"][r]["finetune"]["adjusted"]["balanced"]["ci95_pp"][0] > 0
        for r in ("uniform", "random")
    )
    guards = all(
        summary["coverage_vs_references"][r]["finetune"]["adjusted"][m]["ci95_pp"][0] > -1
        for r in ("uniform", "random")
        for m in ("A", "B")
    ) and all(
        summary["coverage_vs_references"][r]["finetune"]["adjusted"][m]["ci95_pp"][0] > -1
        for r in ("early_stop", "full")
        for m in METRICS
    )
    saving = 1 - summary["processing_seconds"]["coverage"] / summary["processing_seconds"]["full"]
    assert abs(saving - summary["coverage_saving_fraction"]) < 1e-10
    assert summary["gates"]["beats_paced_controls"] == superiority and summary["gates"]["quality_guards"] == guards
    assert summary["gates"]["cost_saving"] == (saving >= 0.25)
    delta = [metric(n, "coverage", "finetune", "balanced") - statistics.fmean(warm_values) for n in names]
    mean = statistics.fmean(delta)
    half = float(t.ppf(0.975, 5)) * statistics.stdev(delta) / 6**0.5
    assert np.allclose(summary["coverage_vs_frozen_context"]["ci95_pp"], [mean - half, mean + half], atol=1e-10)
    assert summary["gates"]["positive_frozen_context"] == (mean - half > 0)
    assert not summary["controller_fit"]
    atomic_json(
        out / "independent-verification.json",
        {
            "passed": True,
            "locked_hashes": len(lock["hashes"]),
            "source_frames": pixel_checks,
            "endpoint_states_checked": 300,
            "new_endpoint_states": 276,
            "prediction_sets_checked": 2430,
            "coverage_windows": coverage_windows,
            "reader_metric_intervals": checks,
            "scientific_ssl_updates": 29952,
            "audit_ssl_updates": 29952,
            "new_primary_fits": 738,
            "reused_primary_fits": 81,
        },
    )
    print(json.dumps(read(out / "independent-verification.json")), flush=True)


if __name__ == "__main__":
    main()
