"""Independent crossed-source, complete-state, observer and block interaction checks."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import t

from cafl4ds.short_buffer import capture, restore
from cafl4ds.state_horizon import observe_before_action
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/regime-windows/20261013-gpu"
from cafl4ds.history_order import CASES, STATES, BLOCKS

SEEDS = (5003, 5009, 5011)
CLOCKS = (1024,)
DRAWS = (51011, 51017, 51023)
READERS = ("finetune", "linear", "knn")
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
    prior_verification = json.loads((PRIOR / "independent-verification.json").read_text())
    assert prior_verification["passed"]
    role = json.loads((PRIOR / "data-role-manifest.json").read_text())
    frames = []
    source_checks = 0
    for clip in role["clips"]:
        array = np.load(clip["frames_path"], allow_pickle=False)
        assert array.shape in ((512, 64, 64, 3), (528, 64, 64, 3)) and array.dtype == np.uint8
        last = -1
        for k, (metadata, image) in enumerate(zip(clip["frames"], array, strict=True)):
            assert metadata["pts_seconds"] > last and metadata["pts_seconds"] >= k / 16
            assert hashlib.sha256(image.tobytes()).hexdigest() == metadata["rgb64_sha256"]
            last = metadata["pts_seconds"]
            source_checks += 1
        frames.append(array)
    arrivals = torch.from_numpy(np.concatenate(frames)).permute(0, 3, 1, 2).float() / 255
    intake = json.loads((ROOT / "outputs/history-order/20261013-frames/intake.json").read_text())
    selection = json.loads((ROOT / "outputs/history-order/20261013-intake/selection.json").read_text())
    from scripts.decode_regime_video import timestamps

    assert len(selection["clips"]) == 24 and not (
        {c["clip_id"] for c in selection["clips"]} & set(selection["excluded_clip_ids"])
    )
    pool_path = ROOT / "outputs/history-order/20261013-intake/selection.pool.json"
    assert digest(pool_path) == selection["eligible_pool_sha256"]
    pool = json.loads(pool_path.read_text())
    eligible = {c["clip_id"] for group in pool.values() for c in group}
    assert {c["clip_id"] for c in selection["clips"]} <= eligible
    secondary = json.loads((ROOT / "outputs/cross-state/20261013-frames/intake.json").read_text())
    intake["blocks"] += [b for b in secondary["blocks"] if b["context"] == "night"]
    streams = {}
    new_frame_checks = 0
    all_new = []
    for block in intake["blocks"]:
        values = []
        expected_scenes = ["city street", "highway"] if block["block"] % 2 == 0 else ["highway", "city street"]
        assert [c["keyframe_attributes"]["scene"] for c in block["clips"]] == expected_scenes
        for clip in block["clips"]:
            assert clip["keyframe_attributes"]["timeofday"] == ("daytime" if block["context"] == "day" else "night")
            assert clip["keyframe_attributes"]["weather"] == "clear"
            cache = Path(clip["frames_path"])
            array = np.load(cache, allow_pickle=False)
            reconstructed = timestamps((cache.parent / "decoder.log").read_text(errors="replace"), count=len(array))
            assert array.shape in ((512, 64, 64, 3), (528, 64, 64, 3))
            for decoded, stored, image in zip(reconstructed, clip["frames"], array, strict=True):
                assert all(decoded[k] == stored[k] for k in decoded)
                assert hashlib.sha256(image.tobytes()).hexdigest() == stored["rgb64_sha256"]
                new_frame_checks += 1
            attributes = json.loads(
                (Path(clip["keyframe_path"]).parents[3] / "labels/train" / f"{clip['clip_id']}.json").read_text()
            )["attributes"]
            assert attributes == clip["keyframe_attributes"]
            values.append(array[:512])
            all_new.append(clip["clip_id"])
        streams[block["context"], block["block"]] = (
            torch.from_numpy(np.concatenate(values)).permute(0, 3, 1, 2).float() / 255
        )
    assert len(all_new) == len(set(all_new)) == 34 and not set(all_new) & {c["clip_id"] for c in role["clips"]}
    rows = json.loads((out / "collection.json").read_text())["records"]
    assert len(rows) == 204 and sum(r["updates"] for r in rows) == 3264
    assert len({(r["state"], r["seed"], r["action"]) for r in rows}) == 204
    with np.load(PRIOR / "evaluation_arrays.npz") as a:
        labels = a["labels"][a["query"]]
        domains = a["domains"][a["query"]]
        assert len(labels) == 400
        assert all(((labels == c) & (domains == d)).sum() == 100 for c in (0, 1) for d in (0, 1))
        assert all(len(a[f"support_{d}"]) == 50 for d in DRAWS)
    counts, hashes = {}, {}

    def score(name, seed, clock, directory):
        with np.load(directory / "predictions.npz") as predictions:
            panels = json.loads((directory / "readout.json").read_text())["panels"]
            for reader in READERS:
                for metric in METRICS:
                    mask = (
                        np.ones(len(labels), bool) if metric == "balanced" else domains == (0 if metric == "A" else 1)
                    )
                    total = 0
                    for draw in DRAWS:
                        correct = predictions[f"{draw}_{reader}"] == labels
                        actual = int((correct & mask).sum())
                        assert abs(panels[str(draw)][reader][metric] - actual / int(mask.sum())) < 1e-6
                        total += actual
                    counts[name, seed, clock, reader, metric] = total

    for name, parent_clock in STATES.items():
        for seed in SEEDS:
            parent_dir = out / "parents" / CASES[name]["parent"] / f"seed_{seed}"
            if CASES[name]["parent"] == "DN":
                original_dir = PRIOR / "runs/w32/full" / f"seed_{seed}" / "at_8448"
                with np.load(parent_dir / "predictions.npz") as a, np.load(original_dir / "predictions.npz") as b:
                    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
            score(name, seed, 0, parent_dir)
    for row in rows:
        name, seed, action = row["state"], row["seed"], row["action"]
        parent_clock = STATES[name]
        directory = out / row["directory"]
        trace = json.loads((directory / "training.json").read_text())
        assert len(trace["blocks"]) == 32
        exposed = []
        for i, block in enumerate(trace["blocks"]):
            assert block["start"] == 32 * i and block["end"] == 32 * (i + 1)
            expected = list(range(32 * i, 32 * (i + 1))) if action == "full" else []
            assert block["selected_indices"] == expected
            assert block["exposure_indices"] == ([expected] if action == "full" else [])
            assert block["updates"] == (1 if action == "full" else 0)
            exposed.extend(expected)
        assert exposed == (list(range(1024)) if action == "full" else [])
        original = torch.load(
            out / "parents" / CASES[name]["parent"] / f"seed_{seed}/state.pt", map_location="cpu", weights_only=True
        )
        for clock in CLOCKS:
            checkpoint = directory / f"at_{clock}"
            saved = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
            updates = clock // 32 if action == "full" else 0
            assert saved["updates"] == updates and saved["total_updates"] == parent_clock // 32 + updates
            assert saved["parent_arrivals"] == parent_clock and saved["arrivals"] == clock
            assert not saved["pending_indices"] and not saved["queued_indices"]
            assert {int(v["step"]) for v in saved["learner"]["optimizer"]["state"].values()} == {
                parent_clock // 32 + updates
            }
            if action == "hold":
                assert fingerprint(saved["learner"]) == fingerprint(original["learner"])
                alias = json.loads((checkpoint / "readout-alias.json").read_text())
                assert alias["directory"].replace("\\", "/") == f"parents/{CASES[name]['parent']}/seed_{seed}"
            else:
                score(name, seed, clock, checkpoint)
            hashes[row["directory"] + f"/at_{clock}"] = {"state": digest(checkpoint / "state.pt")}
    summary = json.loads((out / "summary.json").read_text())

    def block_delta(parent, context, k, reader, metric):
        name = f"{parent}_{context}_b{k}"
        denominator = 1200 if metric == "balanced" else 600
        return (
            100
            * sum(counts[name, s, 1024, reader, metric] - counts[name, s, 0, reader, metric] for s in SEEDS)
            / (3 * denominator)
        )

    def check_interval(saved, values, level):
        mean = statistics.fmean(values)
        alpha = 0.05 if level == "unadjusted" else 0.05 / 6
        half = float(t.ppf(1 - alpha / 2, len(values) - 1)) * statistics.stdev(values) / np.sqrt(len(values))
        assert abs(saved["mean_pp"] - mean) < 1e-10
        assert np.allclose(saved["ci95_pp"], [mean - half, mean + half], atol=1e-10)
        assert np.allclose(saved["source_block_values_pp"], values, atol=1e-10)

    interval_checks = 0
    for parent in ("DN", "ND"):
        for context in ("day", "night"):
            cell = summary["cells"][f"{parent}_{context}"]
            for reader in READERS:
                for metric in METRICS:
                    denominator = 1200 if metric == "balanced" else 600
                    for action in ("hold", "full"):
                        index = 0 if action == "hold" else 1024
                        mean = (
                            100
                            * sum(
                                counts[f"{parent}_{context}_b{k}", s, index, reader, metric]
                                for k in range(BLOCKS[context])
                                for s in SEEDS
                            )
                            / (3 * BLOCKS[context] * denominator)
                        )
                        assert abs(cell["means"][reader][action][metric] - mean) < 1e-10
                    values = [block_delta(parent, context, k, reader, metric) for k in range(BLOCKS[context])]
                    for level in ("unadjusted", "guard_adjusted"):
                        check_interval(cell["contrasts"][reader][level][metric], values, level)
                        interval_checks += 1
            for level in ("unadjusted", "guard_adjusted"):
                contrasts = cell["contrasts"]["finetune"][level]
                expected = {
                    "full": contrasts["balanced"]["ci95_pp"][0] > 0
                    and contrasts["balanced"]["mean_pp"] >= 0.1
                    and all(contrasts[m]["ci95_pp"][0] > -1 for m in ("A", "B")),
                    "hold": contrasts["balanced"]["ci95_pp"][1] < 0
                    and contrasts["balanced"]["mean_pp"] <= -0.1
                    and all(contrasts[m]["ci95_pp"][1] < 1 for m in ("A", "B")),
                }
                assert cell["preferred"][level] == expected
    for context in ("day", "night"):
        for reader in READERS:
            for metric in METRICS:
                values = [
                    block_delta("ND", context, k, reader, metric) - block_delta("DN", context, k, reader, metric)
                    for k in range(BLOCKS[context])
                ]
                for level in ("unadjusted", "guard_adjusted"):
                    check_interval(
                        summary["ND_minus_DN_action_response"][context][reader][level][metric], values, level
                    )
                    interval_checks += 1
    primary = summary["ND_minus_DN_action_response"]["day"]["finetune"]["unadjusted"]["balanced"]
    qualified = (primary["ci95_pp"][0] > 0 or primary["ci95_pp"][1] < 0) and abs(primary["mean_pp"]) >= 0.5
    assert summary["primary_history_effect_qualified"] == qualified
    dn = summary["cells"]["DN_day"]["preferred"]["guard_adjusted"]
    nd = summary["cells"]["ND_day"]["preferred"]["guard_adjusted"]
    flip = qualified and ((dn["full"] and nd["hold"]) or (dn["hold"] and nd["full"]))
    assert summary["same_footage_state_preference_flip"] == {"day": bool(flip), "night": False}
    assert summary["decision"] == (
        "matched_age_preference_flip_qualified"
        if flip
        else "history_changes_action_response"
        if qualified
        else "no_matched_age_history_effect_qualified"
    )
    indices = json.loads((out / "history-indices.json").read_text())
    assert indices == {"DN": list(range(8448)), "ND": list(range(4224, 8448)) + list(range(4224))}
    assert sorted(indices["DN"]) == sorted(indices["ND"])
    histories = {order: arrivals[ids] for order, ids in indices.items()}
    parents = json.loads((out / "parent-collection.json").read_text())["records"]
    assert len(parents) == 6 and sum(r["updates"] for r in parents) == 1584
    for row in parents:
        directory = out / row["directory"]
        trace = json.loads((directory / "training.json").read_text())
        assert trace["updates"] == 264 and trace["accepted_frames"] == 8448
        exposed = [i for b in trace["blocks"] for batch in b["exposure_indices"] for i in batch]
        assert exposed == list(range(8448))
        saved = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
        assert {int(v["step"]) for v in saved["learner"]["optimizer"]["state"].values()} == {264}
        assert saved["updates"] == 264 and not saved["queued_indices"]
        warm = torch.load(out / "warm-parents" / f"seed_{row['seed']}/state.pt", map_location="cpu", weights_only=True)
        assert not warm["optimizer"]["state"]
        hashes[row["directory"]] = {
            "state": digest(directory / "state.pt"),
            "predictions": digest(directory / "predictions.npz"),
            "warm_initial": digest(out / "warm-parents" / f"seed_{row['seed']}/state.pt"),
        }
        if row["order"] == "DN":
            old = torch.load(
                PRIOR / "runs/w32/full" / f"seed_{row['seed']}" / "at_8448/state.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert fingerprint(saved["learner"]) == fingerprint(old["learner"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = _method(OmegaConf.create(lock["config"])).cuda()
    observer_checks = 0
    for name, parent_clock in STATES.items():
        for seed in SEEDS:
            order = CASES[name]["parent"]
            original = torch.load(
                out / "parents" / order / f"seed_{seed}/state.pt", map_location="cpu", weights_only=True
            )
            model, optim = restore(initial, OmegaConf.create(lock["config"]["optim"]), original["learner"])
            trace = json.loads((out / "parents" / order / f"seed_{seed}/training.json").read_text())["blocks"]
            recent = statistics.fmean([b["loss_gradients"][0][0] for b in trace][-4:])
            before = fingerprint(capture(model, optim))
            observation = observe_before_action(
                model,
                histories[order][-128:],
                streams[CASES[name]["context"], CASES[name]["block"]][:32],
                recent,
                seed,
            )
            saved = json.loads((out / "observations" / name / f"seed_{seed}.json").read_text())
            assert fingerprint(capture(model, optim)) == before
            assert {k: v for k, v in observation.items() if k != "observer_seconds"} == {
                k: v for k, v in saved.items() if k != "observer_seconds"
            }
            observer_checks += 1
    replay = json.loads((out / "training-replay-audit.json").read_text())
    assert replay["passed"] and replay["audit_ssl_updates"] == 4848 and replay["checkpoints"] == 204
    payload = {
        "passed": True,
        "locked_hashes": len(lock["hashes"]),
        "source_frames_checked": source_checks,
        "new_source_frames_checked": new_frame_checks,
        "shared_raw_continuations_verified": True,
        "branches": 204,
        "checkpoints": 204,
        "new_prediction_sets": 972,
        "paired_intervals_checked": interval_checks,
        "observers_reproduced_and_isolated": observer_checks,
        "same_order_parent_readouts_exact": True,
        "history_multisets_update_clocks_and_original_DN_checked": True,
        "parent_checkpoints": 6,
        "all_hold_states_invariant": True,
        "all_gates_checked": True,
        "verification_source_sha256": digest(Path(__file__)),
        "artifact_hashes": hashes,
    }
    (out / "independent-verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload | {"artifact_hashes": "archived204 branch and6 parent checkpoint hashes"}))


if __name__ == "__main__":
    main()
