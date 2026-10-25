"""Independently reconstruct reservoir forks, common continuations and action effects."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.filters.reservoir_health_recovery import RecoveryReservoir
from cafl4ds.health_calibration import embeddings
from cafl4ds.historical_loss_screen import bank_losses
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.reservoir_action_screen import opportunity_screen
from cafl4ds.short_buffer import capture, restore
from scripts.natural_health_view import SIGNALS, frozen_predictions, write
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read
from scripts.verify_historical_loss_screen import shadow_audit
from scripts.verify_memory_age_replay import health


class FeatureArrays:
    """Expose a small in-memory bank mapping to the inherited independent geometry audit."""

    def __init__(self, values):
        """Retain the reference and the one post-action feature matrix."""
        self.values, self.files = values, list(values)

    def __getitem__(self, key):
        """Read one declared feature matrix."""
        return self.values[key]


def primitive_diet(trace, stream, parent=None):
    """Rebuild Algorithm R and all samples from primitive slots and private generators."""
    sample = torch.Generator().manual_seed(73919)
    admit = torch.Generator().manual_seed(90101)
    ids, pixels, seen = [], [], 0
    if parent is not None:
        state = parent["memory"]
        ids, seen = list(state["ids"]), state["seen"]
        pixels = [p.numpy().copy() for p in state["images"]]
        sample.set_state(state["sample_rng"])
        admit.set_state(state["admission_rng"])
    ages, current_total, past_total = [], 0, 0
    for row in trace:
        step, count = row["step"], row["past"]
        assert step == seen // 32 + 1 and row["current"] == 32 - count
        incoming = stream.batch(step - 1)
        assert hashlib.sha256(incoming.tobytes()).hexdigest() == row["input_sha256"]
        old = torch.randperm(len(ids), generator=sample)[:count].tolist() if count else []
        now = torch.randperm(32, generator=sample)[: 32 - count].sort().values.tolist() if count else list(range(32))
        old_ids, now_ids = [ids[i] for i in old], [seen + i for i in now]
        assert old_ids == row["diet"]["replay_ids"] and now_ids == row["diet"]["current_ids"]
        assert all(i < seen for i in old_ids)
        mixed = np.concatenate((incoming[now], np.stack([pixels[i] for i in old]))) if count else incoming
        assert hashlib.sha256(mixed.tobytes()).hexdigest() == row["train_sha256"]
        age = [step - 1 - i // 32 for i in old_ids]
        assert age == row["diet"]["replay_arrival_ages"]
        ages.extend(age)
        replaced = 0
        for image in incoming:
            if seen < 256:
                ids.append(seen)
                pixels.append(image.copy())
                replaced += 1
            else:
                slot = int(torch.randint(seen + 1, (1,), generator=admit).item())
                if slot < 256:
                    ids[slot], pixels[slot] = seen, image.copy()
                    replaced += 1
            seen += 1
        assert row["diet"]["admitted_frames_seen"] == seen
        assert row["diet"]["resident_frames"] == min(seen, 256)
        assert row["diet"]["oldest_resident_id"] == min(ids)
        assert row["diet"]["retained_arrivals"] == replaced
        current_total += 32 - count
        past_total += count
    return {
        "ids": ids,
        "images": np.stack(pixels),
        "seen": seen,
        "sample_rng": sample.get_state(),
        "admission_rng": admit.get_state(),
        "hashes": [hashlib.sha256(p.tobytes()).hexdigest() for p in pixels],
        "diet": {
            "current_exposures": current_total,
            "replay_exposures": past_total,
            "mean_replay_age": float(np.mean(ages)),
            "max_replay_age": max(ages),
        },
    }


def checkpoint(path, expected, initial, optim, banks, raw, prefix):
    """Restore actual pixels, parameters, optimizer/RNG and reproduce saved bank arrays."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    memory = RecoveryReservoir()
    memory.load_state_dict(state["memory"])
    for key in ("ids", "seen", "hashes"):
        assert state["memory"][key] == expected[key]
    for key in ("sample_rng", "admission_rng"):
        assert torch.equal(state["memory"][key], expected[key])
    np.testing.assert_array_equal(state["memory"]["images"], expected["images"])
    model, optimizer = restore(initial, optim, state["learner"])
    before = fingerprint(capture(model, optimizer))
    for name, images in banks.items():
        np.testing.assert_array_equal(embeddings(model, images), raw[f"{prefix}_{name}_features"])
        np.testing.assert_array_equal(bank_losses(model, images), raw[f"{prefix}_{name}_losses"])
    assert before == fingerprint(capture(model, optimizer))
    mark = fingerprint({"learner": state["learner"], "memory": memory.marker(), "shadow": state.get("shadow")})
    return mark


def independent_screen(values):
    """Cross-check both preference directions with explicit per-parent/per-draw loops."""
    result = opportunity_screen(values)
    robust_a, robust_b = [], []
    for parent in values:
        pooled = [sum(float(draw[d]) for draw in parent) / 3 for d in range(3)]
        a = pooled[0] >= 0.5 and min(pooled[1:]) >= -1
        b = pooled[0] <= -0.5 and max(pooled[1:]) <= 1
        robust_a.append(a and all(draw[0] > 0 and draw[1] >= -1 and draw[2] >= -1 for draw in parent))
        robust_b.append(b and all(draw[0] < 0 and draw[1] <= 1 and draw[2] <= 1 for draw in parent))
    np.testing.assert_array_equal(result["robust_recovery"], robust_a)
    np.testing.assert_array_equal(result["robust_base"], robust_b)
    assert result["passed"] == (sum(robust_a) >= 2 and sum(robust_b) >= 2)
    return result


def source_audit(config):
    """Reconstruct the reserved prefix and every decoded timestamp/image digest."""
    intake = read(ROOT / config.frames / "intake.json")
    selection = read(ROOT / config.intake / "selection.json")
    exclusion = read(ROOT / config.intake / "prior-exclusions.json")
    excluded = set(exclusion["ids"])
    prior_positions = []
    for manifest in exclusion["manifests"]:
        assert digest(manifest["path"]) == manifest["sha256"]
        assert set(manifest["ids"]) <= excluded
        clips = read(manifest["path"])["clips"]
        clips = clips.values() if isinstance(clips, dict) else clips
        prior_positions.extend(c["archive_position"] for c in clips if isinstance(c, dict) and "archive_position" in c)
    assert max(prior_positions) == exclusion["start_after_archive_position"]
    expected = []
    for position, entry in enumerate(read(ROOT / config.archive_index)["entries"]):
        clip = Path(entry["path"]).stem
        if position <= exclusion["start_after_archive_position"] or "/train/" not in entry["path"] or clip in excluded:
            continue
        expected.append((position, clip))
        if len(expected) == 96:
            break
    assert expected == [(r["archive_position"], r["clip_id"]) for r in selection["clips"]]
    assert [r["clip_id"] for r in intake["clips"]] == [r["clip_id"] for r in selection["clips"][: len(intake["clips"])]]
    assert not excluded.intersection(r["clip_id"] for r in intake["clips"])
    assert intake["batches"] == 1024 and sum(c["shape"][0] for c in intake["clips"]) == 32768
    for clip in intake["clips"]:
        data = np.load(clip["frames_path"], mmap_mode="r", allow_pickle=False)
        assert len(data) == len(clip["frames"])
        seconds = [r["pts_seconds"] for r in clip["frames"]]
        assert all(a < b for a, b in zip(seconds, seconds[1:], strict=False))
        for image, entry in zip(data, clip["frames"], strict=True):
            assert hashlib.sha256(image.tobytes()).hexdigest() == entry["rgb64_sha256"]
    return intake


def verify(config, out):
    """Audit the finite pilot and keep all reader/horizon/support outcomes visible."""
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    assert read(out / "training-verification.json")["passed"]
    intake = source_audit(config)
    with np.load(ROOT / config.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    cache = read(out / "readout_cache.json")
    new = {k: r for k, r in cache.items() if not r.get("inherited", False)}
    counts = {f"{k}_{s}": 0 for k in ("knn", "linear") for s in ("started", "completed")}
    for index, row in enumerate(new.values(), 1):
        assert digest(row["path"]) == row["sha256"]
        with np.load(row["path"]) as saved:
            repeated = frozen_predictions(
                saved["features"], arrays, config.draws, counts, out / "attempts/probe_verification.json"
            )
            for kind in ("knn", "linear"):
                np.testing.assert_array_equal(repeated[kind], saved[kind])
            np.testing.assert_array_equal(saved["finetune"], saved["probabilities"][:, :3663].argmax(-1))
            q = arrays["query"]
            for kind in ("knn", "linear", "finetune"):
                scores = []
                for pred in saved[kind]:
                    cells = [
                        100 * (pred[(arrays["domains"][q] == d) & (arrays["labels"][q] == c)] == c).mean()
                        for d in (0, 1)
                        for c in (0, 1)
                    ]
                    scores.append([np.mean(cells), np.mean(cells[:2]), np.mean(cells[2:])])
                np.testing.assert_allclose(scores, row["scores"][kind], rtol=0, atol=1e-12)
        if index % 16 == 0:
            print({"reader_states_verified": index, "total": len(new)}, flush=True)
    trunk, branches = read(out / "trunk.json"), read(out / "branches.json")
    stream = OrderedFrames(intake["clips"])
    assert len(trunk["trace"]) == 1024
    assert all(r["past"] == (0 if r["step"] == 1 else 8) for r in trunk["trace"])
    trunk_diet = primitive_diet(trunk["trace"], stream)
    method = OmegaConf.create(lock["model_config"])
    initial = _method(method).cuda()
    banks = {name: reference_images(lock[f"{name}_reference"]) for name in ("monitor", "audit")}
    rows, checks = [], []
    with np.load(out / "trunk-measurements.npz") as raw:
        shadow = shadow_audit(trunk, raw)
        health_summary = {}
        for name in banks:
            item = health(raw, name)
            item["loss"] = [float(raw[f"{p}_{name}_losses"].mean(dtype=np.float64)) for p in item["points"]]
            item["seed_losses"] = [raw[f"{p}_{name}_losses"].mean(1, dtype=np.float64) for p in item["points"]]
            ref = float(raw[f"132_{name}_losses"].mean(dtype=np.float64))
            item["reference_loss"] = ref
            item["risk"] = [
                v / ref - 1 if p >= 132 else None for p, v in zip(item["points"], item["loss"], strict=True)
            ]
            health_summary[name] = item
        assert (
            checkpoint(out / "trunk-final.pt", trunk_diet, initial, method.optim, banks, raw, "1024")
            == trunk["final_fingerprint"]
        )
        checks.append("trunk")
        for p in config.landmarks:
            point = int(p)
            parent_path = Path(trunk["parents"][str(point)]["path"])
            parent = torch.load(parent_path, weights_only=True, map_location="cpu")
            expected = primitive_diet(trunk["trace"][:point], stream)
            assert (
                checkpoint(parent_path, expected, initial, method.optim, banks, raw, str(point))
                == trunk["parents"][str(point)]["fingerprint"]
            )
            checks.append(f"parent{point}")
            pair = {arm: branches[f"p{point:04d}_{arm}"] for arm in ("base", "recovery")}
            row = {
                "parent": point,
                "effects": {},
                "ssl_effects": {},
                "health": {},
                "diet": {},
                "parent_loss": {name: float(raw[f"{point}_{name}_losses"].mean(dtype=np.float64)) for name in banks},
            }
            for arm, record in pair.items():
                assert (
                    record["parent"] == point
                    and record["parent_fingerprint"] == trunk["parents"][str(point)]["fingerprint"]
                )
                assert len(record["trace"]) == 128 and record["reset"] is None
                for offset, t in enumerate(record["trace"], 1):
                    want = 8 if arm == "base" or offset > 64 else (24 if offset <= 32 else 16)
                    assert t["offset"] == offset and t["step"] == point + offset and t["past"] == want
                    if arm == "base":
                        assert all(t[k] == v for k, v in trunk["trace"][point + offset - 1].items())
                expected = primitive_diet(record["trace"], stream, parent)
                row["diet"][arm] = expected["diet"]
                assert expected["diet"]["replay_exposures"] == (1024 if arm == "base" else 1792)
                with np.load(out / "branches" / f"p{point:04d}_{arm}" / "measurements.npz") as branch_raw:
                    assert (
                        checkpoint(record["state_path"], expected, initial, method.optim, banks, branch_raw, "128")
                        == record["final_fingerprint"]
                    )
                    for horizon in (32, 64, 128):
                        h = str(horizon)
                        row["ssl_effects"].setdefault(h, {})[arm] = {
                            name: float(branch_raw[f"{h}_{name}_losses"].mean(dtype=np.float64)) for name in banks
                        }
                        row["health"].setdefault(h, {})[arm] = {}
                        for name in banks:
                            values = {
                                f"132_{name}_features": raw[f"132_{name}_features"],
                                f"{point + horizon}_{name}_features": branch_raw[f"{h}_{name}_features"],
                            }
                            computed = health(FeatureArrays(values), name)
                            row["health"][h][arm][name] = {key: computed["values"][key][-1] for key in SIGNALS}
                final = torch.load(record["state_path"], weights_only=True, map_location="cpu")
                memory = RecoveryReservoir()
                memory.load_state_dict(final["memory"])
                assert fingerprint(memory.marker()) == record["memory_fingerprint"]
                checks.append(f"{point}_{arm}")
                for observation in record["observations"].values():
                    assert observation["scores"] == cache[observation["encoder_fingerprint"]]["scores"]
            assert pair["base"]["memory_fingerprint"] == pair["recovery"]["memory_fingerprint"]
            for x, y in zip(pair["base"]["trace"][64:], pair["recovery"]["trace"][64:], strict=True):
                assert x["diet"] == y["diet"] and x["train_sha256"] == y["train_sha256"]
            for horizon in ("32", "64", "128"):
                assert (
                    pair["base"]["observations"][horizon]["rng_fingerprint"]
                    == pair["recovery"]["observations"][horizon]["rng_fingerprint"]
                )
                assert (
                    pair["base"]["observations"][horizon]["learner_fingerprint"]
                    == trunk["monitors"][str(point + int(horizon))]["learner_fingerprint"]
                )
                row["effects"][horizon] = {}
                for kind in ("knn", "linear", "finetune"):
                    a = np.asarray(pair["base"]["observations"][horizon]["scores"][kind])
                    b = np.asarray(pair["recovery"]["observations"][horizon]["scores"][kind])
                    delta = b - a
                    row["effects"][horizon][kind] = {
                        "per_draw_effects": delta,
                        "pooled_effects": delta.mean(0),
                        "per_draw_utility": delta[:, 1:].min(1),
                        "pooled_utility": delta.mean(0)[1:].min(),
                    }
            rows.append(row)
            print(
                {
                    "paired_parent_verified": point,
                    "terminal_effect": row["effects"]["128"]["finetune"]["pooled_effects"].tolist(),
                },
                flush=True,
            )
    primary = independent_screen(np.array([r["effects"]["128"]["finetune"]["per_draw_effects"] for r in rows]))
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collection = [a for a in attempts if a.get("arm") and not a.get("replay")]
    rerun = [a for a in attempts if a.get("replay")]
    assert len(collection) == len(rerun) == 17 and all(a["complete"] for a in collection + rerun)
    assert sum(a["ssl_started"] for a in collection) == sum(a["ssl_completed"] for a in collection) == 3072
    assert sum(a["ssl_started"] for a in rerun) == sum(a["ssl_completed"] for a in rerun) == 3072
    for kind in ("knn", "linear", "finetune"):
        assert sum(a[f"{kind}_completed"] for a in collection) == 3 * len(new) <= 192
    assert sum(a["finetune_completed"] for a in rerun) == 2
    assert counts["knn_completed"] == counts["linear_completed"] == 3 * len(new)
    curves = {
        k: [trunk["observations"][str(p)]["scores"][k] for p in lock["points"]] for k in ("knn", "linear", "finetune")
    }
    write(
        out / "summary.json",
        {
            "parents": rows,
            "screen": primary,
            "trunk": {"points": lock["points"], "curves": curves, "health": health_summary, "shadow": shadow},
            "source_blocks": 1,
            "ssl_seeds": 1,
            "dependent_parents": 8,
            "controller_qualified": False,
        },
    )
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "source_disjoint": True,
            "natural_order_and_frames_verified": True,
            "all_reservoir_pixels_and_rng_verified": True,
            "all_base_updates_equal_trunk": True,
            "common_continuation_and_rng_matched": True,
            "all_five_geometry_signals_crosschecked": True,
            "checkpoint_count": len(checks),
            "checkpoints": checks,
            "new_reader_states": len(new),
            "collection_work": collection,
            "rerun_work": rerun,
            "probe_verification_work": counts,
            "total_ssl_updates": 6144,
            "screen": primary,
        },
    )
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    print({"independent_verification": "passed", "opportunity_screen": primary["passed"]}, flush=True)
