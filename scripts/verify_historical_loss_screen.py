"""Audit causal new-source selection, all image diets, loss alarms and paired outcomes."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from cafl4ds.filters.health_replay_mix import VariableWindowReplay
from cafl4ds.health_calibration import embeddings, geometry
from cafl4ds.historical_loss_screen import bank_losses
from cafl4ds.measurements import cosine_drift, linear_cka
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.short_buffer import restore
from scripts.natural_health_view import SIGNALS, frozen_predictions, write
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read


def primitive_diet(trace, stream, parent=None, reset_point=None):
    """Rebuild every batch from causal pixels without replay-buffer implementation calls."""
    ids = deque()
    pixels = {}
    generator = torch.Generator().manual_seed(73919)
    if parent is not None:
        state = parent["memory"]
        ids.extend(state["ids"])
        generator.set_state(state["rng"])
        pixels = {i: image.numpy().copy() for i, image in zip(state["ids"], state["images"], strict=True)}
    for row in trace:
        if reset_point is not None and row["offset"] == 33:
            generator.manual_seed(89201 + reset_point)
        step, past = row["step"], row["past"]
        count = 32 - past
        incoming = stream.batch(step - 1)
        assert hashlib.sha256(incoming.tobytes()).hexdigest() == row["input_sha256"]
        older = torch.randperm(len(ids), generator=generator)[:past].tolist() if past else []
        selected = torch.randperm(32, generator=generator)[:count].sort().values.tolist() if past else list(range(32))
        old_ids = [ids[i] for i in older]
        new_ids = [(step - 1) * 32 + i for i in selected]
        assert old_ids == row["diet"]["replay_ids"] and new_ids == row["diet"]["current_ids"]
        assert row["diet"]["replay_arrival_ages"] == [step - 1 - i // 32 for i in old_ids]
        batch = np.concatenate([incoming[selected], np.stack([pixels[i] for i in old_ids])]) if past else incoming
        assert hashlib.sha256(batch.tobytes()).hexdigest() == row["train_sha256"]
        for i, image in enumerate(incoming):
            identity = (step - 1) * 32 + i
            ids.append(identity)
            pixels[identity] = image.copy()
            if len(ids) > 256:
                del pixels[ids.popleft()]
        assert row["diet"]["resident_frames"] == len(ids) and row["diet"]["admitted_frames_seen"] == step * 32
        assert row["diet"]["oldest_resident_id"] == ids[0]
    return {
        "ids": list(ids),
        "rng": generator.get_state(),
        "hashes": [hashlib.sha256(pixels[i].tobytes()).hexdigest() for i in ids],
    }


def shadow_audit(trunk, raw):
    """Derive shadow transitions and per-parent risk from raw seed/image losses."""
    reference = float(raw["132_monitor_losses"].mean(dtype=np.float64))
    high_count = low_count = 0
    latched = False
    events = []
    for step in range(140, 1025, 8):
        loss = float(raw[f"{step}_monitor_losses"].mean(dtype=np.float64))
        risk = loss / reference - 1
        high, low = risk >= 0.02 - 1e-12, risk <= 0.01 + 1e-12
        fired = rearmed = False
        if latched:
            high_count = 0
            low_count = low_count + 1 if low else 0
            if low_count == 2:
                latched = False
                low_count = 0
                rearmed = True
        else:
            low_count = 0
            high_count = high_count + 1 if high else 0
            if high_count == 2:
                latched = True
                high_count = 0
                fired = True
        expected = {
            "step": step,
            "loss": loss,
            "risk": risk,
            "high": high,
            "low": low,
            "latched": latched,
            "high_streak": high_count,
            "low_streak": low_count,
            "alarm": fired,
            "rearmed": rearmed,
        }
        assert trunk["events"][len(events)] == expected
        assert trunk["monitors"][str(step)]["shadow_alarm"] == fired
        events.append(expected)
    assert len(events) == len(trunk["events"])
    return {"reference": reference, "events": events}


def checkpoint_audit(path, expected, initial, method, banks, raw=None, prefix=None, shadow=False):
    """Deserialize actual pixels/optimizer state and reproduce final measurement arrays."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    memory = VariableWindowReplay()
    memory.load_state_dict(state["memory"])
    assert memory.marker()["ids"] == expected["ids"]
    assert memory.marker()["hashes"] == expected["hashes"]
    assert torch.equal(memory.marker()["rng"], expected["rng"])
    model, optimizer = restore(initial, method.optim, state["learner"])
    if raw is not None:
        for name, images in banks.items():
            np.testing.assert_array_equal(embeddings(model, images), raw[f"{prefix}_{name}_features"])
            np.testing.assert_array_equal(bank_losses(model, images), raw[f"{prefix}_{name}_losses"])
    result = fingerprint(
        {"learner": state["learner"], "memory": memory.marker(), "shadow": state["shadow"] if shadow else None}
    )
    del model, optimizer
    return result


def verify(config, out):
    """Recompute primary conservative action utility and screen conditions independently."""
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    assert read(out / "training-verification.json")["passed"]
    intake = read(ROOT / config.frames / "intake.json")
    selection = read(ROOT / config.intake / "selection.json")
    exclusions = read(ROOT / config.intake / "prior-exclusions.json")
    pinned = read(ROOT / config.archive_index)["entries"]
    excluded = set(exclusions["ids"])
    expected = []
    for pos, entry in enumerate(pinned):
        if pos <= exclusions["start_after_archive_position"] or "/train/" not in entry["path"]:
            continue
        clip = Path(entry["path"]).stem
        if clip in excluded:
            continue
        expected.append((pos, clip))
        if len(expected) == 96:
            break
    assert expected == [(c["archive_position"], c["clip_id"]) for c in selection["clips"]]
    actual_ids = [c["clip_id"] for c in intake["clips"]]
    assert actual_ids == [c["clip_id"] for c in selection["clips"][: len(actual_ids)]]
    assert not set(actual_ids).intersection(excluded)
    assert sum(c["shape"][0] for c in intake["clips"]) == 32768
    for clip in intake["clips"]:
        frames = np.load(clip["frames_path"], mmap_mode="r", allow_pickle=False)
        assert len(frames) == len(clip["frames"])
        times = [r["pts_seconds"] for r in clip["frames"]]
        assert all(a < b for a, b in zip(times, times[1:], strict=False))
        for image, entry in zip(frames, clip["frames"], strict=True):
            assert hashlib.sha256(image.tobytes()).hexdigest() == entry["rgb64_sha256"]
    cache = read(out / "readout_cache.json")
    with np.load(ROOT / config.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    counts = {f"{kind}_{state}": 0 for kind in ("knn", "linear") for state in ("started", "completed")}
    journal = out / "attempts/probe_verification.json"
    new_states = {k: r for k, r in cache.items() if not r.get("inherited", False)}
    for index, row in enumerate(new_states.values(), 1):
        assert digest(row["path"]) == row["sha256"]
        with np.load(row["path"]) as saved:
            repeated = frozen_predictions(saved["features"], arrays, config.draws, counts, journal)
            for kind in ("knn", "linear"):
                np.testing.assert_array_equal(repeated[kind], saved[kind])
            np.testing.assert_array_equal(saved["finetune"], saved["probabilities"][:, :3663].argmax(-1))
            q = arrays["query"]
            for kind in ("knn", "linear", "finetune"):
                scored = []
                for prediction in saved[kind]:
                    cells = [
                        100 * (prediction[(arrays["domains"][q] == d) & (arrays["labels"][q] == c)] == c).mean()
                        for d in (0, 1)
                        for c in (0, 1)
                    ]
                    scored.append([np.mean(cells), np.mean(cells[:2]), np.mean(cells[2:])])
                np.testing.assert_allclose(scored, row["scores"][kind], rtol=0, atol=1e-12)
        if index % 16 == 0:
            print({"new_reader_states_verified": index, "total": len(new_states)}, flush=True)
    trunk = read(out / "trunk.json")
    branches = read(out / "branches.json")
    stream = OrderedFrames(intake["clips"])
    trunk_diet = primitive_diet(trunk["trace"], stream)
    method = OmegaConf.create(lock["model_config"])
    initial = _method(method).cuda()
    banks = {"monitor": reference_images(lock["monitor_reference"]), "audit": reference_images(lock["audit_reference"])}
    rows = []
    health = {}
    checkpoints = []
    with np.load(out / "trunk-measurements.npz") as raw:
        shadow = shadow_audit(trunk, raw)
        points = sorted(map(int, trunk["monitors"]))
        for name in banks:
            reference = raw[f"132_{name}_features"]
            loss_ref = float(raw[f"132_{name}_losses"].mean(dtype=np.float64))
            values = {key: [] for key in SIGNALS}
            means = []
            seed_means = []
            for p in points:
                z = raw[f"{p}_{name}_features"]
                v = geometry(z)
                v.update(cka=max(0, 1 - linear_cka(reference, z)), cosine=cosine_drift(reference, z))
                for key in SIGNALS:
                    values[key].append(v[key] if p >= 132 or key not in ("cka", "cosine") else None)
                loss = raw[f"{p}_{name}_losses"]
                means.append(float(loss.mean(dtype=np.float64)))
                seed_means.append(loss.mean(1, dtype=np.float64))
                assert means[-1] == trunk["monitors"][str(p)]["means"][name]
            health[name] = {
                "points": points,
                "values": values,
                "loss": means,
                "seed_losses": seed_means,
                "reference_loss": loss_ref,
                "risk": [None if p < 132 else loss / loss_ref - 1 for p, loss in zip(points, means, strict=True)],
            }
        mark = checkpoint_audit(out / "trunk-final.pt", trunk_diet, initial, method, banks, raw, "1024", True)
        assert mark == trunk["final_fingerprint"]
        checkpoints.append({"arm": "trunk", "fingerprint": mark})
        for parent_point in config.landmarks:
            p = int(parent_point)
            parent_path = Path(trunk["parents"][str(p)]["path"])
            parent = torch.load(parent_path, weights_only=True, map_location="cpu")
            parent_diet = primitive_diet(trunk["trace"][:p], stream)
            mark = checkpoint_audit(parent_path, parent_diet, initial, method, banks, raw, str(p), True)
            assert mark == trunk["parents"][str(p)]["fingerprint"]
            loss = float(raw[f"{p}_monitor_losses"].mean(dtype=np.float64))
            risk = loss / shadow["reference"] - 1
            audit_loss = float(raw[f"{p}_audit_losses"].mean(dtype=np.float64))
            row = {
                "parent": p,
                "risk": risk,
                "high_risk": risk >= 0.02 - 1e-12,
                "loss": loss,
                "audit_loss": audit_loss,
                "shadow_latched": trunk["monitors"][str(p)]["shadow"]["latched"],
                "shadow_last_alarm": trunk["monitors"][str(p)]["shadow"]["last_alarm"],
                "shadow_last_check": trunk["monitors"][str(p)]["shadow"]["last_observation"],
                "effects": {},
                "ssl_effects": {},
            }
            pair = {arm: branches[f"p{p:04d}_{arm}"] for arm in ("current", "replay")}
            for arm, record in pair.items():
                assert record["parent_fingerprint"] == trunk["parents"][str(p)]["fingerprint"]
                assert len(record["trace"]) == 64
                for t in record["trace"]:
                    assert t["past"] == (16 if t["offset"] > 32 else (0 if arm == "current" else 24))
                diet = primitive_diet(record["trace"], stream, parent, p)
                with np.load(out / "branches" / f"p{p:04d}_{arm}" / "measurements.npz") as branch_raw:
                    mark = checkpoint_audit(record["state_path"], diet, initial, method, banks, branch_raw, "64")
                    assert mark == record["final_fingerprint"]
                    for h in ("32", "64"):
                        row["ssl_effects"].setdefault(h, {})[arm] = {
                            name: float(branch_raw[f"{h}_{name}_losses"].mean(dtype=np.float64)) for name in banks
                        }
                for observation in record["observations"].values():
                    assert observation["scores"] == cache[observation["encoder_fingerprint"]]["scores"]
                checkpoints.append({"arm": f"p{p:04d}_{arm}", "fingerprint": mark})
            for h in ("32", "64"):
                assert (
                    pair["current"]["observations"][h]["rng_fingerprint"]
                    == pair["replay"]["observations"][h]["rng_fingerprint"]
                )
                row["effects"][h] = {}
                for kind in ("knn", "linear", "finetune"):
                    a = np.array(pair["current"]["observations"][h]["scores"][kind])
                    b = np.array(pair["replay"]["observations"][h]["scores"][kind])
                    effects = b - a
                    row["effects"][h][kind] = {
                        "per_draw_effects": effects,
                        "pooled_effects": effects.mean(0),
                        "per_draw_utility": effects[:, 1:].min(1),
                        "pooled_utility": effects.mean(0)[1:].min(),
                    }
            for x, y in zip(pair["current"]["trace"][32:], pair["replay"]["trace"][32:], strict=True):
                assert x["diet"] == y["diet"] and x["train_sha256"] == y["train_sha256"]
            assert (
                pair["current"]["observations"]["32"]["learner_fingerprint"]
                == trunk["monitors"][str(p + 32)]["learner_fingerprint"]
            )
            rows.append(row)
            print(
                {
                    "paired_parent_verified": p,
                    "risk": risk,
                    "terminal_utility": row["effects"]["64"]["finetune"]["pooled_utility"],
                },
                flush=True,
            )
    risks = np.array([r["risk"] for r in rows])
    high = risks >= 0.02 - 1e-12
    utilities = np.array([r["effects"]["64"]["finetune"]["pooled_utility"] for r in rows])
    perdraw = np.array([r["effects"]["64"]["finetune"]["per_draw_utility"] for r in rows])
    difference = float(utilities[high].mean() - utilities[~high].mean()) if high.any() and (~high).any() else None
    draw_differences = (
        (perdraw[high].mean(0) - perdraw[~high].mean(0)).tolist() if high.any() and (~high).any() else None
    )
    gates = {
        "at_least_three_high_and_low": int(high.sum()) >= 3 and int((~high).sum()) >= 3,
        "helpful_and_harmful_actions_present": int((utilities >= 0.5).sum()) >= 2
        and int((utilities <= -0.5).sum()) >= 2,
        "high_risk_mean_advantage_at_least_half_point": difference is not None and difference >= 0.5,
        "positive_direction_all_draws": draw_differences is not None and all(d > 0 for d in draw_differences),
    }
    screen = {
        "gates": gates,
        "passed": all(gates.values()),
        "high_count": int(high.sum()),
        "low_count": int((~high).sum()),
        "helpful_count": int((utilities >= 0.5).sum()),
        "harmful_count": int((utilities <= -0.5).sum()),
        "high_minus_low_mean_utility": difference,
        "per_draw_high_minus_low": draw_differences,
        "spearman_risk_utility": float(spearmanr(risks, utilities).statistic)
        if np.ptp(risks) > 0 and np.ptp(utilities) > 0
        else None,
        "verdict": "promising_development_screen"
        if all(gates.values())
        else (
            "insufficient_cases"
            if not gates["at_least_three_high_and_low"] or not gates["helpful_and_harmful_actions_present"]
            else "signal_screen_failed"
        ),
        "controller_qualified": False,
    }
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collection = [a for a in attempts if a.get("arm") and not a.get("replay")]
    rerun = [a for a in attempts if a.get("replay")]
    assert len(collection) == len(rerun) == 25 and all(a["complete"] for a in collection + rerun)
    assert sum(a["ssl_completed"] for a in collection) == sum(a["ssl_completed"] for a in rerun) == 2560
    for kind in ("knn", "linear", "finetune"):
        assert sum(a[f"{kind}_completed"] for a in collection) == 3 * len(new_states) <= 231
    assert sum(a["finetune_completed"] for a in rerun) == 2
    assert counts["knn_completed"] == counts["linear_completed"] == 3 * len(new_states)
    curves = {
        k: [trunk["observations"][str(p)]["scores"][k] for p in lock["points"]] for k in ("knn", "linear", "finetune")
    }
    write(
        out / "summary.json",
        {
            "parents": rows,
            "screen": screen,
            "trunk": {"points": lock["points"], "curves": curves, "health": health, "shadow": shadow},
            "source_blocks": 1,
            "ssl_seeds": 1,
            "dependent_parents": 12,
            "controller_qualified": False,
        },
    )
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "source_disjoint": True,
            "natural_order_and_frame_hashes_verified": True,
            "all_diets_reconstructed": True,
            "all_continuation_batches_and_rng_matched": True,
            "shadow_decisions_rebuilt": True,
            "final_and_parent_checkpoints_verified": True,
            "checkpoint_count": len(checkpoints) + 12,
            "checkpoints": checkpoints,
            "new_reader_states": len(new_states),
            "collection_work": collection,
            "rerun_work": rerun,
            "probe_verification_work": counts,
            "total_ssl_updates": 5120,
            "screen": screen,
        },
    )
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
