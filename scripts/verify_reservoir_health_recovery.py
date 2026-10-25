"""Reconstruct causal pulse decisions, exact-dose controls, pixels and downstream margins."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.filters.reservoir_health_recovery import RecoveryReservoir, ReservoirHealthRecovery
from cafl4ds.health_calibration import embeddings
from cafl4ds.historical_loss_screen import bank_losses
from cafl4ds.natural_health_rate import OrderedFrames, path_metrics
from cafl4ds.short_buffer import restore
from scripts.natural_health_view import frozen_predictions, write
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read
from scripts.verify_episode_health_replay import independent_reading
from scripts.verify_memory_age_replay import health


def protection(candidate, baseline):
    """Apply the frozen peak-loss benefit and both endpoint budgets against constant replay."""
    a, b = np.asarray(candidate, float), np.asarray(baseline, float)
    gates = {
        "drawdown_reduction_at_least_one": b[-1] - a[-1] >= 1 - 1e-12,
        "day_endpoint_within_one": a[1] - b[1] >= -1 - 1e-12,
        "night_endpoint_within_one": a[2] - b[2] >= -1 - 1e-12,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "drawdown_reduction": float(b[-1] - a[-1]),
        "endpoint_effect": a[:3] - b[:3],
    }


def timing_screen(candidate, controls, primary_passed):
    """Require a directionally better peak loss and bounded endpoints versus every timing control."""
    a = np.asarray(candidate, float)
    rows = {}
    for arm, values in controls.items():
        b = np.asarray(values, float)
        rows[arm] = {
            "drawdown_reduction": float(b[-1] - a[-1]),
            "endpoint_effect": a[:3] - b[:3],
            "passed": bool(a[-1] < b[-1] and np.all(a[1:3] - b[1:3] >= -1 - 1e-12)),
        }
    return {
        "passed": bool(primary_passed and all(r["passed"] for r in rows.values())),
        "primary_screen_passed": bool(primary_passed),
        "controls": rows,
    }


def reconstruct(record, constant, raw, stream):
    """Use primitive slots and episode counters, without invoking either controller or sampler."""
    ids, pixels = [], []
    sample = torch.Generator().manual_seed(73919)
    admit = torch.Generator().manual_seed(90101)
    seen = 0
    latched = False
    high_count = low_count = 0
    alarm_at = None
    events = []
    ages = []
    current_exposures = replay_exposures = 0
    prefix = True
    prefix_updates = 0
    latched_normal = 0
    for row, base in zip(record["trace"], constant["trace"], strict=True):
        step = row["step"]
        assert step == seen // 32 + 1
        age = None if alarm_at is None else step - alarm_at
        current = 24 if age is None or age > 64 else (8 if age <= 32 else 16)
        if record["schedule"] is not None:
            current = record["schedule"][step - 1]
        if step == 1:
            current = 32
        past = 32 - current
        assert row["mix"] == [current, past]
        if latched and past == 8:
            latched_normal += 1
        batch = stream.batch(step - 1)
        assert hashlib.sha256(batch.tobytes()).hexdigest() == row["input_sha256"] == base["input_sha256"]
        old_slots = torch.randperm(len(ids), generator=sample)[:past].tolist() if past else []
        now_slots = torch.randperm(32, generator=sample)[:current].sort().values.tolist() if past else list(range(32))
        current_ids = [seen + i for i in now_slots]
        past_ids = [ids[i] for i in old_slots]
        assert current_ids == row["diet"]["current_ids"] and past_ids == row["diet"]["replay_ids"]
        assert all(i < seen for i in past_ids)
        if step > 1 and past != 8:
            assert set(current_ids) <= set(base["diet"]["current_ids"])
            assert past_ids[:8] == base["diet"]["replay_ids"]
            prefix = False
        else:
            assert current_ids == base["diet"]["current_ids"] and past_ids == base["diet"]["replay_ids"]
        mixed = np.concatenate((batch[now_slots], np.stack([pixels[i] for i in old_slots]))) if past else batch
        assert hashlib.sha256(mixed.tobytes()).hexdigest() == row["train_sha256"]
        if prefix:
            assert row["train_sha256"] == base["train_sha256"] and row["response"] == base["response"]
            prefix_updates += 1
        age = [step - 1 - i // 32 for i in past_ids]
        assert age == row["diet"]["replay_arrival_ages"]
        ages.extend(age)
        retained = 0
        for image in batch:
            if seen < 256:
                ids.append(seen)
                pixels.append(image.copy())
                retained += 1
            else:
                slot = int(torch.randint(0, seen + 1, (1,), generator=admit).item())
                if slot < 256:
                    ids[slot], pixels[slot] = seen, image.copy()
                    retained += 1
            seen += 1
        assert row["diet"]["admitted_frames_seen"] == seen
        assert row["diet"]["resident_frames"] == len(ids) == min(256, seen)
        assert row["diet"]["oldest_resident_id"] == min(ids)
        assert row["diet"]["retained_arrivals"] == retained
        current_exposures += current
        replay_exposures += past
        if step > 132 and (step - 132) % 8 == 0:
            values = independent_reading(raw["132_monitor_features"], raw[f"{step}_monitor_features"])
            score = float(max(values))
            high = score >= 0.1 - 1e-12
            low = score <= 0.05 + 1e-12
            fired = rearmed = False
            if latched:
                high_count = 0
                low_count = low_count + 1 if low else 0
                if low_count >= 2:
                    latched = False
                    low_count = 0
                    rearmed = True
            else:
                low_count = 0
                high_count = high_count + 1 if high else 0
                if high_count >= 2:
                    latched = True
                    high_count = 0
                    fired = True
                    alarm_at = step
            actual = record["events"][len(events)]
            np.testing.assert_allclose(actual["values"], values, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(actual["score"], score, rtol=1e-10, atol=1e-12)
            expected = {
                "step": step,
                "unhealthy": high,
                "low": low,
                "streak": high_count,
                "low_streak": low_count,
                "latched": latched,
                "rearmed": rearmed,
                "alarm": fired,
            }
            assert all(actual[k] == v for k, v in expected.items())
            assert record["monitors"][str(step)]["alarm"] == fired
            events.append(actual)
    assert seen == 131072 and len(events) == len(record["events"]) == 495
    return {
        "ids": ids,
        "images": np.stack(pixels),
        "sample_rng": sample.get_state(),
        "admission_rng": admit.get_state(),
        "hashes": [hashlib.sha256(x.tobytes()).hexdigest() for x in pixels],
        "seen": seen,
        "latched": latched,
        "streak": high_count,
        "low_streak": low_count,
        "last_alarm": alarm_at,
        "summary": {
            "current_exposures": current_exposures,
            "replay_exposures": replay_exposures,
            "middle_arrivals": sum(r["mix"] == [16, 16] for r in record["trace"]),
            "base_arrivals": sum(r["mix"] == [24, 8] for r in record["trace"]),
            "heavy_arrivals": sum(r["mix"][1] == 24 for r in record["trace"]),
            "mean_replay_age": float(np.mean(ages)),
            "max_replay_age": max(ages),
            "prefix_updates_equal_constant": prefix_updates,
            "latched_normal_arrivals": latched_normal,
        },
    }


def verify(config, out):
    """Audit five new physical trajectories, exact-dose contrasts and preserved baseline lineage."""
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    check(read(out / "schedule-lock.json")["hashes"])
    assert read(out / "training-verification.json")["passed"]
    records = read(out / "collection.json")["records"]
    constant = read(out / "constant-record.json")
    base_alloc = np.array([r["mix"][0] for r in records["health"]["trace"]])
    schedules = read(out / "schedules.json")
    for offset in (256, 512, 1024):
        expected = np.r_[base_alloc[:132], np.roll(base_alloc[132:], offset)]
        np.testing.assert_array_equal(expected, schedules[f"shift_{offset}"])
    assert read(out / "schedule-lock.json")["frozen_unix"] <= min(
        read(p)["started_unix"] for p in (out / "attempts").glob("collect_shift_*.json")
    )
    cache = read(out / "readout_cache.json")
    with np.load(ROOT / config.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    counts = {f"{k}_{s}": 0 for k in ("knn", "linear") for s in ("started", "completed")}
    journal = out / "attempts/probe_verification.json"
    new = {k: r for k, r in cache.items() if not r.get("inherited", False)}
    for index, row in enumerate(new.values(), 1):
        assert digest(row["path"]) == row["sha256"]
        with np.load(row["path"]) as saved:
            predictions = frozen_predictions(saved["features"], arrays, config.draws, counts, journal)
            for kind in ("knn", "linear"):
                np.testing.assert_array_equal(predictions[kind], saved[kind])
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
            print({"new_reader_states_verified": index, "total": len(new)}, flush=True)
    stream = OrderedFrames(read(ROOT / config.frames / "intake.json")["clips"])
    prior_final = torch.load(ROOT / config.prior / "final_state.pt", weights_only=True, map_location="cpu")
    method = OmegaConf.create(lock["model_config"])
    initial = _method(method).cuda()
    banks = {name: reference_images(lock[f"{name}_reference"]) for name in ("monitor", "audit")}
    summary = {}
    checkpoints = []
    for arm in config.arms:
        record = records[arm]
        directory = out / arm
        assert record["alarms_drive_actions"] == (arm == "health")
        assert record["schedule"] == (
            None if arm == "health" else ([32] + [24] * 4095 if arm == "constant" else schedules[arm])
        )
        assert record["points"] == lock["points"]
        allocations = [r["mix"][0] for r in record["trace"]]
        if arm != "constant":
            assert sorted(allocations) == sorted(base_alloc.tolist())
        else:
            assert allocations == [32] + [24] * 4095
        state = torch.load(directory / "final_state.pt", weights_only=True, map_location="cpu")
        with np.load(directory / "measurements.npz") as raw:
            diet = reconstruct(record, constant, raw, stream)
            for key in ("ids", "seen", "hashes"):
                assert state["memory"][key] == diet[key] == prior_final["memory"][key]
            for key in ("sample_rng", "admission_rng"):
                assert torch.equal(state["memory"][key], diet[key])
                assert torch.equal(state["memory"][key], prior_final["memory"][key])
            np.testing.assert_array_equal(state["memory"]["images"].numpy(), diet["images"])
            np.testing.assert_array_equal(state["memory"]["images"], prior_final["memory"]["images"])
            assert torch.equal(state["learner"]["cpu_rng"], prior_final["learner"]["cpu_rng"])
            for a, b in zip(state["learner"]["cuda_rng"], prior_final["learner"]["cuda_rng"], strict=True):
                assert torch.equal(a, b)
            memory = RecoveryReservoir(config.capacity, config.sample_seed, config.admission_seed)
            memory.load_state_dict(state["memory"])
            control = ReservoirHealthRecovery()
            control.load_state_dict(state["controller"])
            for key in ("latched", "streak", "low_streak", "last_alarm"):
                assert getattr(control, key) == diet[key]
            assert control.last_observation == 4092 and control.credit == 0 and control.events == record["events"]
            np.testing.assert_array_equal(control.reference, raw["132_monitor_features"])
            mark = fingerprint(
                {"learner": state["learner"], "memory": memory.marker(), "controller": control.state_dict()}
            )
            assert mark == record["final_fingerprint"]
            model, _ = restore(initial, method.optim, state["learner"])
            for name, bank in banks.items():
                np.testing.assert_array_equal(embeddings(model, bank), raw[f"4096_{name}_features"])
                np.testing.assert_array_equal(bank_losses(model, bank, config.loss_seeds), raw[f"4096_{name}_losses"])
            bank_health = {name: health(raw, name) for name in banks}
            losses = {}
            for name in banks:
                means = [float(raw[f"{p}_{name}_losses"].mean(dtype=np.float64)) for p in lock["points"]]
                ref = means[lock["points"].index(132)]
                losses[name] = {
                    "points": lock["points"],
                    "loss": means,
                    "reference": ref,
                    "risk": [m / ref - 1 if p >= 132 else None for p, m in zip(lock["points"], means, strict=True)],
                }
        curves = {
            k: np.array([record["observations"][str(p)]["scores"][k] for p in lock["points"]])
            for k in ("knn", "linear", "finetune")
        }
        for obs in record["observations"].values():
            assert obs["scores"] == cache[obs["encoder_fingerprint"]]["scores"]
        summary[arm] = {
            "points": lock["points"],
            "reader_curves": curves,
            "mean_curves": {k: v.mean(1) for k, v in curves.items()},
            "metrics": {k: path_metrics(v.mean(1), lock["points"]) for k, v in curves.items()},
            "per_draw_metrics": {
                k: [path_metrics(v[:, d], lock["points"]) for d in range(3)] for k, v in curves.items()
            },
            "warm_floor_loss": {k: np.maximum(0, v.mean(1)[0] - v.mean(1).min(0)) for k, v in curves.items()},
            "health": bank_health["monitor"],
            "audit_health": bank_health["audit"],
            "losses": losses,
            "diet": diet["summary"],
            "alarm_count": sum(e["alarm"] for e in record["events"]),
            "alarm_arrivals": [e["step"] for e in record["events"] if e["alarm"]],
            "updates": 4096,
        }
        checkpoints.append({"arm": arm, "fingerprint": mark, "sha256": digest(directory / "final_state.pt")})
        print({"independent_arm_verified": arm, "replay_exposures": diet["summary"]["replay_exposures"]}, flush=True)
    primary = protection(summary["health"]["metrics"]["finetune"], summary["constant"]["metrics"]["finetune"])
    timing = timing_screen(
        summary["health"]["metrics"]["finetune"],
        {a: summary[a]["metrics"]["finetune"] for a in config.arms if a.startswith("shift_")},
        primary["passed"],
    )
    per_draw = []
    for d in range(3):
        p = protection(
            summary["health"]["per_draw_metrics"]["finetune"][d], summary["constant"]["per_draw_metrics"]["finetune"][d]
        )
        t = timing_screen(
            summary["health"]["per_draw_metrics"]["finetune"][d],
            {a: summary[a]["per_draw_metrics"]["finetune"][d] for a in config.arms if a.startswith("shift_")},
            p["passed"],
        )
        per_draw.append({"protection": p, "timing": t})
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collection = [a for a in attempts if a.get("arm") and not a.get("replay")]
    rerun = [a for a in attempts if a.get("replay")]
    assert len(collection) == 5 and len(rerun) == 1 and all(a["complete"] for a in collection + rerun)
    assert {a["arm"] for a in collection} == set(config.arms) and rerun[0]["arm"] == "health"
    assert all(a["ssl_started"] == a["ssl_completed"] == 4096 for a in collection + rerun)
    assert sum(a["ssl_completed"] for a in collection + rerun) == config.max_new_ssl_updates
    for kind in ("knn", "linear", "finetune"):
        assert sum(a[f"{kind}_completed"] for a in collection) == 3 * len(new) <= config.max_collection_fits_per_kind
    assert counts["knn_completed"] == counts["linear_completed"] == 3 * len(new)
    assert rerun[0]["finetune_completed"] == 2
    write(
        out / "summary.json",
        {
            "models": summary,
            "protection_screen": primary,
            "timing_screen": timing,
            "per_draw_screens": per_draw,
            "controller_qualified": False,
            "source_streams": 1,
            "ssl_seeds": 1,
        },
    )
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "actual_new_ssl_updates": 24576,
            "inherited_ssl_updates": 0,
            "new_readout_states": len(new),
            "collection_work": collection,
            "rerun_work": rerun,
            "probe_verification_work": counts,
            "checkpoints": checkpoints,
            "all_pulse_decisions_rebuilt": True,
            "all_mixed_pixels_reconstructed": True,
            "equal_doses_verified": True,
            "common_reservoir_pixels_and_rng_verified": True,
            "prefixes_match_constant": True,
            "all_five_health_signals_crosschecked": True,
            "final_embeddings_and_losses_exact": True,
            "protection_screen": primary,
            "timing_screen": timing,
            "per_draw_screens": per_draw,
        },
    )
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    print(
        {"independent_verification": "passed", "protection": primary["passed"], "timing": timing["passed"]}, flush=True
    )
