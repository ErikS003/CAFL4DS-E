"""Independently audit T1a decisions, pixels, readers, timing controls and saved state."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.filters.episode_health_replay import EpisodeHealthReplay
from cafl4ds.filters.health_replay_mix import VariableWindowReplay
from cafl4ds.health_calibration import embeddings, geometry
from cafl4ds.measurements import cosine_drift, linear_cka
from cafl4ds.natural_health_rate import OrderedFrames, path_metrics
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import restore
from scripts.natural_health_view import SIGNALS, frozen_predictions, write
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read


def independent_reading(reference, features):
    """Derive the three trigger inputs from raw banks without the controller reader."""
    a, b = np.asarray(reference, float), np.asarray(features, float)

    def statistics(values):
        singular = np.linalg.svd(values, compute_uv=False)
        proportions = singular / max(singular.sum(), 1e-12) + 1e-7
        rank = np.exp(-np.sum(proportions * np.log(proportions)))
        unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return rank, unit.var(0, ddof=1).mean()

    ar, av = statistics(a)
    br, bv = statistics(b)
    x, y = a - a.mean(0), b - b.mean(0)
    cka = np.square(x.T @ y).sum() / max(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y), 1e-30)
    return np.array([1 - br / ar, 1 - bv / av, max(0, 1 - cka)])


def reconstruct(record, banks, stream):
    """Rebuild each mix, episode transition and FIFO population using primitive state."""
    ids, pixels = deque(), {}
    generator = torch.Generator().manual_seed(73919)
    latched, high_count, low_count, alarm_at = False, 0, 0, None
    events, ages, mixes, episodes = [], [], [], []
    unhealthy_all_current = 0
    for row in record["trace"]:
        step = row["step"]
        stage = 4 if alarm_at is None else min(4, 1 + (step - alarm_at - 1) // 32)
        current = 8 * stage if record["schedule"] is None else record["schedule"][step - 1]
        past = 32 - current
        assert row["mix"] == [current, past] and row["rate"] == current / 32 and row["accepted"]
        batch = stream.batch(step - 1)
        assert hashlib.sha256(batch.tobytes()).hexdigest() == row["input_sha256"]
        old_order = torch.randperm(len(ids), generator=generator)[:past].tolist() if past else []
        now_order = (
            torch.randperm(32, generator=generator)[:current].sort().values.tolist() if past else list(range(32))
        )
        arriving = [(step - 1) * 32 + i for i in now_order]
        historical = [ids[i] for i in old_order]
        assert arriving == row["diet"]["current_ids"] and historical == row["diet"]["replay_ids"]
        mixed = np.concatenate([batch[now_order], np.stack([pixels[i] for i in historical])]) if past else batch
        assert hashlib.sha256(mixed.tobytes()).hexdigest() == row["train_sha256"]
        expected_ages = [step - 1 - i // 32 for i in historical]
        assert expected_ages == row["diet"]["replay_arrival_ages"] and all(1 <= a <= 8 for a in expected_ages)
        ages.extend(expected_ages)
        for i, image in enumerate(batch):
            identity = (step - 1) * 32 + i
            ids.append(identity)
            pixels[identity] = image.copy()
            if len(ids) > 256:
                del pixels[ids.popleft()]
        assert row["diet"]["resident_frames"] == len(ids)
        assert row["diet"]["admitted_frames_seen"] == step * 32
        assert row["diet"]["oldest_resident_id"] == ids[0]
        mixes.append(current)
        if latched and current == 32:
            unhealthy_all_current += 1
        if step > 132 and (step - 132) % 8 == 0:
            value = independent_reading(banks["132"], banks[str(step)])
            score = float(max(value))
            high, low = score >= 0.1 - 1e-12, score <= 0.05 + 1e-12
            fired = rearmed = False
            if latched:
                high_count = 0
                low_count = low_count + 1 if low else 0
                if low_count == 2:
                    latched, low_count, rearmed = False, 0, True
                    episodes[-1]["rearmed_at"] = step
            else:
                low_count = 0
                high_count = high_count + 1 if high else 0
                if high_count == 2:
                    latched, high_count, fired, alarm_at = True, 0, True, step
                    episodes.append({"alarm_at": step, "rearmed_at": None})
            actual = record["events"][len(events)]
            np.testing.assert_allclose(actual["values"], value, rtol=1e-10, atol=1e-12)
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
            assert all(actual[k] == v for k, v in expected.items()), (step, actual, expected)
            assert record["monitors"][str(step)]["alarm"] == fired
            events.append(actual)
    assert len(events) == len(record["events"])
    assert len(mixes) == record["updates"] == 4096
    assert record["first_alarm"] == next((e["step"] for e in events if e["alarm"]), None)
    return {
        "resident_ids": list(ids),
        "rng": generator.get_state(),
        "seen": 131072,
        "current_exposures": sum(mixes),
        "replay_exposures": 131072 - sum(mixes),
        "mix_arrivals": {f"{n}/{32 - n}": mixes.count(n) for n in (8, 16, 24, 32)},
        "replay_age_mean_arrivals": float(np.mean(ages)) if ages else None,
        "replay_age_max_arrivals": max(ages) if ages else None,
        "latched_all_current_arrivals": unhealthy_all_current,
        "episodes": episodes,
        "final_latched": latched,
        "final_high_count": high_count,
        "final_low_count": low_count,
    }


def health_summary(banks):
    """Cross-check all five instruments and expose pre-reference movement as unavailable."""
    result = {"points": sorted(map(int, banks.files)), "values": {s: [] for s in SIGNALS}}
    reference = banks["132"]
    with threadpool_limits(limits=4):
        for step in result["points"]:
            z = banks[str(step)]
            values, other = health_values(z, reference), geometry(z)
            other.update(cka=max(0.0, 1 - linear_cka(reference, z)), cosine=cosine_drift(reference, z))
            for name in SIGNALS:
                np.testing.assert_allclose(values[name], other[name], rtol=2e-5, atol=2e-6)
                result["values"][name].append(values[name] if step >= 132 or name not in ("cka", "cosine") else None)
    return result


def verify(config, out):
    """Verify new science without modifying collection artifacts or prior baselines."""
    check(read(out / "input_lock.json")["hashes"])
    check(read(out / "complete.json")["hashes"])
    assert read(out / "training-verification.json")["passed"]
    records = read(out / "collection.json")["records"]
    cache = read(out / "readout_cache.json")
    with np.load(ROOT / config.evaluation / "data.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    counts = {f"{k}_{state}": 0 for k in ("knn", "linear") for state in ("started", "completed")}
    journal = out / "attempts/probe_verification.json"
    new_states = {k: row for k, row in cache.items() if not row.get("inherited", False)}
    for index, row in enumerate(new_states.values(), 1):
        assert digest(row["path"]) == row["sha256"]
        with np.load(row["path"]) as saved:
            repeated = frozen_predictions(saved["features"], arrays, config.draws, counts, journal)
            for kind in ("knn", "linear"):
                np.testing.assert_array_equal(repeated[kind], saved[kind])
            np.testing.assert_array_equal(saved["finetune"], saved["probabilities"][:, :3663].argmax(-1))
            query = arrays["query"]
            for kind in ("knn", "linear", "finetune"):
                scored = []
                for prediction in saved[kind]:
                    cells = [
                        100 * (prediction[(arrays["domains"][query] == d) & (arrays["labels"][query] == c)] == c).mean()
                        for d in (0, 1)
                        for c in (0, 1)
                    ]
                    scored.append([np.mean(cells), np.mean(cells[:2]), np.mean(cells[2:])])
                np.testing.assert_allclose(scored, row["scores"][kind], rtol=0, atol=1e-12)
        if index % 16 == 0:
            print({"new_probe_states_verified": index, "total": len(new_states)}, flush=True)
    lock = read(out / "input_lock.json")
    audit = read(out / "audit_manifest.json")
    excluded = {Path(r["path"]).stem for r in lock["monitor_reference"]["images"]}
    selection = read(ROOT / config.selection)
    excluded.update(selection["evaluation_clip_ids"])
    excluded.update(c["clip_id"] for c in selection["clips"])
    assert not excluded.intersection(r["clip_id"] for r in audit["images"])
    expected = sorted(
        set(selection["warm_clip_ids"]) - excluded, key=lambda c: hashlib.sha256(f"88001:{c}".encode()).digest()
    )[:256]
    assert [r["clip_id"] for r in audit["images"]] == expected
    for row in audit["images"]:
        assert digest(row["path"]) == row["sha256"]
    base = np.array([r["mix"][0] for r in records["episode"]["trace"]])
    schedules = read(out / "schedules.json")
    for seed in (87001, 87007, 87013):
        gen = torch.Generator().manual_seed(seed)
        order = torch.randperm(495, generator=gen).numpy()
        expected_schedule = np.concatenate((base[:132], base[132:4092].reshape(495, 8)[order].ravel(), base[4092:]))
        np.testing.assert_array_equal(expected_schedule, schedules[f"perm_{seed}"])
    np.testing.assert_array_equal(np.r_[base[:132], np.roll(base[132:], 256)], schedules["shift_256"])
    stream = OrderedFrames(read(ROOT / config.frames / "intake.json")["clips"])
    prior_summary = read(ROOT / config.prior / "summary.json")["models"]
    summary = copy_baselines(prior_summary)
    checkpoints = []
    method = OmegaConf.create(lock["model_config"])
    initial = _method(method).cuda()
    bank_images, audit_images = reference_images(lock["monitor_reference"]), reference_images(audit)
    for arm in config.arms:
        record = records[arm]
        assert record["points"] == lock["points"]
        if arm != "episode":
            assert record["schedule"] == schedules[arm]
        mixes = [r["mix"][0] for r in record["trace"]]
        assert sorted(mixes) == sorted(base.tolist())
        with np.load(record["embedding_path"]) as banks, np.load(record["audit_embedding_path"]) as audit_banks:
            diet = reconstruct(record, banks, stream)
            health, audit_health = health_summary(banks), health_summary(audit_banks)
            state = torch.load(record["state_path"], weights_only=True, map_location="cpu")
            memory = VariableWindowReplay()
            memory.load_state_dict(state["memory"])
            control = EpisodeHealthReplay()
            control.load_state_dict(state["controller"])
            assert state["memory"]["ids"] == diet["resident_ids"]
            assert torch.equal(state["memory"]["rng"], diet.pop("rng"))
            assert state["memory"]["seen"] == diet["seen"]
            assert control.events == record["events"] and control.latched == diet["final_latched"]
            actual_marker = fingerprint(
                {"learner": state["learner"], "memory": memory.marker(), "controller": control.state_dict()}
            )
            assert actual_marker == record["final_fingerprint"]
            model, optimizer = restore(initial, method.optim, state["learner"])
            np.testing.assert_array_equal(embeddings(model, bank_images), banks["4096"])
            np.testing.assert_array_equal(embeddings(model, audit_images), audit_banks["4096"])
            del model, optimizer
        points = lock["points"]
        curves = {
            k: np.array([record["observations"][str(p)]["scores"][k] for p in points])
            for k in ("knn", "linear", "finetune")
        }
        for obs in record["observations"].values():
            assert obs["scores"] == cache[obs["encoder_fingerprint"]]["scores"]
        summary[arm] = {
            "points": points,
            "reader_curves": curves,
            "mean_curves": {k: v.mean(1) for k, v in curves.items()},
            "metrics": {k: path_metrics(v.mean(1), points) for k, v in curves.items()},
            "warm_floor_loss": {k: np.maximum(0, v.mean(1)[0] - v.mean(1).min(0)) for k, v in curves.items()},
            "per_draw_metrics": {k: [path_metrics(v[:, d], points) for d in range(3)] for k, v in curves.items()},
            "updates": 4096,
            "first_alarm": record["first_alarm"],
            "alarm_count": sum(e["alarm"] for e in record["events"]),
            "diet": diet,
            "health": health,
            "audit_health": audit_health,
        }
        checkpoints.append({"arm": arm, "sha256": digest(record["state_path"]), "fingerprint": actual_marker})
        print({"independent_arm_verified": arm, "replay_exposures": diet["replay_exposures"]}, flush=True)
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collected = [a for a in attempts if a.get("arm") and not a.get("replay")]
    rerun = [a for a in attempts if a.get("replay")]
    assert len(collected) == 5 and len(rerun) == 1 and all(a["complete"] for a in collected + rerun)
    assert sum(a["ssl_completed"] for a in collected + rerun) == int(config.max_new_ssl_updates)
    for kind in ("knn", "linear", "finetune"):
        total = sum(a[f"{kind}_completed"] for a in collected)
        assert total == 3 * len(new_states) <= int(config.max_new_readout_fits_per_kind)
    assert counts["knn_completed"] == counts["linear_completed"] == 3 * len(new_states)
    assert sum(a["finetune_completed"] for a in rerun) == 2
    write(
        out / "summary.json",
        {
            "models": summary,
            "stream_count": 1,
            "ssl_seed_count": 1,
            "equal_online_updates": True,
            "retrospective_timing_controls": True,
            "controller_qualified": False,
        },
    )
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "models": 8,
            "new_trajectories": 5,
            "inherited_verified_trajectories": 3,
            "new_ssl_updates": int(config.max_new_ssl_updates),
            "new_readout_states": len(new_states),
            "collection_work": collected,
            "rerun_work": rerun,
            "probe_verification_work": counts,
            "checkpoints": checkpoints,
            "equal_replay_dose_verified": True,
            "independent_episode_decisions": True,
            "mixed_pixels_reconstructed": True,
            "audit_bank_disjoint": True,
            "final_checkpoint_reembedding_exact": True,
            "all_five_health_signals_crosschecked": True,
        },
    )
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])


def copy_baselines(prior):
    """Retain independently verified baselines and explicitly mark their unavailable audit bank."""
    import copy  # noqa: PLC0415

    result = copy.deepcopy(prior)
    for item in result.values():
        item["audit_health"] = None
        curves = {k: np.array(v) for k, v in item["reader_curves"].items()}
        item["per_draw_metrics"] = {
            k: [path_metrics(v[:, d], item["points"]) for d in range(3)] for k, v in curves.items()
        }
    return result
