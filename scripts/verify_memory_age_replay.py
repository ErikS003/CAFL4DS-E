"""Audit uniform retention, exact image diets, readouts and the fixed protection screen."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.filters.historical_reservoir import HistoricalReservoir
from cafl4ds.health_calibration import embeddings, geometry
from cafl4ds.historical_loss_screen import bank_losses
from cafl4ds.measurements import cosine_drift, linear_cka
from cafl4ds.natural_health_rate import OrderedFrames, path_metrics
from cafl4ds.prediction_benchmark import health_values
from cafl4ds.short_buffer import restore
from scripts.natural_health_view import SIGNALS, frozen_predictions, write
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read


def reconstruct(record, fixed, stream, config):
    """Rebuild Algorithm R from primitive slots and verify every consumed image byte."""
    ids, pixels = [], []
    sample = torch.Generator().manual_seed(int(config.sample_seed))
    admit = torch.Generator().manual_seed(int(config.admission_seed))
    ages, retained, seen = [], [], 0
    rolling = []
    for row, fifo in zip(record["trace"], fixed["trace"], strict=True):
        step = row["step"]
        assert step == seen // 32 + 1
        batch = stream.batch(step - 1)
        assert hashlib.sha256(batch.tobytes()).hexdigest() == row["input_sha256"] == fifo["input_sha256"]
        count = 16 if ids else 0
        old_slots = torch.randperm(len(ids), generator=sample)[:count].tolist() if count else []
        new_slots = (
            torch.randperm(32, generator=sample)[: 32 - count].sort().values.tolist() if count else list(range(32))
        )
        current = [seen + i for i in new_slots]
        history = [ids[i] for i in old_slots]
        assert current == row["diet"]["current_ids"] == fifo["diet"]["current_ids"]
        assert history == row["diet"]["replay_ids"] and all(i < seen for i in history)
        mixed = np.concatenate((batch[new_slots], np.stack([pixels[i] for i in old_slots]))) if count else batch
        assert hashlib.sha256(mixed.tobytes()).hexdigest() == row["train_sha256"]
        if step <= 9:
            assert row["train_sha256"] == fifo["train_sha256"] and row["response"] == fifo["response"]
        age = [step - 1 - i // 32 for i in history]
        assert age == row["diet"]["replay_arrival_ages"]
        ages.extend(age)
        replacements = 0
        for image in batch:
            if seen < 256:
                ids.append(seen)
                pixels.append(image.copy())
                replacements += 1
            else:
                slot = torch.randint(0, seen + 1, (1,), generator=admit).item()
                if slot < 256:
                    ids[slot], pixels[slot] = seen, image.copy()
                    replacements += 1
            seen += 1
        assert row["diet"]["admitted_frames_seen"] == seen
        assert row["diet"]["resident_frames"] == len(ids) == min(256, seen)
        assert row["diet"]["oldest_resident_id"] == min(ids)
        assert row["diet"]["retained_arrivals"] == replacements
        retained.append(replacements)
        if step % 128 == 0:
            recent = ages[-128 * 16 :]
            rolling.append(
                {
                    "arrival": step,
                    "mean": float(np.mean(recent)),
                    "q10": float(np.quantile(recent, 0.1)),
                    "q90": float(np.quantile(recent, 0.9)),
                    "max": max(recent),
                    "oldest_resident_age": step - 1 - min(ids) // 32,
                }
            )
    assert seen == 131072
    return {
        "ids": ids,
        "images": np.stack(pixels),
        "sample_rng": sample.get_state(),
        "admission_rng": admit.get_state(),
        "hashes": [hashlib.sha256(x.tobytes()).hexdigest() for x in pixels],
        "seen": seen,
        "summary": {
            "current_exposures": 32 + 4095 * 16,
            "replay_exposures": 4095 * 16,
            "mean_replay_age": float(np.mean(ages)),
            "median_replay_age": float(np.median(ages)),
            "max_replay_age": max(ages),
            "replay_older_than_eight_fraction": float(np.mean(np.array(ages) > 8)),
            "retained_arrivals": retained,
            "rolling": rolling,
            "final_ids": ids,
        },
    }


def health(raw, name):
    """Cross-check all five geometry instruments on the saved original or audit bank."""
    points = sorted(int(k.split("_")[0]) for k in raw.files if k.endswith(f"_{name}_features"))
    ref = raw[f"132_{name}_features"]
    result = {"points": points, "values": {k: [] for k in SIGNALS}}
    with threadpool_limits(limits=4):
        for step in points:
            z = raw[f"{step}_{name}_features"]
            a = health_values(z, ref)
            b = geometry(z)
            b.update(cka=max(0, 1 - linear_cka(ref, z)), cosine=cosine_drift(ref, z))
            for key in SIGNALS:
                np.testing.assert_allclose(a[key], b[key], rtol=2e-5, atol=2e-6)
                result["values"][key].append(a[key] if step >= 132 or key not in ("cka", "cosine") else None)
    return result


def screen(reservoir, fifo, current):
    """Apply the declared drawdown benefit and two explicit endpoint-loss budgets."""
    a, b, c = (np.asarray(x, float) for x in (reservoir, fifo, current))
    reduction = float(b[-1] - a[-1])
    effects = a[:3] - c[:3]
    gates = {
        "drawdown_reduction_vs_fifo_at_least_one": reduction >= 1 - 1e-12,
        "day_endpoint_within_one_of_current": effects[1] >= -1 - 1e-12,
        "night_endpoint_within_one_of_current": effects[2] >= -1 - 1e-12,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "drawdown_reduction_vs_fifo": reduction,
        "endpoint_effect_vs_current": effects,
        "endpoint_effect_vs_fifo": a[:3] - b[:3],
        "drawdown_reduction_vs_current": float(c[-1] - a[-1]),
    }


def verify(config, out):
    """Independently verify new physical training while preserving imported baseline evidence."""
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    assert read(out / "training-verification.json")["passed"]
    cache = read(out / "readout_cache.json")
    record, baselines = read(out / "record.json"), read(out / "baselines.json")
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
            query = arrays["query"]
            for kind in ("knn", "linear", "finetune"):
                scores = []
                for pred in saved[kind]:
                    cells = [
                        100 * (pred[(arrays["domains"][query] == d) & (arrays["labels"][query] == c)] == c).mean()
                        for d in (0, 1)
                        for c in (0, 1)
                    ]
                    scores.append([np.mean(cells), np.mean(cells[:2]), np.mean(cells[2:])])
                np.testing.assert_allclose(scores, row["scores"][kind], atol=1e-12, rtol=0)
        if index % 8 == 0:
            print({"new_reader_states_verified": index, "total": len(new)}, flush=True)
    intake = read(ROOT / config.frames / "intake.json")
    assert len(intake["clips"]) == 211
    stream = OrderedFrames(intake["clips"])
    diet = reconstruct(record, baselines["fixed"], stream, config)
    state = torch.load(out / "final_state.pt", weights_only=True, map_location="cpu")
    for key in ("ids", "seen", "hashes"):
        assert state["memory"][key] == diet[key]
    for key in ("sample_rng", "admission_rng"):
        assert torch.equal(state["memory"][key], diet[key])
    np.testing.assert_array_equal(state["memory"]["images"].numpy(), diet["images"])
    memory = HistoricalReservoir(config.capacity, config.sample_seed, config.admission_seed)
    memory.load_state_dict(state["memory"])
    final_marker = fingerprint({"learner": state["learner"], "memory": memory.marker()})
    assert final_marker == record["final_fingerprint"]
    method = OmegaConf.create(lock["model_config"])
    initial = _method(method).cuda()
    model, _ = restore(initial, method.optim, state["learner"])
    points = lock["points"]
    summary = {k: copy.deepcopy(read(ROOT / config.prior / "summary.json")["models"][k]) for k in ("current", "fixed")}
    curves = {
        k: np.array([record["observations"][str(p)]["scores"][k] for p in points])
        for k in ("knn", "linear", "finetune")
    }
    with np.load(out / "measurements.npz") as raw:
        observer_health = {name: health(raw, name) for name in ("monitor", "audit")}
        losses = {}
        for name in ("monitor", "audit"):
            bank = reference_images(lock[f"{name}_reference"])
            np.testing.assert_array_equal(embeddings(model, bank), raw[f"4096_{name}_features"])
            np.testing.assert_array_equal(bank_losses(model, bank, config.loss_seeds), raw[f"4096_{name}_losses"])
            means = [float(raw[f"{p}_{name}_losses"].mean(dtype=np.float64)) for p in points]
            ref = means[points.index(132)]
            losses[name] = {
                "points": points,
                "loss": means,
                "reference": ref,
                "risk": [m / ref - 1 if p >= 132 else None for p, m in zip(points, means, strict=True)],
                "seed_means": [raw[f"{p}_{name}_losses"].mean(1).tolist() for p in points],
            }
    summary["reservoir"] = {
        "points": points,
        "reader_curves": curves,
        "mean_curves": {k: v.mean(1) for k, v in curves.items()},
        "metrics": {k: path_metrics(v.mean(1), points) for k, v in curves.items()},
        "per_draw_metrics": {k: [path_metrics(v[:, d], points) for d in range(3)] for k, v in curves.items()},
        "warm_floor_loss": {k: np.maximum(0, v.mean(1)[0] - v.mean(1).min(0)) for k, v in curves.items()},
        "health": observer_health["monitor"],
        "audit_health": observer_health["audit"],
        "losses": losses,
        "diet": diet["summary"],
        "updates": 4096,
    }
    for arm, source in {**baselines, "reservoir": record}.items():
        assert source["points"] == points
        for obs in source["observations"].values():
            assert obs["scores"] == cache[obs["encoder_fingerprint"]]["scores"]
        for kind in ("knn", "linear", "finetune"):
            v = np.array([source["observations"][str(p)]["scores"][kind] for p in points])
            np.testing.assert_array_equal(v, summary[arm]["reader_curves"][kind])
            np.testing.assert_allclose(
                path_metrics(v.mean(1), points), summary[arm]["metrics"][kind], atol=1e-12, rtol=0
            )
    primary = screen(*(summary[k]["metrics"]["finetune"] for k in ("reservoir", "fixed", "current")))
    draws = [
        screen(*(summary[k]["per_draw_metrics"]["finetune"][d] for k in ("reservoir", "fixed", "current")))
        for d in range(3)
    ]
    attempts = [read(p) for p in (out / "attempts").glob("*.json")]
    collection = [a for a in attempts if a.get("arm") and not a.get("replay")]
    rerun = [a for a in attempts if a.get("replay")]
    assert len(collection) == len(rerun) == 1 and all(a["complete"] for a in collection + rerun)
    assert all(a["ssl_started"] == a["ssl_completed"] == 4096 for a in collection + rerun)
    for kind in ("knn", "linear", "finetune"):
        assert collection[0][f"{kind}_completed"] == 3 * len(new) <= config.max_collection_fits_per_kind
    assert counts["knn_completed"] == counts["linear_completed"] == 3 * len(new)
    assert rerun[0]["finetune_completed"] == 2
    write(
        out / "summary.json",
        {
            "models": summary,
            "screen": primary,
            "per_draw_screens": draws,
            "controller_qualified": False,
            "source_streams": 1,
            "ssl_seeds": 1,
        },
    )
    write(
        out / "independent-verification.json",
        {
            "passed": True,
            "uniform_algorithm_r_rebuilt": True,
            "all_mixed_pixels_reconstructed": True,
            "all_current_subsets_match_fifo": True,
            "all_five_health_signals_crosschecked": True,
            "final_resident_pixels_rng_verified": True,
            "final_checkpoint_reembedding_and_losses_exact": True,
            "new_readout_states": len(new),
            "collection_work": collection,
            "rerun_work": rerun,
            "probe_verification_work": counts,
            "actual_new_ssl_updates": 8192,
            "inherited_ssl_updates": 8192,
            "screen": primary,
            "per_draw_screens": draws,
            "final_fingerprint": final_marker,
        },
    )
    check(lock["hashes"])
    check(read(out / "complete.json")["hashes"])
    print({"independent_verification": "passed", "screen": primary["passed"]}, flush=True)
