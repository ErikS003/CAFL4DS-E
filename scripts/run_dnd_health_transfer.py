"""Fixed-guard transfer across new histories, with no transition metadata in control."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.dnd_health_transfer import MONITOR, POINTS, class_accuracy
from cafl4ds.health_calibration import embeddings
from cafl4ds.short_buffer import capture, restore
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, plain, read
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_precision_budget import fit_and_score

CONFIG = ROOT / "cafl4ds/configs/dnd_health_transfer.yaml"


def write(path, value):
    atomic_json(Path(path), plain(value))


def reserve(out):
    assert shutil.disk_usage(out).free >= 512 * 2**20, "Scientific write reserve threatened"


def prepare(config, out):
    assert not out.exists(), "Use a new unused output directory"
    intake = ROOT / config.intake
    audit = read(intake / "independent-verification.json")
    assert audit["passed"]
    check(audit["hashes"])
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/prepare_dnd_health_transfer.py",
        ROOT / "scripts/verify_dnd_transfer_intake.py",
        ROOT / "scripts/verify_dnd_health_transfer.py",
        ROOT / "scripts/report_dnd_health_transfer.py",
        ROOT / "cafl4ds/dnd_health_transfer.py",
        ROOT / "cafl4ds/dnd_health_stopping.py",
        ROOT / "tests/unit/test_dnd_health_transfer.py",
        ROOT / "docs/project-plan/dnd-health-transfer-frozen.md",
        ROOT / "uv.lock",
    ]
    for name in (
        "scripts/run_preaction_stopping.py",
        "scripts/run_regime_windows_experiment.py",
        "scripts/run_target_precision_budget.py",
        "scripts/run_adaptation_experiment.py",
        "scripts/run_retention_anchor_signal.py",
        "scripts/run_paired_update_response.py",
        "scripts/run_health_measurement.py",
        "scripts/run_short_buffer_experiment.py",
        "cafl4ds/short_buffer.py",
        "cafl4ds/update_filtering.py",
        "cafl4ds/health_calibration.py",
        "cafl4ds/measurements.py",
        "cafl4ds/models/vit.py",
        "cafl4ds/ssl/mae.py",
        "cafl4ds/models/vit_mae_pretrained.py",
    ):
        paths.append(ROOT / name)
    for name in (
        "input_lock.json",
        "independent-verification.json",
        "data-role-manifest.json",
        "data.npz",
        "images.pt",
    ):
        paths.append(ROOT / config.evaluation / name)
    assert read(ROOT / config.evaluation / "independent-verification.json")["passed"]
    historical = read(ROOT / config.evaluation / "input_lock.json")["hashes"]
    historical = {str(Path(k)): v for k, v in historical.items()}
    check(
        {str(p): historical[str(p)] for p in paths if str(p) in historical and p.suffix not in (".py", ".md", ".yaml")}
    )
    original = read(ROOT / config.original / "input_lock.json")
    frozen_guard = ROOT / "cafl4ds/dnd_health_stopping.py"
    prior_guard_lock = read(ROOT / "outputs/dnd-health-stopping/20261016-development/input_lock.json")
    assert digest(frozen_guard) == prior_guard_lock["hashes"][str(frozen_guard)], "Frozen guard implementation changed"
    paths += [
        ROOT / config.original / "input_lock.json",
        Path(original["config"]["well"]),
        ROOT / "outputs/dnd-health-stopping/20261016-development/input_lock.json",
    ]
    reference = read(ROOT / config.monitor / "input_lock.json")["reference"]
    paths += [ROOT / config.monitor / n for n in ("input_lock.json", "independent-verification.json")]
    assert read(ROOT / config.monitor / "independent-verification.json")["passed"]
    paths += [Path(r["path"]) for r in reference["images"]]
    paths += [
        intake / "independent-verification.json",
        intake / "preparation-lock.json",
        intake / "selection.json",
        ROOT / config.frames / "intake.json",
    ]
    paths += [Path(p) for p in audit["hashes"]]
    hashes = {str(p): digest(p) for p in sorted(set(paths))}
    out.mkdir(parents=True)
    reserve(out)
    for path in sorted(set(paths)):
        if path.suffix in (".py", ".md", ".yaml") and path.is_relative_to(ROOT):
            dest = out / "source_snapshot" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    write(
        out / "input_lock.json",
        {
            "hashes": hashes,
            "config": OmegaConf.to_container(config, resolve=True),
            "model_config": original["config"],
            "monitor_reference": reference,
            "prepared_unix": time.time(),
            "frozen_before_new_learner_outcomes": True,
            "guard_unmodified": True,
            "new_training_histories": 8,
            "reused_evaluation_pool": True,
        },
    )
    print({"prepared": True, "frozen_inputs": len(hashes)}, flush=True)


def data_inputs(config, lock):
    images = torch.load(ROOT / config.evaluation / "images.pt", weights_only=True)
    with np.load(ROOT / config.evaluation / "data.npz") as a:
        arrays = {k: a[k] for k in a.files}
    assert len(arrays["query"]) == 3663
    for draw in config.draws:
        selected = arrays[f"support_{draw}"]
        assert len(selected) == 50 and len(np.intersect1d(selected, arrays["query"])) == 0
        assert [(arrays["labels"][selected] == c).sum() for c in (0, 1)] == [25, 25]
    bank = reference_images(lock["monitor_reference"])
    return images, arrays, bank, read(ROOT / config.frames / "intake.json")


def batch_reader(history, clips):
    ordered = [c for period in history["periods"] for c in period]
    caches = [np.load(clips[c]["frames_path"], mmap_mode="r", allow_pickle=False) for c in ordered]

    def batch(step):
        frames = caches[step // 16][(step % 16) * 32 : (step % 16 + 1) * 32]
        return torch.from_numpy(np.array(frames, copy=True)).permute(0, 3, 1, 2).float() / 255

    return batch


def readout(config, out, directory, model, optimizer, images, arrays, journal, journal_path, replay=False):
    marker = fingerprint(capture(model, optimizer))
    encoder_marker = fingerprint(model.encoder.state_dict())
    scores = []
    if not replay:
        directory.mkdir(parents=True, exist_ok=True)
    for draw in config.draws:
        path = directory / f"panel_{draw}.npz"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            record = read(meta)
            assert digest(path) == record["sha256"] and record["encoder_fingerprint"] == encoder_marker
            with np.load(path) as a:
                probabilities = a["probabilities"]
            journal["cached_fits"] += 1
        else:
            assert not replay, "Missing primary fit during replay"
            assert not path.exists() and not meta.exists(), "Preserve incomplete fit artifacts for repair"
            reserve(out)
            journal["fits_started"] += 1
            write(journal_path, journal)
            tick = time.perf_counter()
            with torch.random.fork_rng(devices=[0]):
                probabilities, audit = fit_and_score(model.encoder, images, arrays, int(draw))
            assert probabilities.shape == (4063, 2) and np.isfinite(probabilities).all()
            assert marker == fingerprint(capture(model, optimizer)), "Readout changed online model/Adam/RNG"
            np.savez_compressed(path, probabilities=probabilities)
            write(
                meta,
                {
                    "sha256": digest(path),
                    "encoder_fingerprint": encoder_marker,
                    "learner_fingerprint": marker,
                    "draw": int(draw),
                    "fine_tuning": audit,
                    "seconds": time.perf_counter() - tick,
                },
            )
            journal["fits_completed"] += 1
            write(journal_path, journal)
        scores.append(
            class_accuracy(probabilities[:3663], arrays["labels"][arrays["query"]], arrays["domains"][arrays["query"]])
        )
    assert marker == fingerprint(capture(model, optimizer)), "Reading cached outcomes changed learner"
    return np.array(scores)


def collect(config, out, replay=False):
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    images, arrays, bank, intake = data_inputs(config, lock)
    method = OmegaConf.create(lock["model_config"])
    torch.manual_seed(11)
    initial = _method(method).cuda()
    sources = {c["clip_id"]: c for c in intake["clips"]}
    history_records = []
    started = time.perf_counter()
    attempts = out / "attempts"
    attempts.mkdir(exist_ok=True)
    journal_path = attempts / f"{'replay' if replay else 'collect'}_{os.getpid()}_{time.time_ns()}.json"
    journal = {
        "replay": replay,
        "ssl_started": 0,
        "ssl_completed": 0,
        "fits_started": 0,
        "fits_completed": 0,
        "cached_fits": 0,
        "replay_fits": 0,
    }
    write(journal_path, journal)
    replay_checks = 0
    try:
        for source_index, history in enumerate(intake["histories"]):
            name = history["name"]
            directory = out / name
            if not replay:
                directory.mkdir(exist_ok=True)
            if not replay and (directory / "complete.json").exists():
                check(read(directory / "complete.json")["hashes"])
                history_records.append(read(directory / "record.json"))
                continue
            target = read(directory / "record.json") if replay else None
            stored = np.load(directory / "embeddings.npz") if replay else None
            torch.manual_seed(config.seed)
            optimizer = instantiate(method.optim, params=list(initial.parameters()))
            model, optimizer = restore(initial, method.optim, capture(initial, optimizer))
            guard = instantiate(config.guard)
            get_batch = batch_reader(history, sources)
            seen, features, trace = {}, {}, []
            selected_state = None

            def observe(step):
                nonlocal selected_state, replay_checks
                marker = fingerprint(capture(model, optimizer))
                tick = time.perf_counter()
                z = embeddings(model, bank)
                forward_seconds = time.perf_counter() - tick
                assert marker == fingerprint(capture(model, optimizer)), "Monitor changed learner"
                tick = time.perf_counter()
                active_observation = step in MONITOR and guard.stop is None
                if step in MONITOR:
                    fired = guard.observe(step, z, activation=132, kind="combined")
                    if fired and selected_state is None:
                        selected_state = capture(model, optimizer)
                geometry_seconds = time.perf_counter() - tick
                readout_dir = out / "warm" if step == 0 else directory / f"at_{step}"
                panel = readout(
                    config, out, readout_dir, model, optimizer, images, arrays, journal, journal_path, replay
                )
                seen[str(step)] = {
                    "learner_fingerprint": marker,
                    "encoder_fingerprint": fingerprint(model.encoder.state_dict()),
                    "readout": str(readout_dir),
                    "panel_accuracy_pp": panel,
                    "controller_observed": active_observation,
                    "on_controller_schedule": step in MONITOR,
                    "forward_seconds": forward_seconds,
                    "guard_seconds": geometry_seconds,
                }
                features[str(step)] = z
                if replay:
                    np.testing.assert_array_equal(z, stored[str(step)])
                    assert marker == target["observations"][str(step)]["learner_fingerprint"]
                    np.testing.assert_array_equal(panel, target["observations"][str(step)]["panel_accuracy_pp"])
                    replay_checks += 1
                    if (source_index, step) in ((0, 148), (7, 264)):
                        with torch.random.fork_rng(devices=[0]):
                            actual, _ = fit_and_score(model.encoder, images, arrays, 61001)
                        with np.load(readout_dir / "panel_61001.npz") as a:
                            np.testing.assert_array_equal(actual, a["probabilities"])
                        assert marker == fingerprint(capture(model, optimizer))
                        journal["replay_fits"] += 1
                        write(journal_path, journal)
                else:
                    total_completed = sum(read(p).get("fits_completed", 0) for p in attempts.glob("collect_*.json"))
                    write(
                        out / "status.json",
                        {
                            "stage": "collect",
                            "history": name,
                            "histories_completed": source_index,
                            "step": step,
                            "new_fits_completed": total_completed,
                            "planned_fits": 1155,
                            "free_bytes": shutil.disk_usage(out).free,
                        },
                    )
                assert marker == fingerprint(capture(model, optimizer))

            observe(0)
            ssl_seconds = 0
            ssl_update_seconds = []
            for step in range(1, 449):
                journal["ssl_started"] += 1
                write(journal_path, journal)
                tick = time.perf_counter()
                trace.append(list(optimize_current(model, optimizer, get_batch(step - 1).cuda(), True)))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                ssl_seconds += elapsed
                ssl_update_seconds.append(elapsed)
                journal["ssl_completed"] += 1
                write(journal_path, journal)
                if step in POINTS:
                    observe(step)
            stop = guard.stop or 448
            record = {
                "name": name,
                "observations": seen,
                "stop": stop,
                "alarm_fired": guard.stop is not None,
                "events": guard.events,
                "loss_gradients": trace,
                "ssl_seconds": ssl_seconds,
                "ssl_update_seconds": ssl_update_seconds,
                "day_updates": history["day_updates"],
                "night_updates": history["night_updates"],
            }
            if replay:
                assert trace == target["loss_gradients"] and guard.events == target["events"] and stop == target["stop"]
                stored.close()
            else:
                reserve(out)
                np.savez_compressed(directory / "embeddings.npz", **features)
                if selected_state is None:
                    selected_state = capture(model, optimizer)
                branch, branch_optimizer = restore(initial, method.optim, selected_state)
                marker = fingerprint(capture(branch, branch_optimizer))
                for step in range(stop, 448):
                    assert optimize_current(branch, branch_optimizer, get_batch(step), False) is None
                assert marker == fingerprint(capture(branch, branch_optimizer))
                assert marker == seen[str(stop)]["learner_fingerprint"]
                torch.save(selected_state, directory / "stopped_state.pt")
                record["hold_verification"] = {
                    "passed": True,
                    "skipped_batches": 448 - stop,
                    "state_fingerprint": marker,
                }
                write(directory / "record.json", record)
                files = [
                    directory / "record.json",
                    directory / "embeddings.npz",
                    directory / "stopped_state.pt",
                    *directory.glob("at_*/*"),
                ]
                write(
                    directory / "complete.json",
                    {"complete": True, "hashes": {str(p): digest(p) for p in files if p.is_file()}},
                )
                history_records.append(record)
                del branch, branch_optimizer
            print(
                {
                    "replay": replay,
                    "history": name,
                    "stop": stop,
                    "alarm_fired": guard.stop is not None,
                    "ssl_work": journal["ssl_completed"],
                    "new_fits": journal["fits_completed"],
                },
                flush=True,
            )
    except Exception as error:
        journal["error"] = str(error)
        write(journal_path, journal)
        raise
    journal["complete"] = True
    write(journal_path, journal)
    check(lock["hashes"])
    if replay:
        assert journal["ssl_completed"] == 3584 and journal["replay_fits"] == 2
        write(
            out / "training-replay-verification.json",
            {
                "passed": True,
                "ssl_updates": 3584,
                "dense_states": replay_checks,
                "exact_classifier_fit_replays": 2,
                "seconds": time.perf_counter() - started,
            },
        )
    else:
        all_attempts = [read(p) for p in attempts.glob("collect_*.json")]
        work = {
            "ssl_updates": sum(j["ssl_completed"] for j in all_attempts),
            "classifier_fits": sum(j["fits_completed"] for j in all_attempts),
            "started_fits": sum(j["fits_started"] for j in all_attempts),
        }
        assert work["classifier_fits"] == work["started_fits"] == 1155
        write(
            out / "collection.json",
            {"records": history_records, "work": work, "seconds": time.perf_counter() - started},
        )
        full = np.array(
            [
                [np.mean(r["observations"][str(step)]["panel_accuracy_pp"], axis=0) for step in POINTS]
                for r in history_records
            ]
        )
        study = instantiate(config.study)
        summary, shuffle = study.summarize(
            full,
            [r["stop"] for r in history_records],
            [(h["day_updates"], h["night_updates"]) for h in intake["histories"]],
        )
        summary["alarm_fired_histories"] = [r["name"] for r in history_records if r["alarm_fired"]]
        write(out / "summary.json", summary)
        np.savez_compressed(out / "shuffle_assignments.npz", **shuffle)
        files = [
            out / "collection.json",
            out / "summary.json",
            out / "shuffle_assignments.npz",
            *out.glob("transfer_*/complete.json"),
            *out.glob("warm/*"),
            *attempts.glob("collect_*.json"),
        ]
        write(out / "complete.json", {"complete": True, "hashes": {str(p): digest(p) for p in files if p.is_file()}})
        write(out / "status.json", {"complete": True, **work, "histories": 8})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "collect", "replay", "all"), default="all")
    args = parser.parse_args()
    setup()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    if args.stage in ("prepare", "all"):
        prepare(config, out)
    if args.stage in ("collect", "all"):
        assert not (out / "complete.json").exists()
        collect(config, out)
    if args.stage in ("replay", "all"):
        assert not (out / "training-replay-verification.json").exists()
        check(read(out / "complete.json")["hashes"])
        collect(config, out, True)


if __name__ == "__main__":
    main()
