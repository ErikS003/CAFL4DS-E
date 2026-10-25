"""Collect and exactly replay the fixed long archive-order admission experiment."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.health_calibration import embeddings
from cafl4ds.natural_health_rate import NaturalHealthRate, OrderedFrames, paced_mask, path_metrics
from cafl4ds.short_buffer import capture, restore
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_dnd_health_transfer import data_inputs, readout, reserve
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, check, digest, read, write
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_precision_budget import fit_and_score

CONFIG = ROOT / "cafl4ds/configs/natural_health_rate.yaml"


def prepare(config, out):
    assert not out.exists() and not (ROOT / config.notebook).exists()
    prior, intake = ROOT / config.prior, ROOT / config.intake
    for name in ("input_lock.json", "complete.json", "presentation-verification.json", "delivery-verification.json"):
        record = read(prior / name)
        check(record["hashes"])
    assert read(prior / "independent-verification.json")["passed"]
    source = read(prior / "input_lock.json")
    frame_intake = read(ROOT / config.frames / "intake.json")
    assert frame_intake["complete"]
    intake_verification = intake / "independent-verification.json"
    assert read(intake_verification)["passed"]
    check(read(intake_verification)["hashes"])
    horizon = frame_intake["batches"]
    assert config.minimum_batches <= horizon <= config.max_batches
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "cafl4ds/natural_health_rate.py",
        ROOT / "cafl4ds/data/attributes.py",
        ROOT / "scripts/prepare_natural_health_rate.py",
        ROOT / "scripts/verify_natural_health_rate.py",
        ROOT / "scripts/report_natural_health_rate.py",
        ROOT / "tests/unit/test_natural_health_rate.py",
        ROOT / "docs/project-plan/natural-health-rate-frozen.md",
        intake / "preparation-lock.json",
        intake / "selection.json",
        intake / "metadata.json",
        intake_verification,
        ROOT / config.frames / "intake.json",
        ROOT / "uv.lock",
    ]
    for p, expected in read(intake / "preparation-lock.json")["hashes"].items():
        assert digest(p) == expected, p
        paths.append(Path(p))
    paths += [Path(p) for p in source["hashes"]]
    paths += [Path(p) for p in read(prior / "complete.json")["hashes"]]
    for record_name in ("presentation-verification.json", "delivery-verification.json"):
        paths += [Path(p) for p in read(prior / record_name)["hashes"]]
    paths += [Path(p) for p in read(intake_verification)["hashes"]]
    if (intake / "contact-first-six.png").exists():
        paths.append(intake / "contact-first-six.png")
    for name in (
        "input_lock.json",
        "complete.json",
        "independent-verification.json",
        "presentation-verification.json",
        "delivery-verification.json",
        "collection.json",
    ):
        paths.append(prior / name)
    for row in frame_intake["clips"]:
        paths.extend(
            [
                Path(row["frames_path"]),
                Path(row["frames_path"]).parent / "manifest.json",
                Path(row["frames_path"]).parent / "source.json",
                Path(row["frames_path"]).parent / "decoder.log",
            ]
        )
        assert digest(row["frames_path"]) == row["frames_sha256"]
    paths = sorted(set(paths))
    hashes = {str(p): digest(p) for p in paths}
    out.mkdir(parents=True)
    reserve(out)
    for p in paths:
        if p.suffix in (".py", ".yaml", ".md") and p.is_relative_to(ROOT):
            target = out / "source_snapshot" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    base = sorted({0, 132, horizon, *range(128, horizon + 1, 128)})
    write(
        out / "input_lock.json",
        {
            "hashes": hashes,
            "config": OmegaConf.to_container(config, resolve=True),
            "model_config": source["model_config"],
            "monitor_reference": source["monitor_reference"],
            "horizon": horizon,
            "base_points": base,
            "max_ssl_updates": 4 * horizon,
            "max_classifier_fits": 4 * (len(base) + 1) * 3,
            "frozen_unix": time.time(),
            "single_source_stream": True,
            "ssl_seeds": [int(config.seed)],
            "controller_qualified": False,
        },
    )
    print({"frozen": True, "batches": horizon, "files": len(hashes)}, flush=True)


def summarize(records, points, fork):
    rules = {}
    for arm, row in records.items():
        panel = np.array([row["observations"][str(p)]["panel_accuracy_pp"] for p in points])
        curve = panel.mean(1)
        rules[arm] = {
            "updates": row["updates"],
            "accepted_images": row["updates"] * 32,
            "reader_curves": panel.transpose(1, 0, 2),
            "reader_metrics": np.array([path_metrics(reader, points) for reader in panel.transpose(1, 0, 2)]),
            "curve": curve,
            "metrics": path_metrics(curve, points),
            "gain_over_warm": curve[-1] - curve[0],
        }
    full = np.asarray(rules["full"]["curve"])
    full_readers = np.asarray(rules["full"]["reader_curves"])
    for arm, curve, count in (
        ("frozen", np.repeat(full[:1], len(points), axis=0), 0),
        (
            "permanent",
            np.array([full[points.index(min(p, fork))] for p in points]) if fork is not None else full,
            fork if fork is not None else points[-1],
        ),
    ):
        readers = (
            np.repeat(full_readers[:, :1], len(points), axis=1)
            if arm == "frozen"
            else full_readers[:, [points.index(min(p, fork)) for p in points]]
            if fork is not None
            else full_readers
        )
        rules[arm] = {
            "updates": count,
            "accepted_images": count * 32,
            "reader_curves": readers,
            "reader_metrics": np.array([path_metrics(reader, points) for reader in readers]),
            "curve": curve,
            "metrics": path_metrics(curve, points),
            "gain_over_warm": curve[-1] - curve[0],
        }
    contrasts = {}
    for arm in ("full", "permanent", "quarter", "paced"):
        a, b = rules["adaptive"], rules[arm]
        contrasts[arm] = {
            "final_difference_pp": a["curve"][-1] - b["curve"][-1],
            "drawdown_reduction_pp": b["metrics"][-1] - a["metrics"][-1],
        }
    return {
        "points": points,
        "first_alarm": fork,
        "rules": rules,
        "contrasts": contrasts,
        "exact_work_matched": rules["adaptive"]["updates"] == rules["paced"]["updates"],
        "ordered_streams": 1,
        "ssl_seeds": 1,
        "controller_qualified": False,
        "uncertainty_intervals": None,
    }


def run_arm(
    config,
    out,
    arm,
    mask,
    initial,
    method,
    start,
    images,
    arrays,
    bank,
    stream,
    points,
    expected_fork,
    cache,
    journal,
    journal_path,
    replay,
):
    directory = out / arm
    if not replay:
        assert not directory.exists(), "Preserve incomplete scientific branches"
        directory.mkdir()
    target = read(directory / "record.json") if replay else None
    stored = np.load(directory / "embeddings.npz") if replay else None
    model, optimizer = restore(initial, method.optim, start)
    controller = instantiate(config.controller)
    observations, banks, monitors, trace, accepted, rates = {}, {}, {}, [], [], []
    fork, skip_checks, monitor_seconds = None, 0, 0.0
    horizon = read(out / "input_lock.json")["horizon"]
    actual_points = set(points)

    def observe(step):
        nonlocal fork, monitor_seconds
        marker = fingerprint(capture(model, optimizer))
        tick = time.perf_counter()
        z = embeddings(model, bank)
        assert marker == fingerprint(capture(model, optimizer)), "Monitor changed learner state"
        scheduled = NaturalHealthRate.scheduled(step)
        fired = controller.observe(step, z) if scheduled else False
        elapsed = time.perf_counter() - tick
        if scheduled:
            monitor_seconds += elapsed
            monitors[str(step)] = {"learner_fingerprint": marker, "health_alarm": fired, "seconds": elapsed}
        if fired and fork is None:
            fork = step
            if arm == "full":
                actual_points.add(step)
            else:
                assert fork == expected_fork, "Shared causal first alarm changed"
        banks[str(step)] = z
        if step in actual_points:
            destination = Path(cache[marker]) if marker in cache else out / "readouts" / marker
            panel = readout(config, out, destination, model, optimizer, images, arrays, journal, journal_path, replay)
            for draw in config.draws:
                assert read(destination / f"panel_{draw}.json")["learner_fingerprint"] == marker
            cache[marker] = str(destination)
            observations[str(step)] = {
                "learner_fingerprint": marker,
                "encoder_fingerprint": fingerprint(model.encoder.state_dict()),
                "readout": str(destination),
                "panel_accuracy_pp": panel,
            }
            if not replay:
                write(out / "readout_cache.json", cache)
        assert marker == fingerprint(capture(model, optimizer))
        if replay:
            np.testing.assert_array_equal(z, stored[str(step)])
            if scheduled:
                assert marker == target["monitors"][str(step)]["learner_fingerprint"]
                assert fired == target["monitors"][str(step)]["health_alarm"]
            if step in actual_points:
                expected = target["observations"][str(step)]
                assert marker == expected["learner_fingerprint"]
                np.testing.assert_array_equal(
                    observations[str(step)]["panel_accuracy_pp"], expected["panel_accuracy_pp"]
                )
            if (arm, step) in (("full", 132), ("paced", horizon)):
                journal["replay_fits_started"] += 1
                write(journal_path, journal)
                with torch.random.fork_rng(devices=[0]):
                    probability, _ = fit_and_score(model.encoder, images, arrays, 61001)
                with np.load(Path(observations[str(step)]["readout"]) / "panel_61001.npz") as saved:
                    np.testing.assert_array_equal(probability, saved["probabilities"])
                assert marker == fingerprint(capture(model, optimizer))
                journal["replay_fits"] += 1
                write(journal_path, journal)

    observe(0)
    for step in range(1, horizon + 1):
        causal_take, causal_rate = controller.admit(step)
        take = causal_take if arm == "adaptive" else bool(mask[step - 1])
        rate = (
            causal_rate
            if arm == "adaptive"
            else (
                1.0
                if expected_fork is None or step <= expected_fork or arm == "full"
                else 0.25
                if arm == "quarter"
                else float(mask[expected_fork:].mean())
            )
        )
        before = fingerprint(capture(model, optimizer)) if not take else None
        tick = time.perf_counter()
        input_digest = None
        if take:
            journal["ssl_started"] += 1
            write(journal_path, journal)
            pixels = stream.batch(step - 1)
            input_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            batch = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().cuda() / 255
            response = optimize_current(model, optimizer, batch, True)
            journal["ssl_completed"] += 1
            write(journal_path, journal)
        else:
            response = None
            assert before == fingerprint(capture(model, optimizer)), "Reject changed model/Adam/RNG"
            skip_checks += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        accepted.append(take)
        rates.append(rate)
        row = {
            "step": step,
            "accepted": take,
            "response": list(response) if response else None,
            "seconds": elapsed,
            "skipped_state_fingerprint": before,
            "input_sha256": input_digest,
        }
        trace.append(row)
        if replay:
            expected = target["trace"][step - 1]
            assert all(
                row[k] == expected[k]
                for k in ("step", "accepted", "response", "skipped_state_fingerprint", "input_sha256")
            )
        if step in actual_points or NaturalHealthRate.scheduled(step):
            observe(step)
        if step % 64 == 0 or step == horizon:
            reserve(out)
            write(
                out / "status.json",
                {
                    "stage": "replay" if replay else "collect",
                    "arm": arm,
                    "step": step,
                    "horizon": horizon,
                    "accepted_batches_current_arm": sum(accepted),
                    "first_alarm": fork,
                    "current_target_rate": rate,
                    "confirmed_health_alarms": sum(event["alarm"] for event in controller.events),
                    "ssl_completed": journal["ssl_completed"],
                    "fits_completed": journal["fits_completed"],
                    "free_bytes": shutil.disk_usage(out).free,
                },
            )
    state = capture(model, optimizer)
    record = {
        "arm": arm,
        "execution": "physical",
        "accepted": accepted,
        "rates": rates,
        "updates": sum(accepted),
        "first_alarm": fork,
        "observations": observations,
        "monitors": monitors,
        "events": controller.events,
        "trace": trace,
        "skips_verified": skip_checks,
        "monitor_seconds": monitor_seconds,
        "final_fingerprint": fingerprint(state),
        "initial_fingerprint": fingerprint(start),
        "state_path": str(directory / "final_state.pt"),
        "embedding_path": str(directory / "embeddings.npz"),
        "controller_deployed": arm == "adaptive",
    }
    if replay:
        for key in (
            "accepted",
            "rates",
            "events",
            "updates",
            "first_alarm",
            "final_fingerprint",
            "initial_fingerprint",
        ):
            assert record[key] == target[key], key
        assert set(banks) == set(stored.files)
        stored.close()
        record = target
    else:
        reserve(out)
        torch.save(state, directory / "final_state.pt")
        np.savez_compressed(directory / "embeddings.npz", **banks)
        write(directory / "record.json", record)
        paths = [directory / "record.json", directory / "final_state.pt", directory / "embeddings.npz"]
        write(directory / "complete.json", {"hashes": {str(p): digest(p) for p in paths}})
    print({"replay": replay, "arm": arm, "updates": record["updates"], "first_alarm": fork}, flush=True)
    return record


def collect(config, out, replay=False):
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    images, arrays, bank, intake = data_inputs(config, lock)
    stream = OrderedFrames(intake["clips"])
    method = OmegaConf.create(lock["model_config"])
    torch.manual_seed(11)
    initial = _method(method).cuda()
    if replay:
        start = torch.load(out / "initial_state.pt", map_location="cpu", weights_only=True)
    else:
        assert not (out / "initial_state.pt").exists(), "Preserve an incomplete collection's initial state"
        torch.manual_seed(config.seed)
        optimizer = instantiate(method.optim, params=list(initial.parameters()))
        start = capture(initial, optimizer)
        torch.save(start, out / "initial_state.pt")
    previous = read(ROOT / config.prior / "collection.json")["records"]
    cache = {o["learner_fingerprint"]: o["readout"] for r in previous for o in r["observations"].values()}
    if (out / "readout_cache.json").exists():
        cache.update(read(out / "readout_cache.json"))
    attempts = out / "attempts"
    attempts.mkdir(exist_ok=True)
    journal_path = attempts / f"{'replay' if replay else 'collect'}_{os.getpid()}_{time.time_ns()}.json"
    journal = {
        "ssl_started": 0,
        "ssl_completed": 0,
        "fits_started": 0,
        "fits_completed": 0,
        "cached_fits": 0,
        "replay_fits": 0,
        "replay_fits_started": 0,
        "replay": replay,
    }
    write(journal_path, journal)
    records, fork, points = {}, None, lock["base_points"]
    horizon = lock["horizon"]
    try:
        for arm in config.arms:
            count = (
                0
                if fork is None
                else (horizon - fork) // 4
                if arm == "quarter"
                else sum(records["adaptive"]["accepted"][fork:])
                if arm == "paced" and fork is not None
                else 0
            )
            mask = (
                None
                if arm == "adaptive" and fork is not None
                else (np.ones(horizon, dtype=bool) if arm in ("full", "adaptive") else paced_mask(horizon, fork, count))
            )
            # No full-path alarm implies identical adaptive states by induction: admission never changes.
            prior = next((r for r in records.values() if mask is not None and r["accepted"] == mask.tolist()), None)
            path = out / arm / "record.json"
            if replay and read(path)["execution"] == "mask_identical_alias":
                record = read(path)
                assert prior is not None and record["accepted"] == prior["accepted"]
                assert record["final_fingerprint"] == prior["final_fingerprint"]
                if arm == "paced":
                    journal["replay_fits_started"] += 1
                    write(journal_path, journal)
                    model, optimizer = restore(
                        initial,
                        method.optim,
                        torch.load(Path(record["state_path"]), map_location="cpu", weights_only=True),
                    )
                    with torch.random.fork_rng(devices=[0]):
                        probability, _ = fit_and_score(model.encoder, images, arrays, 61001)
                    with np.load(Path(record["observations"][str(horizon)]["readout"]) / "panel_61001.npz") as saved:
                        np.testing.assert_array_equal(probability, saved["probabilities"])
                    assert fingerprint(capture(model, optimizer)) == record["final_fingerprint"]
                    journal["replay_fits"] += 1
                    write(journal_path, journal)
            elif not replay and prior is not None:
                directory = out / arm
                directory.mkdir()
                record = {
                    **copy.deepcopy(prior),
                    "arm": arm,
                    "execution": "mask_identical_alias",
                    "trajectory_from": prior["arm"],
                    "controller_deployed": arm == "adaptive",
                }
                record["rates"] = [
                    1.0 if fork is None or step <= fork else (0.25 if arm == "quarter" else count / (horizon - fork))
                    for step in range(1, horizon + 1)
                ]
                write(path, record)
                write(directory / "complete.json", {"hashes": {str(path): digest(path)}})
            else:
                record = run_arm(
                    config,
                    out,
                    arm,
                    mask,
                    initial,
                    method,
                    start,
                    images,
                    arrays,
                    bank,
                    stream,
                    points,
                    fork,
                    cache,
                    journal,
                    journal_path,
                    replay,
                )
            records[arm] = record
            if arm == "full":
                fork = record["first_alarm"]
                points = sorted(set(points) | ({fork} if fork is not None else set()))
        journal["complete"] = True
        write(journal_path, journal)
    except Exception as error:
        journal["error"] = repr(error)
        write(journal_path, journal)
        raise
    check(lock["hashes"])
    if replay:
        assert journal["replay_fits"] == journal["replay_fits_started"] == 2
        write(
            out / "training-replay-verification.json",
            {
                "passed": True,
                "ssl_updates": journal["ssl_completed"],
                "classifier_fit_replays": journal["replay_fits"],
                "logical_branches": 4,
                "physical_branches": sum(r["execution"] == "physical" for r in records.values()),
            },
        )
    else:
        assert journal["ssl_started"] == journal["ssl_completed"] <= lock["max_ssl_updates"]
        assert journal["fits_started"] == journal["fits_completed"] <= lock["max_classifier_fits"] <= 432
        write(
            out / "collection.json",
            {"records": records, "physical_work": journal, "points": points, "first_alarm": fork},
        )
        write(out / "summary.json", summarize(records, points, fork))
        paths = [
            out / "collection.json",
            out / "summary.json",
            out / "initial_state.pt",
            out / "readout_cache.json",
            *out.glob("*/complete.json"),
            *out.glob("readouts/*/*"),
            *attempts.glob("collect_*.json"),
        ]
        write(out / "complete.json", {"hashes": {str(p): digest(p) for p in paths if p.is_file()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "collect", "replay"), required=True)
    args = parser.parse_args()
    setup()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    if args.stage == "prepare":
        prepare(config, out)
    else:
        assert not (
            out / ("complete.json" if args.stage == "collect" else "training-replay-verification.json")
        ).exists()
        if args.stage == "replay":
            check(read(out / "complete.json")["hashes"])
        collect(config, out, args.stage == "replay")


if __name__ == "__main__":
    main()
