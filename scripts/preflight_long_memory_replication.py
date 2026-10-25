"""Qualify the reader budget from old results and test the finite GPU machinery."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time

import numpy as np
import torch
from omegaconf import OmegaConf

from cafl4ds.filters.base import FilterContext
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.reservoir_action_replication import restore
from cafl4ds.update_filtering import optimize_current
from scripts.natural_health_view import write
from scripts.prepare_long_memory_replication import CONFIG
from scripts.run_adaptation_experiment import _method
from scripts.run_health_measurement import reference_images
from scripts.run_long_memory_replication import initial_state, memory_for, observe, state
from scripts.run_paired_update_response import fingerprint
from scripts.run_preaction_stopping import ROOT, digest, read
from scripts.run_retention_anchor_signal import setup
from scripts.run_target_precision_budget import fit_and_score


def main():
    setup()
    study = OmegaConf.load(CONFIG)
    out = ROOT / study.study_root
    assert not (out / "input_lock.json").exists()
    assert not (out / "preflight.json").exists(), "Preserve previous preflight attempts"
    planning_path = ROOT / study.evaluation / "planning.json"
    planning = read(planning_path)
    eligible = [
        r
        for r in planning["candidates"]
        if r["support_precision_fraction"] >= 0.8 and r["crossed_precision_fraction"] >= 0.8
    ]
    assert eligible and min(r["panels"] for r in eligible) == study.endpoint_draws == 256
    assert planning["selected_panels"] == 256
    write(
        out / "reader-planning.json",
        {
            "selected_endpoint_draws": 256,
            "curve_draws": list(study.curve_draws),
            "archive_planning_sha256": digest(planning_path),
            "archived_candidates": planning["candidates"],
            "rule": "smallest archived budget with >=80% of domain contrasts meeting both precision requirements",
            "new_outcomes_used": False,
            "new_endpoint_precision_guaranteed": False,
        },
    )
    lock = read(ROOT / study.prior / "input_lock.json")
    lock["initial_state"] = str(ROOT / study.prior / "initial_state.pt")
    method = OmegaConf.create(lock["model_config"])
    template = _method(method).cuda()
    banks = {"monitor": reference_images(lock["monitor_reference"]), "audit": reference_images(lock["audit_reference"])}
    # The original inspected stream supplies discarded engineering inputs only.
    stream = OrderedFrames(read(ROOT / "outputs/natural-health-rate/20261016-frames/intake.json")["clips"])
    journal = {
        "ssl_started": 0,
        "ssl_completed": 0,
        "finetune_started": 0,
        "finetune_completed": 0,
        "engineering_only": True,
        "passed": False,
    }
    path = out / "preflight-attempt.json"
    write(path, journal)
    results, starts = {}, time.perf_counter()
    for arm in study.arms:
        for replay in (False, True):
            model, optimizer = restore(template, method.optim, initial_state(lock, 5003))
            memory = memory_for(study, arm)
            responses = []
            for step in range(10):
                pixels = torch.from_numpy(stream.batch(step))
                mixed = pixels if memory is None else memory.observe(pixels, FilterContext(model, step))
                journal["ssl_started"] += 1
                write(path, journal)
                responses.append(
                    list(optimize_current(model, optimizer, mixed.permute(0, 3, 1, 2).float().cuda() / 255, True))
                )
                journal["ssl_completed"] += 1
                write(path, journal)
            measured, _ = observe(model, optimizer, memory, banks, {}, 132, True)
            result = {
                "responses": responses,
                "measurement": measured,
                "state": fingerprint(state(model, optimizer, memory)),
            }
            if replay:
                assert result == results[arm]
            else:
                results[arm] = result
    images = torch.load(ROOT / study.evaluation / "images.pt", weights_only=True)
    with np.load(ROOT / study.evaluation / "data.npz") as stored:
        arrays = {k: stored[k] for k in stored.files}
    journal["finetune_started"] += 1
    write(path, journal)
    a, _ = fit_and_score(model.encoder, images, arrays, 61001)
    journal["finetune_completed"] += 1
    journal["finetune_started"] += 1
    write(path, journal)
    b, _ = fit_and_score(model.encoder, images, arrays, 61001)
    journal["finetune_completed"] += 1
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4063, 2)
    journal.update(
        passed=True,
        exact_three_arm_reexecution=True,
        observation_isolation=True,
        exact_reader_reexecution=True,
        elapsed_seconds=time.perf_counter() - starts,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),
    )
    write(path, journal)
    write(out / "preflight.json", journal)
    print(journal, flush=True)


if __name__ == "__main__":
    main()
