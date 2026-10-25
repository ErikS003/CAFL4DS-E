"""Freeze and run a cached, parent-only cumulative-effect forecasting pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.h8_health_information import geometry_features
from cafl4ds.preaction_stopping import past_losses

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/preaction_stopping.yaml"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def check(hashes):
    for path, expected in hashes.items():
        assert digest(path) == expected, path


def collect(config):
    """Keep causal parent observations separate from offline response targets."""
    ref = ROOT / config.reference
    with np.load(ref / "warm_features.npz") as a:
        warm = a["bank"]
    with np.load(ref / "reference_targets.npz") as a:
        domains = a["domains"]
    losses, geometry = [], []
    for source in config.sources:
        directory = ROOT / config.guard / "runs" / source / "seed_5003"
        with np.load(directory / "banks.npz") as bank:
            geometry.append(geometry_features(bank["128"], warm, domains))
        losses.append(past_losses(read(directory / "parent-path.json")["trace"]))
    stage = read(ROOT / config.outcomes / "stage_512.json")
    assert [e["source"] for e in stage["episodes"]] == list(config.sources)
    truth, lower, upper = [], [], []
    for e in stage["episodes"]:
        assert [p["step"] for p in e["points"]] == [128, 136, 144, 152, 160]
        truth.append([p["acquired"]["mean_gain_pp"][1:] for p in e["points"][1:]])
        lower.append([p["acquired"]["crossed"]["lower"][1:] for p in e["points"][1:]])
        upper.append([p["acquired"]["crossed"]["upper"][1:] for p in e["points"][1:]])
    return {"losses": np.array(losses), "geometry": np.array(geometry)}, {
        "gain": np.array(truth),
        "lower": np.array(lower),
        "upper": np.array(upper),
    }


def prepare(config, out):
    if out.exists():
        raise ValueError("Use a new unused output directory")
    scientific = [
        CONFIG,
        Path(__file__),
        ROOT / "cafl4ds/preaction_stopping.py",
        ROOT / "cafl4ds/collapse_risk_regression.py",
        ROOT / "cafl4ds/h8_health_information.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "scripts/verify_preaction_stopping.py",
        ROOT / "scripts/report_preaction_stopping.py",
        ROOT / "tests/unit/test_preaction_stopping.py",
        ROOT / "docs/project-plan/preaction-stopping-frozen.md",
        ROOT / "uv.lock",
    ]
    inherited = []
    for key in ("outcomes", "health"):
        directory = ROOT / config[key]
        assert read(directory / "independent-verification.json")["passed"]
        inherited += [directory / n for n in ("input_lock.json", "complete.json", "independent-verification.json")]
    outcome = ROOT / config.outcomes
    inherited += [outcome / n for n in ("stage_512.json", "summary.json", "delivery-verification.json")]
    assert read(outcome / "delivery-verification.json")["passed"]
    check(
        {
            k: v
            for k, v in read(outcome / "complete.json")["hashes"].items()
            if Path(k).name in ("stage_512.json", "summary.json")
        }
    )
    inherited += [ROOT / config.health / "regression_data.npz"]
    check(read(ROOT / config.health / "complete.json")["hashes"])
    inherited += [ROOT / config.reference / n for n in ("warm_features.npz", "reference_targets.npz")]
    for source in config.sources:
        inherited += [
            ROOT / config.guard / "runs" / source / "seed_5003" / n for n in ("banks.npz", "parent-path.json")
        ]
    historical = {
        **read(outcome / "input_lock.json")["hashes"],
        **read(ROOT / config.health / "input_lock.json")["hashes"],
    }
    check({str(p): historical[str(p)] for p in inherited if str(p) in historical})
    hashes = {str(p): digest(p) for p in scientific + inherited}
    out.mkdir(parents=True)
    for path in scientific:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write(
        out / "input_lock.json",
        {
            "hashes": hashes,
            "config": OmegaConf.to_container(config, resolve=True),
            "frozen_before_new_model_fitting": True,
            "already_inspected_outcome_population": True,
            "sources": list(config.sources),
            "prepared_unix": time.time(),
            "scope": "exploratory_reused_history_exclusion_not_fresh_confirmation",
        },
    )
    print({"prepared": True, "frozen_inputs": len(hashes)}, flush=True)


def analyze(config, out):
    assert not (out / "complete.json").exists(), "Completed science is immutable"
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    started = time.perf_counter()
    inputs, targets = collect(config)
    # Independently audited prior parent readings are only an integrity cross-check.
    with np.load(ROOT / config.health / "regression_data.npz") as old:
        for i, source in enumerate(config.sources):
            row = np.flatnonzero((old["groups"] == source) & (old["contexts"] == "day_end")).item()
            np.testing.assert_allclose(inputs["geometry"][i], old["geometry"][row, 0], atol=1e-12)
            np.testing.assert_allclose(inputs["losses"][i], old["controls"][row, -2:], atol=1e-12)
    np.savez_compressed(out / "causal_inputs.npz", **inputs)
    np.savez_compressed(out / "offline_targets.npz", **targets)
    study = instantiate(config.study)
    fitted = study.crossfit(inputs["losses"], inputs["geometry"], targets["gain"])
    write(out / "forecasts.json", fitted)
    stage = read(ROOT / config.outcomes / "stage_512.json")
    oracle = stage["additional_diagnostics"]["acquired_conditional"]["informed_stop"]["blocks"]
    summary = study.summarize(targets["gain"], fitted, targets["lower"], targets["upper"], oracle)
    summary.update(
        {
            "sources": list(config.sources),
            "inherited_precision_pass": read(ROOT / config.outcomes / "summary.json")["precision_pass"],
            "seconds": time.perf_counter() - started,
            "scope": "parent_only_forecasts_on_reused_histories_not_fresh_confirmation",
        }
    )
    write(out / "summary.json", summary)
    check(lock["hashes"])
    artifacts = [out / n for n in ("causal_inputs.npz", "offline_targets.npz", "forecasts.json", "summary.json")]
    write(out / "complete.json", {"complete": True, "hashes": {str(p): digest(p) for p in artifacts}})
    print(
        {"complete": True, "prediction_gate": summary["exploratory_prediction_gate"], "seconds": summary["seconds"]},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "analyze", "all"), default="all")
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG)
    out = ROOT / config.output
    if args.stage in ("prepare", "all"):
        prepare(config, out)
    if args.stage in ("analyze", "all"):
        analyze(config, out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
