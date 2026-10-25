"""Freeze and run the cached warning diagnostic without a learner runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.dnd_health_transfer import MONITOR, POINTS
from cafl4ds.dnd_signal_diagnostic import features_and_alarms, horizon_steps, outcome_targets

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cafl4ds/configs/dnd_signal_diagnostic.yaml"


def read(path: Path) -> dict[str, Any]:
    """Read a saved record."""
    return json.loads(path.read_text(encoding="utf-8"))


def plain(value: object) -> object:
    """Convert scientific arrays for strict JSON."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path: Path, value: object) -> None:
    """Write a new record; completed scientific files cannot be overwritten."""
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    """Hash logical bytes."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def check(hashes: dict[str, str]) -> None:
    """Require unchanged inputs and completed artifacts."""
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed artifact: {name}")


def prepare(config: DictConfig, out: Path) -> None:
    """Bind source science and analysis before computing new diagnostic outcomes."""
    if out.exists() or (ROOT / config.notebook).exists():
        raise FileExistsError("Use unused analysis and notebook paths")
    source = ROOT / config.source
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "cafl4ds/dnd_signal_diagnostic.py",
        ROOT / "scripts/verify_dnd_signal_diagnostic.py",
        ROOT / "scripts/report_dnd_signal_diagnostic.py",
        ROOT / "tests/unit/test_dnd_signal_diagnostic.py",
        ROOT / "docs/project-plan/dnd-signal-diagnostic-frozen.md",
        ROOT / "cafl4ds/dnd_health_stopping.py",
        ROOT / "cafl4ds/dnd_health_transfer.py",
        ROOT / "uv.lock",
    ]
    for name in (
        "input_lock.json",
        "complete.json",
        "independent-verification.json",
        "training-replay-verification.json",
        "delivery-verification.json",
    ):
        path = source / name
        record = read(path)
        if name not in ("input_lock.json", "complete.json") and not record["passed"]:
            raise ValueError(path)
        if "hashes" in record:
            check(record["hashes"])
        paths.append(path)
    paths += [source / "collection.json", ROOT / config.evaluation / "data.npz"]
    for path in sorted(source.glob("transfer_*/complete.json")):
        record = read(path)
        check(record["hashes"])
        paths += [path, *[Path(p) for p in record["hashes"]]]
    paths += sorted((source / "warm").glob("panel_*.*"))
    hashes = {str(path): digest(path) for path in sorted(set(paths))}
    out.mkdir(parents=True)
    for path in paths:
        if path.suffix in (".py", ".yaml", ".md") and path.is_relative_to(ROOT) and path.parent != source:
            target = out / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write(
        out / "input_lock.json",
        {
            "hashes": hashes,
            "config": OmegaConf.to_container(config, resolve=True),
            "frozen_unix": time.time(),
            "exploratory_reuse": True,
            "frozen_before_new_diagnostic_targets": True,
        },
    )
    print({"prepared": True, "frozen_files": len(hashes)}, flush=True)


def analyze(config: DictConfig, out: Path) -> None:
    """Derive causal observations, matched-horizon outcomes and held-history forecasts."""
    lock = read(out / "input_lock.json")
    check(lock["hashes"])
    if (out / "derived_windows.npz").exists():
        raise FileExistsError("Preserve the completed or interrupted diagnostic")
    records = read(ROOT / config.source / "collection.json")["records"]
    features, alarms, panels = [], [], []
    for record in records:
        with np.load(ROOT / config.source / record["name"] / "embeddings.npz") as a:
            banks = {point: a[str(point)] for point in MONITOR}
        f, alarm = features_and_alarms(banks, MONITOR)
        features.append(f)
        alarms.append(alarm)
        panels.append([record["observations"][str(point)]["panel_accuracy_pp"] for point in POINTS])
    features, alarms, panels = np.array(features), np.array(alarms), np.array(panels)
    means = panels.mean(2)
    windows, saved = (
        {},
        {
            "features_all": features,
            "alarms_all": alarms,
            "panel_curve": panels,
            "mean_curve": means,
            "monitor": np.array(MONITOR),
            "points": np.array(POINTS),
        },
    )
    actual = []
    for horizon in config.horizons:
        steps = horizon_steps(MONITOR, POINTS, horizon)
        ids = [MONITOR.index(int(p)) for p in steps]
        targets = np.array([outcome_targets(curve, POINTS, steps, horizon) for curve in means])
        support = np.array(
            [
                np.stack([outcome_targets(curve[:, draw], POINTS, steps, horizon) for draw in range(3)], axis=1)
                for curve in panels
            ]
        )
        data = {
            "features": features[:, ids],
            "alarms": alarms[:, ids],
            "steps": steps,
            "targets": targets,
            "support_targets": support,
        }
        windows[int(horizon)] = data
        saved.update({f"{key}_{horizon}": value for key, value in data.items()})
        for i, record in enumerate(records):
            stop = record["stop"]
            available = stop in steps
            actual.append(
                {
                    "history": record["name"],
                    "horizon": int(horizon),
                    "stop": stop,
                    "full_horizon_available": available,
                    "targets_pp": targets[i, list(steps).index(stop)] if available else None,
                    "phase": "day1"
                    if stop <= record["day_updates"]
                    else "night"
                    if stop <= record["day_updates"] + record["night_updates"]
                    else "day2",
                }
            )
    np.savez_compressed(out / "derived_windows.npz", **saved)
    study = instantiate(config.study)
    summary, predictions, folds = study.analyze(windows, config.primary_horizon)
    summary["actual_first_alarm_outcomes"] = actual
    summary["history_names"] = [r["name"] for r in records]
    np.savez_compressed(out / "predictions.npz", **predictions)
    write(out / "folds.json", {"fits": folds, "statistical_model_fits": len(folds)})
    write(out / "summary.json", summary)
    check(lock["hashes"])
    files = [out / name for name in ("derived_windows.npz", "predictions.npz", "folds.json", "summary.json")]
    write(
        out / "complete.json",
        {
            "complete": True,
            "hashes": {str(path): digest(path) for path in files},
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
            "statistical_model_fits": len(folds),
        },
    )
    print(
        {
            "cached_diagnostic": "complete",
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
            "statistical_model_fits": len(folds),
            "primary_candidates": summary["primary_candidates"],
        },
        flush=True,
    )


def main() -> None:
    """Dispatch explicit preparation and analysis stages."""
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
