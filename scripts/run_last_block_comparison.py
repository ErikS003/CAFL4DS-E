"""Run a matched last-block MAE development comparison against full learning and frozen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.confirm_replay_advantage import _SEEDS, paired_interval, verdict
from scripts.confirm_replay_causal import _absolute_panel_metrics, _load_readout
from scripts.run_adaptation_experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "outputs/adaptation-bdd/replay-causal-ablation/20260927"
OUTPUT = ROOT / "outputs/adaptation-bdd/last-block-comparison/20260927"
PROTOCOL = ROOT / "docs/experiments/phase1/P1.2-last-block-comparison.md"
METRICS = ("knn", "linear", "final_conditions")


def comparable(config: DictConfig, seed: int) -> dict[str, Any]:
    """Normalize pre-run seed snapshots without hiding scientific settings."""
    result = dict(cast(dict[str, Any], OmegaConf.to_container(config, resolve=True)))
    result["seed"] = seed
    result["stream"]["seed"] = seed
    for key in ("train_scope", "run_name", "seed_split"):
        result.pop(key, None)
    return result


def check_config(config: DictConfig, control_dir: Path, seed: int) -> None:
    """Permit only the registered scope and metadata differences from the completed control."""
    baseline = cast(DictConfig, OmegaConf.load(control_dir / f"seed_{seed}/config.yaml"))
    if config.train_scope != "last_block" or baseline.train_scope != "full":
        raise ValueError("Expected last_block versus full scopes.")
    if comparable(config, seed) != comparable(baseline, seed):
        raise ValueError(f"Training/evaluation config mismatch for seed {seed}.")


def record(config: DictConfig, control_dir: Path, out_dir: Path) -> None:
    """Bind the study to the existing checkpoint, runner, and five paired controls."""
    replay = json.loads(
        (ROOT / "outputs/adaptation-bdd/replay-advantage-confirmation/20260927/protocol.json").read_text()
    )
    well_hash = hashlib.sha256(Path(str(config.well)).read_bytes()).hexdigest()
    runner_hash = hashlib.sha256((ROOT / "scripts/run_adaptation_experiment.py").read_bytes()).hexdigest()
    if (
        well_hash != replay["checkpoint_sha256"]
        or runner_hash != replay["source_sha256"]["run_adaptation_experiment.py"]
    ):
        raise ValueError("Checkpoint or training runner changed since the controls.")
    payload = {
        "config": OmegaConf.to_container(config, resolve=True),
        "checkpoint_sha256": well_hash,
        "runner_sha256": runner_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "controls_sha256": {
            str(seed): hashlib.sha256((control_dir / f"seed_{seed}/comparison.json").read_bytes()).hexdigest()
            for seed in _SEEDS
        },
    }
    path = out_dir / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["payload"] != payload:
            raise ValueError("Recorded study differs; use a new output directory.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "payload": payload}, indent=2),
            encoding="utf-8",
        )
        (out_dir / "protocol.md").write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")


def summarize(out_dir: Path, control_dir: Path) -> dict[str, Any]:
    """Average panels within seed and report accuracy and paired retention differences."""
    rows: list[dict[str, Any]] = []
    frozen_difference = 0.0
    for seed in _SEEDS:
        directory = out_dir / f"seed_{seed}"
        check_config(cast(DictConfig, OmegaConf.load(directory / "config.yaml")), control_dir, seed)
        last, last_abs = _load_readout(directory, seed)
        full, full_abs = _load_readout(control_dir / f"seed_{seed}", seed)
        lp, fp = _absolute_panel_metrics(last), _absolute_panel_metrics(full)
        if set(lp) != {0, 20260927, 20260928} or set(fp) != set(lp):
            raise ValueError("Missing evaluation panels.")
        frozen_difference = max(
            frozen_difference, *(abs(lp[p][k] - fp[p][k]) for p in lp for k in lp[p] if k.startswith("frozen_"))
        )
        rows.append(
            {
                "seed": seed,
                "last": last,
                "full": full,
                "last_absolute": last_abs,
                "full_absolute": full_abs,
                "last_panels": lp,
                "full_panels": fp,
            }
        )
    if frozen_difference > 1e-10:
        raise ValueError("Frozen panel scores differ from the matched control.")
    metrics = {
        name: {
            key: paired_interval([row[a][f"adapted_{key}"] - row[b][f"{target}_{key}"] for row in rows])
            for key in METRICS
        }
        for name, a, b, target in (
            ("last_minus_full", "last_absolute", "full_absolute", "adapted"),
            ("last_minus_frozen", "last_absolute", "last_absolute", "frozen"),
            ("full_minus_frozen", "full_absolute", "full_absolute", "frozen"),
        )
    }
    retention = {
        key: paired_interval(
            [row["last"]["retention_panel0"][key] - row["full"]["retention_panel0"][key] for row in rows]
        )
        for key in ("forgetting_measure", "backward_transfer")
    }
    result = {
        "study": "Last-block development comparison, reused seeds and validation panels",
        "metrics": metrics,
        "retention_last_minus_full": retention,
        "retention_means_pp": {
            arm: {key: 100 * statistics.fmean(row[arm]["retention_panel0"][key] for row in rows) for key in retention}
            for arm in ("last", "full")
        },
        "frozen_advantage_verdict": verdict(metrics["last_minus_frozen"]),
        "broader_accuracy_superiority": verdict(metrics["last_minus_frozen"]) == "supported"
        and metrics["last_minus_frozen"]["linear"]["ci95_pp"][0] > 0,
        "retention_accuracy_candidate": retention["forgetting_measure"]["ci95_pp"][1] < 0
        and all(m["ci95_pp"][0] > -1 for m in metrics["last_minus_full"].values()),
        "frozen_panel_max_difference_pp": 100 * frozen_difference,
        "per_panel": {
            str(p): {
                contrast: {
                    key: paired_interval(
                        [row["last_panels"][p][f"adapted_{key}"] - row[other][p][f"{target}_{key}"] for row in rows]
                    )
                    for key in METRICS
                }
                for contrast, other, target in (
                    ("last_minus_full", "full_panels", "adapted"),
                    ("last_minus_frozen", "last_panels", "frozen"),
                )
            }
            for p in (0, 20260927, 20260928)
        },
        "seeds": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Last-block development comparison",
        "",
        f"Frozen advantage: {result['frozen_advantage_verdict']}",
        f"Retention/accuracy candidate: {result['retention_accuracy_candidate']}",
        "",
        "Intervals describe five reused seeds, conditional on one warm checkpoint and three validation panels.",
        "",
    ]
    for name, values in {**metrics, "retention_last_minus_full": retention}.items():
        lines += [f"## {name}", "", "| Metric | Mean difference (pp) | 95% interval |", "| --- | ---: | ---: |"]
        for key, value in values.items():
            low, high = value["ci95_pp"]
            lines.append(f"| {key} | {value['mean_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
        lines.append("")
    lines += [
        "Negative forgetting differences favor last-block; positive BWT differences favor last-block.",
        "Retention uses panel 0; sparse conditions make its macro-average noisy.",
        "These are development results; any selected candidate requires fresh-seed confirmation.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run five fixed last-block seeds, reusing the existing full/frozen controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out_dir, control = args.out.resolve(), args.control.resolve()
    if not args.summarize_only:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "cafl4ds/configs")):
            config = compose(config_name="adaptation_bdd_last_block_comparison")
        config.well = (ROOT / str(config.well)).resolve().as_posix()
        for seed in _SEEDS:
            one = copy.deepcopy(config)
            one.seeds = [seed]
            check_config(one, control, seed)
        record(config, control, out_dir)
        logger.add(out_dir / "execution.log", level="INFO")
        for seed in _SEEDS:
            directory = out_dir / f"seed_{seed}"
            if (directory / "comparison.json").exists():
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Partial seed at {directory}; preserve it and use a new output directory.")
            one = copy.deepcopy(config)
            one.seeds = [seed]
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(one, directory / "config.yaml", resolve=True)
            logger.info(f"starting last-block seed {seed}")
            run_experiment(one, directory)
            logger.info(f"completed last-block seed {seed}")
    result = summarize(out_dir, control)
    logger.info(
        f"Frozen advantage: {result['frozen_advantage_verdict']}; "
        f"retention candidate: {result['retention_accuracy_candidate']}"
    )


if __name__ == "__main__":
    main()
