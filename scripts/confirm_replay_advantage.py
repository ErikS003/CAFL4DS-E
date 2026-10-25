"""Run the fixed five-seed BDD replay confirmation and summarize paired uncertainty."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.run_adaptation_experiment import run_experiment

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / "docs/experiments/phase1/P1.2-replay-confirmation.md"
_SEEDS = [2003, 2011, 2017, 2027, 2029]
_T_CRITICAL_DF4 = 2.7764451051977987


def paired_interval(values: list[float]) -> dict[str, Any]:
    """Summarize five independent training-seed differences in percentage points."""
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("The fixed confirmation requires exactly five finite paired seed differences.")
    mean = statistics.fmean(values)
    half_width = _T_CRITICAL_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "n_seeds": len(values),
        "mean_pp": 100 * mean,
        "ci95_pp": [100 * (mean - half_width), 100 * (mean + half_width)],
        "wins": sum(value > 0 for value in values),
        "seed_gains_pp": [100 * value for value in values],
        "interval": "paired Student-t across training seeds, conditional on fixed well and panels",
    }


def verdict(metrics: dict[str, Any]) -> str:
    """Apply the recorded joint hypothesis rule, retaining inconclusive as a possible outcome."""
    primary = [metrics["knn"], metrics["final_conditions"]]
    if all(metric["ci95_pp"][0] > 0 and metric["wins"] >= 4 for metric in primary):
        if metrics["knn"]["mean_pp"] >= 2:
            return "supported"
    if any(metric["ci95_pp"][1] <= 0 for metric in primary):
        return "contradicted_in_this_setting"
    return "not_established_inconclusive"


def seed_readout(report: dict[str, Any]) -> dict[str, Any]:
    """Average repeated probe panels within one seed, avoiding panel pseudoreplication."""
    panels = [
        {
            "panel_seed": 0,
            "global": report["global"],
            "final_per_condition": report["conditions"]["final_per_condition"],
        },
        *report["probe_panels"],
    ]
    panel_results = []
    for panel in panels:
        condition_gains = [row["gain"] for row in panel["final_per_condition"]]
        panel_results.append(
            {
                "panel_seed": panel["panel_seed"],
                "knn": panel["global"]["knn"]["gain"],
                "linear": panel["global"]["linear"]["gain"],
                "final_conditions": statistics.fmean(condition_gains),
                "absolute_global": panel["global"],
                "condition_gains": panel["final_per_condition"],
            }
        )
    return {
        "seed": report["seed"],
        "panel_means": {
            key: statistics.fmean(panel[key] for panel in panel_results)
            for key in ("knn", "linear", "final_conditions")
        },
        "panels": panel_results,
        "evaluation_sizes": report["evaluation_sizes"],
        "retention_panel0": report["conditions"]["adapted_summary"],
    }


def summarize(out_dir: Path) -> dict[str, Any]:
    """Reduce all five completed runs and write the decision plus individual results."""
    rows = []
    for seed in _SEEDS:
        comparison = json.loads((out_dir / f"seed_{seed}/comparison.json").read_text(encoding="utf-8"))
        report = comparison["seeds"][0]
        if report["seed"] != seed:
            raise ValueError(f"Unexpected seed in saved report for {seed}.")
        rows.append(seed_readout(report))
    metrics = {
        key: paired_interval([row["panel_means"][key] for row in rows]) for key in ("knn", "linear", "final_conditions")
    }
    result = {
        "study": "P1.2 replay advantage confirmation, 2026-09-27",
        "verdict": verdict(metrics),
        "broader_accuracy_superiority": verdict(metrics) == "supported" and metrics["linear"]["ci95_pp"][0] > 0,
        "metrics": metrics,
        "per_panel": {
            str(panel_seed): {
                key: paired_interval(
                    [next(panel[key] for panel in row["panels"] if panel["panel_seed"] == panel_seed) for row in rows]
                )
                for key in ("knn", "linear", "final_conditions")
            }
            for panel_seed in (0, 20260927, 20260928)
        },
        "mean_forgetting_pp": 100 * statistics.fmean(row["retention_panel0"]["forgetting_measure"] for row in rows),
        "mean_bwt_pp": 100 * statistics.fmean(row["retention_panel0"]["backward_transfer"] for row in rows),
        "seeds": rows,
        "limitations": [
            "One shared pretrained well and one attribute-defined regime order.",
            "Scene classification on three retained classes, not detection or all BDD tasks.",
            "Expanded development-validation evaluation, not an untouched benchmark test.",
            "Panel overlaps and shared queries are not counted as independent replication.",
            "Intervals quantify training-seed variation, not independent dataset or pretraining variation.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Replay advantage confirmation", "", f"Verdict: **{result['verdict']}**", ""]
    lines += ["| Metric | Mean gain (pp) | 95% interval (pp) | Positive seeds |", "| --- | ---: | ---: | ---: |"]
    for key, metric in metrics.items():
        low, high = metric["ci95_pp"]
        lines.append(f"| {key} | {metric['mean_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {metric['wins']}/5 |")
    lines += ["", "Panels are averaged within each training seed; n=5, not n=15.", ""]
    limitations: list[str] = result["limitations"]  # type: ignore[assignment]
    lines += [f"- {item}" for item in limitations]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def record_protocol(config: DictConfig, well_path: Path, out_dir: Path) -> None:
    """Write immutable configuration provenance before any training starts."""
    resolved = OmegaConf.to_container(config, resolve=True)
    source_files = [Path(__file__), _ROOT / "scripts/run_adaptation_experiment.py"]
    fingerprint_input = {
        "config": resolved,
        "checkpoint_sha256": hashlib.sha256(well_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(_PROTOCOL.read_bytes()).hexdigest(),
        "source_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in source_files},
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Existing run fingerprint differs; use a new output directory.")
    else:
        provenance = {
            "fingerprint": fingerprint,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(  # noqa: S603 - fixed read-only Git command
                [shutil.which("git") or "git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
            ).strip(),
            **fingerprint_input,
        }
        protocol_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        (out_dir / "protocol.md").write_text(_PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    """Freeze provenance, run all fixed seeds, and resume only matching completed seed artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=_ROOT / "outputs/adaptation-bdd/replay-advantage-confirmation/20260927"
    )
    parser.add_argument("--bdd-root", type=Path)
    parser.add_argument("--well", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(out_dir)["metrics"], indent=2))
        return
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds/configs")):
        config = compose(config_name="adaptation_bdd_replay_confirmation")
    if args.bdd_root:
        config.bdd_root = args.bdd_root.resolve().as_posix()
    if args.well:
        config.well = args.well.resolve().as_posix()
    well_path = Path(str(config.well))
    if not well_path.is_absolute():
        well_path = _ROOT / well_path
    config.well = well_path.as_posix()
    if list(config.seeds) != _SEEDS:
        raise ValueError("The registered training seeds must not change.")
    record_protocol(config, well_path, out_dir)
    logger.add(out_dir / "execution.log", level="INFO")
    for seed in _SEEDS:
        seed_dir = out_dir / f"seed_{seed}"
        if (seed_dir / "comparison.json").exists():
            logger.info(f"seed {seed}: already completed under matching fingerprint")
            continue
        if seed_dir.exists() and any(seed_dir.iterdir()):
            raise RuntimeError(f"Partial seed output at {seed_dir}; preserve it and use a new directory for a rerun.")
        seed_config = copy.deepcopy(config)
        seed_config.seeds = [seed]
        seed_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(seed_config, seed_dir / "config.yaml", resolve=True)
        start = time.monotonic()
        logger.info(f"starting fixed confirmation seed {seed}")
        run_experiment(seed_config, seed_dir)
        logger.info(f"completed seed {seed} in {time.monotonic() - start:.1f} seconds")
    result = summarize(out_dir)
    logger.info(f"confirmation verdict: {result['verdict']}; metrics={result['metrics']}")


if __name__ == "__main__":
    main()
