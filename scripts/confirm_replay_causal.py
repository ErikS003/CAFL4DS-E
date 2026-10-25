"""Run and summarize the matched no-replay causal arm for the BDD replay study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.confirm_replay_advantage import _SEEDS, seed_readout
from scripts.run_adaptation_experiment import run_experiment

_ROOT = Path(__file__).resolve().parents[1]
_REPLAY_ROOT = _ROOT / "outputs/adaptation-bdd/replay-advantage-confirmation/20260927"
_DEFAULT_OUT = _ROOT / "outputs/adaptation-bdd/replay-causal-ablation/20260927"


def _absolute_panel_metrics(readout: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Recover adapted and frozen absolute scores for each predeclared panel."""
    metrics: dict[int, dict[str, float]] = {}
    for panel in readout["panels"]:
        global_scores = panel["absolute_global"]
        condition_rows = panel["condition_gains"]
        metrics[int(panel["panel_seed"])] = {
            "adapted_knn": float(global_scores["knn"]["adapted_acc"]),
            "frozen_knn": float(global_scores["knn"]["b5_acc"]),
            "adapted_linear": float(global_scores["linear"]["adapted_acc"]),
            "frozen_linear": float(global_scores["linear"]["b5_acc"]),
            "adapted_final_conditions": statistics.fmean(row["adapted_acc"] for row in condition_rows),
            "frozen_final_conditions": statistics.fmean(row["frozen_acc"] for row in condition_rows),
        }
    return metrics


def _seed_absolute(readout: dict[str, Any]) -> dict[str, float]:
    """Average absolute scores across panels within one training seed."""
    panels = _absolute_panel_metrics(readout)
    return {key: statistics.fmean(panel[key] for panel in panels.values()) for key in next(iter(panels.values()))}


def _load_readout(path: Path, expected_seed: int) -> tuple[dict[str, Any], dict[str, float]]:
    comparison = json.loads((path / "comparison.json").read_text(encoding="utf-8"))
    report = comparison["seeds"][0]
    if report["seed"] != expected_seed:
        raise ValueError(f"Unexpected seed in {path / 'comparison.json'}: {report['seed']}")
    readout = seed_readout(report)
    return readout, _seed_absolute(readout)


def _arm_metrics(replay: list[dict[str, float]], no_replay: list[dict[str, float]], arm: str) -> dict[str, Any]:
    """Build paired intervals for causal and frozen-reference effects."""
    keys = {"knn": "knn", "linear": "linear", "final_conditions": "final_conditions"}
    if arm == "replay_minus_no_replay":
        values = {
            metric: [
                100 * (r[f"adapted_{source}"] - n[f"adapted_{source}"]) for r, n in zip(replay, no_replay, strict=True)
            ]
            for metric, source in keys.items()
        }
    elif arm == "replay_minus_frozen":
        values = {
            metric: [100 * (r[f"adapted_{source}"] - r[f"frozen_{source}"]) for r in replay]
            for metric, source in keys.items()
        }
    elif arm == "no_replay_minus_frozen":
        values = {
            metric: [100 * (n[f"adapted_{source}"] - n[f"frozen_{source}"]) for n in no_replay]
            for metric, source in keys.items()
        }
    else:
        raise ValueError(arm)
    return {metric: _interval_from_pp(seed_values) for metric, seed_values in values.items()}


def _interval_from_pp(values: list[float]) -> dict[str, Any]:
    """Return the same five-seed paired interval as the replay confirmation."""
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("Causal confirmation requires exactly five finite seed values.")
    mean = statistics.fmean(values)
    half_width = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "n_seeds": 5,
        "mean_pp": mean,
        "ci95_pp": [mean - half_width, mean + half_width],
        "wins": sum(value > 0 for value in values),
        "seed_gains_pp": values,
        "interval": "paired Student-t across training seeds, conditional on fixed well and panels",
    }


def _verdict(metrics: dict[str, Any]) -> str:
    """Apply the broad replay-advantage decision rule to the causal comparison."""
    primary = [metrics["knn"], metrics["final_conditions"]]
    if all(metric["ci95_pp"][0] > 0 and metric["wins"] >= 4 for metric in primary):
        if metrics["knn"]["mean_pp"] >= 2:
            return "supported"
    if any(metric["ci95_pp"][1] <= 0 for metric in primary):
        return "contradicted_in_this_setting"
    return "not_established_inconclusive"


def validate_pair(replay_dir: Path, no_replay_dir: Path, seed: int) -> None:
    """Reject comparisons confounded by any setting other than the selection policy."""
    configs = [
        OmegaConf.to_container(OmegaConf.load(path / "config.yaml"), resolve=True)
        for path in (replay_dir, no_replay_dir)
    ]
    for config in configs:
        if not isinstance(config, dict) or config.get("seeds") != [seed]:
            raise ValueError("Expected exactly the paired training seed in each saved config.")
    replay, no_replay = configs
    if not isinstance(replay, dict) or not isinstance(no_replay, dict):
        raise ValueError("Expected configuration mappings.")
    if "ReservoirReplay" not in str(replay["filter"]):
        raise ValueError("Expected a replay selection policy.")
    if no_replay["filter"] != {"_target_": "cafl4ds.filters.accept_all.AcceptAll"}:
        raise ValueError("Expected AcceptAll for the no-replay arm.")
    for config in (replay, no_replay):
        for key in ("filter", "run_name", "seed_split"):
            config.pop(key, None)
    if replay != no_replay:
        differing = sorted(
            str(key) for key in replay.keys() | no_replay.keys() if replay.get(key) != no_replay.get(key)
        )
        raise ValueError(f"Unmatched configuration fields: {differing}")


def summarize(out_dir: Path, replay_dir: Path) -> dict[str, Any]:
    """Combine the completed no-replay arm with the existing replay confirmation."""
    rows = []
    replay_readouts = []
    no_replay_readouts = []
    replay_absolute = []
    no_replay_absolute = []
    frozen_panel_differences: list[float] = []
    for seed in _SEEDS:
        validate_pair(replay_dir / f"seed_{seed}", out_dir / f"seed_{seed}", seed)
        replay_readout, replay_scores = _load_readout(replay_dir / f"seed_{seed}", seed)
        no_readout, no_scores = _load_readout(out_dir / f"seed_{seed}", seed)
        rp, np = _absolute_panel_metrics(replay_readout), _absolute_panel_metrics(no_readout)
        if set(rp) != {0, 20260927, 20260928} or set(np) != set(rp):
            raise ValueError("Expected the same three evaluation panels in each arm.")
        for panel in rp:
            frozen_panel_differences.extend(
                abs(rp[panel][key] - np[panel][key]) for key in rp[panel] if key.startswith("frozen_")
            )
        replay_readouts.append(replay_readout)
        no_replay_readouts.append(no_readout)
        replay_absolute.append(replay_scores)
        no_replay_absolute.append(no_scores)
        rows.append(
            {
                "seed": seed,
                "replay": replay_scores,
                "no_replay": no_scores,
                "replay_minus_no_replay": {
                    key: replay_scores[f"adapted_{key}"] - no_scores[f"adapted_{key}"]
                    for key in ("knn", "linear", "final_conditions")
                },
            }
        )
    frozen_differences = [
        max(
            abs(replay_scores[key] - no_scores[key])
            for key in ("frozen_knn", "frozen_linear", "frozen_final_conditions")
        )
        for replay_scores, no_scores in zip(replay_absolute, no_replay_absolute, strict=True)
    ]
    metrics = {
        arm: _arm_metrics(replay_absolute, no_replay_absolute, arm)
        for arm in ("replay_minus_no_replay", "replay_minus_frozen", "no_replay_minus_frozen")
    }
    retention = {
        "replay_forgetting_pp": 100
        * statistics.fmean(item["retention_panel0"]["forgetting_measure"] for item in replay_readouts),
        "no_replay_forgetting_pp": 100
        * statistics.fmean(item["retention_panel0"]["forgetting_measure"] for item in no_replay_readouts),
        "replay_bwt_pp": 100
        * statistics.fmean(item["retention_panel0"]["backward_transfer"] for item in replay_readouts),
        "no_replay_bwt_pp": 100
        * statistics.fmean(item["retention_panel0"]["backward_transfer"] for item in no_replay_readouts),
    }
    result: dict[str, Any] = {
        "study": "P1.2 replay causal ablation, 2026-09-27",
        "causal_verdict": _verdict(metrics["replay_minus_no_replay"]),
        "metrics": metrics,
        "retention": retention,
        "frozen_reference_max_abs_difference_pp": 100 * max(frozen_differences),
        "frozen_panel_max_abs_difference_pp": 100 * max(frozen_panel_differences),
        "matched_configurations_verified": True,
        "retention_replay_minus_no_replay": {
            name: _interval_from_pp(
                [
                    100 * (r["retention_panel0"][key] - n["retention_panel0"][key])
                    for r, n in zip(replay_readouts, no_replay_readouts, strict=True)
                ]
            )
            for name, key in (("forgetting", "forgetting_measure"), ("bwt", "backward_transfer"))
        },
        "seeds": rows,
        "limitations": [
            "The replay/frozen arm is reused from the fixed five-seed confirmation under the same protocol.",
            "One shared pretrained well and one attribute-defined regime order.",
            "Three retained scene classes, not detection or all BDD tasks.",
            "Development-validation panels, not the untouched test split.",
        ],
    }
    (out_dir / "causal_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Replay causal ablation", "", f"Causal verdict: **{result['causal_verdict']}**", ""]
    for arm, arm_metrics in metrics.items():
        lines += [
            f"## {arm}",
            "",
            "| Metric | Mean gain (pp) | 95% interval (pp) | Positive seeds |",
            "| --- | ---: | ---: | ---: |",
        ]
        for metric, value in arm_metrics.items():
            low, high = value["ci95_pp"]
            lines.append(f"| {metric} | {value['mean_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {value['wins']}/5 |")
        lines.append("")
    lines += [
        (
            "Frozen-reference consistency check: max absolute difference = "
            f"{result['frozen_reference_max_abs_difference_pp']:.6f} pp."
        ),
        "",
        "## Retention",
        "",
        (
            f"- Replay forgetting: {retention['replay_forgetting_pp']:.2f} pp; "
            f"no-replay forgetting: {retention['no_replay_forgetting_pp']:.2f} pp."
        ),
        f"- Replay BWT: {retention['replay_bwt_pp']:.2f} pp; no-replay BWT: {retention['no_replay_bwt_pp']:.2f} pp.",
        "",
    ]
    limitations: list[str] = result["limitations"]
    lines += [f"- {item}" for item in limitations]
    (out_dir / "causal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _save_protocol(path: Path, provenance: dict[str, Any]) -> None:
    """Preserve the original timestamp and reject mismatched resumed configurations."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config", "replay_summary_sha256", "protocol_sha256"):
            if previous[key] != provenance[key]:
                raise ValueError(f"Existing run differs on {key}; use a new output directory.")
    else:
        path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the no-replay arm and combine it with the fixed replay confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--replay-dir", type=Path, default=_REPLAY_ROOT)
    parser.add_argument("--bdd-root", type=Path)
    parser.add_argument("--well", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    replay_dir = args.replay_dir.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(out_dir, replay_dir), indent=2))
        return
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "cafl4ds/configs")):
        config: DictConfig = compose(config_name="adaptation_bdd_no_replay_confirmation")
    if args.bdd_root:
        config.bdd_root = args.bdd_root.resolve().as_posix()
    if args.well:
        config.well = args.well.resolve().as_posix()
    well_path = Path(str(config.well))
    if not well_path.is_absolute():
        well_path = _ROOT / well_path
    config.well = well_path.as_posix()
    replay_protocol = json.loads((replay_dir / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(well_path.read_bytes()).hexdigest() != replay_protocol["checkpoint_sha256"]:
        raise ValueError("Warm checkpoint differs from the recorded replay experiment.")
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": OmegaConf.to_container(config, resolve=True),
        "replay_summary_sha256": hashlib.sha256((replay_dir / "summary.json").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (_ROOT / "docs/experiments/phase1/P1.2-replay-causal-ablation.md").read_bytes()
        ).hexdigest(),
        "git_commit": subprocess.check_output(  # noqa: S603 - fixed read-only Git command
            [shutil.which("git") or "git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
    }
    _save_protocol(out_dir / "protocol.json", provenance)
    protocol_path = _ROOT / "docs/experiments/phase1/P1.2-replay-causal-ablation.md"
    (out_dir / "protocol.md").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.add(out_dir / "execution.log", level="INFO")
    for seed in _SEEDS:
        seed_dir = out_dir / f"seed_{seed}"
        if (seed_dir / "comparison.json").exists():
            logger.info(f"seed {seed}: already completed")
            continue
        if seed_dir.exists() and any(seed_dir.iterdir()):
            raise RuntimeError(f"Partial seed output at {seed_dir}; preserve it and use a new output directory.")
        seed_config = copy.deepcopy(config)
        seed_config.seeds = [seed]
        seed_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(seed_config, seed_dir / "config.yaml", resolve=True)
        logger.info(f"starting no-replay causal seed {seed}")
        run_experiment(seed_config, seed_dir)
        logger.info(f"completed no-replay causal seed {seed}")
    result = summarize(out_dir, replay_dir)
    logger.info(f"causal replay verdict: {result['causal_verdict']}")


if __name__ == "__main__":
    main()
