"""Run the matched warm-start scope-by-replay extension using completed no-replay controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from scripts.confirm_last_block import (
    METRICS,
    PANELS,
    ROOT,
    SEEDS,
    audit_weights,
    check_frozen_pair,
    lock_json,
    preflight,
    read_arm,
    write_json,
)
from scripts.confirm_replay_advantage import paired_interval, verdict
from scripts.run_adaptation_experiment import run_experiment

CONTROL = ROOT / "outputs/adaptation-bdd/last-block-confirmation/20260928"
OUTPUT = ROOT / "outputs/adaptation-bdd/scope-replay/20260929"
PROTOCOL = ROOT / "docs/experiments/phase1/P1.2-scope-replay.md"
SCOPES = ("full", "last_block")
ARMS = tuple(f"{scope}_{policy}" for scope in SCOPES for policy in ("no_replay", "replay"))
CONTRASTS = {
    **{f"{s}_replay_minus_no_replay": (f"{s}_replay", f"{s}_no_replay", "adapted") for s in SCOPES},
    "last_minus_full_with_replay": ("last_block_replay", "full_replay", "adapted"),
    "last_minus_full_without_replay": ("last_block_no_replay", "full_no_replay", "adapted"),
    **{f"{a}_minus_frozen": (a, a, "frozen") for a in ARMS},
}


def digest(path: Path) -> str:
    """Hash an immutable source or small checkpoint artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_payload(path: Path) -> dict[str, Any]:
    """Read an existing locked provenance payload."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))["payload"])


def one_config(config: DictConfig, scope: str, seed: int) -> DictConfig:
    """Resolve the paired seed consistently for stream ordering and the private reservoir RNG."""
    one = copy.deepcopy(config)
    one.seed = seed
    one.stream.seed = seed
    one.seeds = [seed]
    one.train_scope = scope
    return one


def validate_config(replay: DictConfig, control: DictConfig, seed: int, scope: str) -> None:
    """Permit only the fixed replay policy and descriptive metadata to differ within a scope."""
    configs = [cast(dict[str, Any], OmegaConf.to_container(c, resolve=True)) for c in (replay, control)]
    expected = {
        "_target_": "cafl4ds.filters.composite.CompositeSelector",
        "admission": [],
        "buffer": {
            "_target_": "cafl4ds.filters.reservoir.ReservoirReplay",
            "capacity": 256,
            "replay_batch": 16,
            "max_train_batch": 32,
            "seed": seed,
        },
    }
    if configs[0]["filter"] != expected or configs[1]["filter"] != {"_target_": "cafl4ds.filters.accept_all.AcceptAll"}:
        raise ValueError("Unexpected selection/replay policy.")
    for c in configs:
        if c["seeds"] != [seed] or c["seed"] != seed or c["stream"]["seed"] != seed or c["train_scope"] != scope:
            raise ValueError("Scope or seed mismatch.")
        for key in ("filter", "run_name", "seed_split"):
            c.pop(key, None)
    if configs[0] != configs[1]:
        raise ValueError("Replay/control configuration mismatch.")


def verify_controls(control: Path) -> dict[str, Any]:
    """Verify the original implementation and snapshot every reused scientific artifact."""
    recorded = load_payload(control / "protocol.json")
    if torch.__version__ != recorded["torch_version"]:
        raise ValueError("PyTorch changed since the controls.")
    for name, expected in recorded["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Control implementation changed: {name}")
    if digest(Path(recorded["config"]["well"])) != recorded["checkpoint_sha256"]:
        raise ValueError("Shared warm checkpoint changed.")
    files = [
        control / name for name in ("protocol.json", "evaluation_protocol.json", "summary.json", "weight_audit.json")
    ]
    for scope in SCOPES:
        for seed in SEEDS:
            directory = control / scope / f"seed_{seed}"
            files.extend(
                directory / name for name in ("config.yaml", "comparison.json", f"checkpoints/adapted_s{seed}.pt")
            )
    return {
        "control_files_sha256": {p.relative_to(control).as_posix(): digest(p) for p in files},
        "checkpoint_sha256": recorded["checkpoint_sha256"],
    }


def check_locked_files(out: Path, control: Path) -> dict[str, Any]:
    """Reject source, control or checkpoint changes before reporting a matched result."""
    saved = load_payload(out / "protocol.json")
    for name, expected in saved["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Recorded study source changed: {name}")
    for name, expected in saved["control_files_sha256"].items():
        if digest(control / name) != expected:
            raise ValueError(f"Reused control changed: {name}")
    if (
        digest(Path(saved["config"]["well"])) != saved["checkpoint_sha256"]
        or digest(PROTOCOL) != saved["protocol_sha256"]
    ):
        raise ValueError("Checkpoint or protocol changed.")
    if load_payload(out / "evaluation_protocol.json") != load_payload(control / "evaluation_protocol.json"):
        raise ValueError("Evaluation or training data/order differ from the controls.")
    return saved


def collect_rows(out: Path, control: Path, eligible: list[int] | None) -> list[dict[str, Any]]:
    """Validate and read each matched four-arm seed, including individual frozen condition scores."""
    rows = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for scope in SCOPES:
            replay_dir, control_dir = out / scope / f"seed_{seed}", control / scope / f"seed_{seed}"
            validate_config(
                cast(DictConfig, OmegaConf.load(replay_dir / "config.yaml")),
                cast(DictConfig, OmegaConf.load(control_dir / "config.yaml")),
                seed,
                scope,
            )
            row[f"{scope}_replay"] = read_arm(replay_dir, seed, eligible)
            row[f"{scope}_no_replay"] = read_arm(control_dir, seed, eligible)
        for arm in ARMS:
            check_frozen_pair({"full": row["full_no_replay"], "last_block": row[arm]})
        rows.append(row)
    return rows


def interaction(row: dict[str, Any], section: str, key: str) -> float:
    """Return replay's effect in last-block minus its effect in full-backbone learning."""
    return float(
        (row["last_block_replay"][section][key] - row["last_block_no_replay"][section][key])
        - (row["full_replay"][section][key] - row["full_no_replay"][section][key])
    )


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all declared contrasts using the seed as the replication unit."""
    accuracy = {
        name: {
            key: paired_interval(
                [r[a]["absolute"][f"adapted_{key}"] - r[b]["absolute"][f"{target}_{key}"] for r in rows]
            )
            for key in METRICS
        }
        for name, (a, b, target) in CONTRASTS.items()
    }
    retention = {
        name: {
            key: paired_interval([r[a]["retention"][key] - r[b]["retention"][key] for r in rows])
            for key in ("forgetting", "bwt")
        }
        for name, (a, b, target) in CONTRASTS.items()
        if target == "adapted"
    }
    return {
        "accuracy": accuracy,
        "retention": retention,
        "absolute_means_percent_or_pp": {
            a: {
                **{key: 100 * statistics.fmean(r[a]["absolute"][key] for r in rows) for key in rows[0][a]["absolute"]},
                **{key: 100 * statistics.fmean(r[a]["retention"][key] for r in rows) for key in ("forgetting", "bwt")},
            }
            for a in ARMS
        },
        "replay_retention_accuracy_candidate": {
            scope: retention[f"{scope}_replay_minus_no_replay"]["forgetting"]["ci95_pp"][1] < 0
            and all(m["ci95_pp"][0] > -1 for m in accuracy[f"{scope}_replay_minus_no_replay"].values())
            for scope in SCOPES
        },
        "frozen_advantage_verdict": {a: verdict(accuracy[f"{a}_minus_frozen"]) for a in ARMS},
        "broader_accuracy_superiority": {
            a: verdict(accuracy[f"{a}_minus_frozen"]) == "supported"
            and accuracy[f"{a}_minus_frozen"]["linear"]["ci95_pp"][0] > 0
            for a in ARMS
        },
        "interaction_replay_effect_last_minus_full": {
            **{key: paired_interval([interaction(r, "absolute", f"adapted_{key}") for r in rows]) for key in METRICS},
            **{key: paired_interval([interaction(r, "retention", key) for r in rows]) for key in ("forgetting", "bwt")},
        },
        "per_panel": {
            p: {
                name: {
                    key: paired_interval(
                        [r[a]["panels"][p][f"adapted_{key}"] - r[b]["panels"][p][f"{target}_{key}"] for r in rows]
                    )
                    for key in METRICS
                }
                for name, (a, b, target) in CONTRASTS.items()
            }
            for p in map(str, PANELS)
        },
        "seeds": rows,
    }


def write_report(out: Path, result: dict[str, Any]) -> None:
    """Write readable effect sizes and the fixed primary/secondary decision labels."""
    lines = [
        "# Matched scope-by-replay study",
        "",
        "Five reused training seeds; three panels averaged within seed; retention uses panel 31001.",
        "Primary intervention: last-block replay. Full-backbone replay and interaction are secondary/exploratory.",
        "Nominal intervals are conditional on one checkpoint and reused validation panels, not simultaneous bounds.",
        "",
    ]
    for name in ("primary", "all_conditions_secondary"):
        report = result[name]
        lines += [
            f"## {name}",
            "",
            f"Replay retention/accuracy candidates: {report['replay_retention_accuracy_candidate']}",
            f"Frozen-advantage verdicts: {report['frozen_advantage_verdict']}",
            "",
        ]
        for section in ("accuracy", "retention"):
            for contrast, values in report[section].items():
                lines += [
                    f"### {section}: {contrast}",
                    "",
                    "| Metric | Mean difference (pp) | 95% interval |",
                    "| --- | ---: | ---: |",
                ]
                for metric, value in values.items():
                    low, high = value["ci95_pp"]
                    lines.append(f"| {metric} | {value['mean_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
                lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_report(out: Path, result: dict[str, Any]) -> None:
    """Plot direct replay effects separately by scope, not cross-study raw-score differences."""
    report = result["primary"]
    fig = Figure(figsize=(14, 4))
    axes = fig.subplots(1, 4)
    for ax, metric in zip(axes, (*METRICS, "forgetting"), strict=True):
        section = "retention" if metric == "forgetting" else "accuracy"
        values = [report[section][f"{s}_replay_minus_no_replay"][metric] for s in SCOPES]
        means = [v["mean_pp"] for v in values]
        ax.errorbar(
            range(2),
            means,
            yerr=[[v["mean_pp"] - v["ci95_pp"][0] for v in values], [v["ci95_pp"][1] - v["mean_pp"] for v in values]],
            fmt="o",
            capsize=5,
        )
        ax.axhline(0, color="0.4", linewidth=1)
        ax.set_xticks(range(2), ["Full encoder", "Last block"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(metric.replace("_", " ") + (" (lower better)" if metric == "forgetting" else ""))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Replay minus no replay (pp), mean and 95% interval")
    fig.suptitle("Warm-start BDD100K: matched replay effects across five training seeds")
    fig.tight_layout()
    fig.savefig(out / "replay_effects.png", dpi=160)


def summarize(out: Path, control: Path) -> dict[str, Any]:
    """Audit all twenty checkpoints and reduce the complete matched four-arm study."""
    locked = check_locked_files(out, control)
    evaluation = load_payload(out / "evaluation_protocol.json")
    result = {
        "study": "P1.2 matched scope-by-replay development study, 2026-09-29",
        "primary": reduce_rows(collect_rows(out, control, evaluation["eligible_conditions"])),
        "all_conditions_secondary": reduce_rows(collect_rows(out, control, None)),
        "eligible_conditions": evaluation["eligible_conditions"],
        "controls_unchanged": True,
    }
    well = Path(locked["config"]["well"])
    write_json(
        out / "weight_audit.json",
        {"new_replay": audit_weights(out, well), "reused_no_replay": audit_weights(control, well)},
    )
    write_json(out / "summary.json", result)
    write_report(out, result)
    plot_report(out, result)
    return result


def main() -> None:
    """Run only the ten missing replay arms, preserving completed matched controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out, control = args.out.resolve(), args.control.resolve()
    if args.summarize_only:
        summarize(out, control)
        return
    controls = verify_controls(control)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "cafl4ds/configs")):
        config = compose(config_name="adaptation_bdd_scope_replay")
    config.well = (ROOT / str(config.well)).resolve().as_posix()
    for seed in SEEDS:
        for scope in SCOPES:
            validate_config(
                one_config(config, scope, seed),
                cast(DictConfig, OmegaConf.load(control / scope / f"seed_{seed}/config.yaml")),
                seed,
                scope,
            )
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "cafl4ds").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    lock_json(
        out / "protocol.json",
        {
            "config": OmegaConf.to_container(config, resolve=True),
            "control_root": str(control),
            **controls,
            "protocol_sha256": digest(PROTOCOL),
            "torch_version": torch.__version__,
            "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        },
    )
    if not (out / "protocol.md").exists():
        (out / "protocol.md").write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    logger.add(out / "execution.log", level="INFO")
    logger.info("Matched configs and control provenance verified; loading data for fingerprint checks")
    preflight(config, out)
    check_locked_files(out, control)
    for index, seed in enumerate(SEEDS):
        for scope in SCOPES if index % 2 == 0 else tuple(reversed(SCOPES)):
            directory = out / scope / f"seed_{seed}"
            one = one_config(config, scope, seed)
            if (directory / "comparison.json").exists():
                if OmegaConf.to_container(
                    OmegaConf.load(directory / "config.yaml"), resolve=True
                ) != OmegaConf.to_container(one, resolve=True):
                    raise ValueError("Completed replay arm config mismatch.")
                logger.info(f"Skipping completed replay {scope} seed {seed}")
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Partial arm at {directory}; preserve before explicitly restarting it.")
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(one, directory / "config.yaml", resolve=True)
            logger.info(f"Starting replay {scope} seed {seed}")
            run_experiment(one, directory)
            logger.info(f"Completed replay {scope} seed {seed}")
    result = summarize(out, control)
    logger.info(f"All ten replay arms complete. Candidates: {result['primary']['replay_retention_accuracy_candidate']}")


if __name__ == "__main__":
    main()
