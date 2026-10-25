"""Run the locked fresh-seed last-block/full/frozen comparison with count-qualified evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cafl4ds.eval import backward_transfer, forgetting_measure
from scripts.confirm_replay_advantage import paired_interval, verdict
from scripts.run_adaptation_experiment import _shared_source, run_experiment

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/adaptation-bdd/last-block-confirmation/20260928"
PROTOCOL = ROOT / "docs/experiments/phase1/P1.2-last-block-confirmation.md"
SEEDS = [3001, 3011, 3019, 3023, 3037]
PANELS = [31001, 31013, 31019]
METRICS = ("knn", "linear", "final_conditions")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a finite, human-readable study artifact."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lock_json(path: Path, value: dict[str, Any]) -> None:
    """Preserve the first record and reject inconsistent resumes."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["payload"] != value:
            raise ValueError(f"Study fingerprint mismatch: {path}")
    else:
        write_json(path, {"created_utc": datetime.now(timezone.utc).isoformat(), "payload": value})


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash tensor content in bounded chunks without copying an entire image corpus."""
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    for start in range(0, len(tensor), 128):
        chunk = tensor[start : start + 128]
        digest.update(chunk.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def eligible_conditions(panels: dict[str, Any]) -> list[int]:
    """Select common conditions using counts only, never prediction outcomes."""
    sets = [
        {
            int(era)
            for era, cell in panel["conditions"].items()
            if cell["n"] >= 30 and sum(n >= 5 for n in cell["class_counts"].values()) >= 2
        }
        for panel in panels.values()
    ]
    return sorted(set.intersection(*sets))


def preflight(config: DictConfig, out: Path) -> dict[str, Any]:
    """Decode once, lock evaluation composition, and verify paired stream orders before training."""
    source = _shared_source(config.data)
    evaluation = _shared_source(config.evaluation_data)
    panels: dict[str, Any] = {}
    for panel_seed in PANELS:
        stream = instantiate(config.stream, source=source, evaluation_source=evaluation, canary_seed=panel_seed)
        eval_sets = stream.eval_sets
        conditions = {}
        for era, data in eval_sets.per_era.items():
            labels, counts = data.labels.unique(return_counts=True)
            conditions[str(era)] = {
                "name": stream.era_names[era],
                "n": len(data.labels),
                "class_counts": {str(int(k)): int(v) for k, v in zip(labels, counts, strict=True)},
                "images_sha256": tensor_hash(data.images),
                "labels_sha256": tensor_hash(data.labels),
            }
        panels[str(panel_seed)] = {
            "conditions": conditions,
            "support": len(eval_sets.probe_support.labels),
            "global_query": len(eval_sets.probe_query.labels),
            "global_sha256": {
                name: [tensor_hash(data.images), tensor_hash(data.labels)]
                for name, data in (("support", eval_sets.probe_support), ("query", eval_sets.probe_query))
            },
        }
    eligible = eligible_conditions(panels)
    final_era = max(stream.era_names)
    if len([era for era in eligible if era < final_era]) < 2:
        raise ValueError("Fewer than two well-sampled past conditions; cannot run the registered test.")
    orders = {}
    for seed in SEEDS:
        hashes = []
        for _scope in ("full", "last_block"):
            paired = instantiate(config.stream, source=source, evaluation_source=evaluation, seed=seed)
            hashes.append(tensor_hash(torch.tensor(paired._order_stream)))
        if hashes[0] != hashes[1]:
            raise ValueError("Paired training orders differ.")
        orders[str(seed)] = hashes[0]
    result = {
        "panels": panels,
        "eligible_conditions": eligible,
        "excluded_conditions": sorted(set(stream.era_names) - set(eligible)),
        "era_names": {str(k): v for k, v in stream.era_names.items()},
        "train_order_sha256": orders,
        "train_images": len(source.load().images),
        "train_images_sha256": tensor_hash(source.load().images),
        "evaluation_images": len(evaluation.load().images),
        "updates_per_arm": len(stream),
    }
    lock_json(out / "evaluation_protocol.json", result)
    logger.info(f"Preflight: {len(eligible)} eligible conditions, {len(stream)} updates per arm")
    return result


def retention(matrix: dict[str, Any], eligible: list[int] | None) -> dict[str, float]:
    """Restrict condition columns but retain every chronological checkpoint, including the last."""
    selected = {
        int(after): {int(era): float(acc) for era, acc in row.items() if eligible is None or int(era) in eligible}
        for after, row in matrix.items()
    }
    values = {"forgetting": forgetting_measure(selected), "bwt": backward_transfer(selected)}
    if any(value is None for value in values.values()):
        raise ValueError("Insufficient condition history for retention.")
    return {key: float(value) for key, value in values.items() if value is not None}


def read_arm(directory: Path, seed: int, eligible: list[int] | None) -> dict[str, Any]:
    """Reduce panel scores with an explicit base-panel identity and fixed eligible conditions."""
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))["seeds"]
    if len(report) != 1 or report[0]["seed"] != seed:
        raise ValueError("Unexpected training seed in comparison.")
    row = report[0]
    base = {
        "panel_seed": PANELS[0],
        "global": row["global"],
        "final_per_condition": row["conditions"]["final_per_condition"],
    }
    raw_panels = [base, *row["probe_panels"]]
    if [p["panel_seed"] for p in raw_panels] != PANELS:
        raise ValueError("Unexpected evaluation panels.")
    panels = {}
    for panel in raw_panels:
        conditions = [r for r in panel["final_per_condition"] if eligible is None or int(r["eval_era"]) in eligible]
        if not conditions or (eligible is not None and {int(r["eval_era"]) for r in conditions} != set(eligible)):
            raise ValueError("Missing registered conditions.")
        scores = {
            f"{arm}_{metric}": float(panel["global"][metric][key])
            for metric in ("knn", "linear")
            for arm, key in (("adapted", "adapted_acc"), ("frozen", "b5_acc"))
        }
        for arm in ("adapted", "frozen"):
            scores[f"{arm}_final_conditions"] = statistics.fmean(r[f"{arm}_acc"] for r in conditions)
        panels[str(panel["panel_seed"])] = scores
    return {
        "panels": panels,
        "absolute": {key: statistics.fmean(p[key] for p in panels.values()) for key in next(iter(panels.values()))},
        "retention": retention(row["conditions"]["adapted_matrix"], eligible),
        "frozen_conditions": [
            [(int(r["eval_era"]), float(r["frozen_acc"])) for r in p["final_per_condition"]] for p in raw_panels
        ],
    }


def validate_pair(out: Path, seed: int) -> None:
    """Require matching settings except the intended trainable scope."""
    configs = []
    for scope in ("full", "last_block"):
        config = OmegaConf.to_container(OmegaConf.load(out / scope / f"seed_{seed}/config.yaml"), resolve=True)
        if not isinstance(config, dict) or config["train_scope"] != scope or config["seeds"] != [seed]:
            raise ValueError("Wrong paired scope or seed.")
        config.pop("train_scope")
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError("Paired configuration mismatch.")


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired five-seed accuracy and retention effects without pooling panels as replicates."""
    contrasts = (
        ("last_minus_full", "last_block", "full", "adapted"),
        ("last_minus_frozen", "last_block", "last_block", "frozen"),
        ("full_minus_frozen", "full", "full", "frozen"),
    )
    metrics = {
        name: {
            key: paired_interval(
                [row[a]["absolute"][f"adapted_{key}"] - row[b]["absolute"][f"{target}_{key}"] for row in rows]
            )
            for key in METRICS
        }
        for name, a, b, target in contrasts
    }
    effects = {
        key: paired_interval([row["last_block"]["retention"][key] - row["full"]["retention"][key] for row in rows])
        for key in ("forgetting", "bwt")
    }
    return {
        "metrics": metrics,
        "retention_last_minus_full": effects,
        "retention_means_pp": {
            scope: {
                key: 100 * statistics.fmean(row[scope]["retention"][key] for row in rows)
                for key in ("forgetting", "bwt")
            }
            for scope in ("full", "last_block")
        },
        "retention_accuracy_confirmed": effects["forgetting"]["ci95_pp"][1] < 0
        and all(m["ci95_pp"][0] > -1 for m in metrics["last_minus_full"].values()),
        "frozen_advantage_verdict": verdict(metrics["last_minus_frozen"]),
        "broader_accuracy_superiority": verdict(metrics["last_minus_frozen"]) == "supported"
        and metrics["last_minus_frozen"]["linear"]["ci95_pp"][0] > 0,
        "per_panel": {
            p: {
                name: {
                    key: paired_interval(
                        [row[a]["panels"][p][f"adapted_{key}"] - row[b]["panels"][p][f"{target}_{key}"] for row in rows]
                    )
                    for key in METRICS
                }
                for name, a, b, target in contrasts
            }
            for p in map(str, PANELS)
        },
        "seeds": rows,
    }


def audit_weights(out: Path, well: Path) -> dict[str, Any]:
    """Check frozen-layer integrity and finite final checkpoints for every completed arm."""
    initial = torch.load(well, map_location="cpu", weights_only=True)
    results = {}
    for seed in SEEDS:
        for scope in ("full", "last_block"):
            final = torch.load(
                out / scope / f"seed_{seed}/checkpoints/adapted_s{seed}.pt", map_location="cpu", weights_only=True
            )
            if final.keys() != initial.keys() or any(not torch.isfinite(v).all() for v in final.values()):
                raise ValueError("Invalid final checkpoint.")
            changed = [key for key in initial if not torch.equal(initial[key], final[key])]
            groups = ("encoder.blocks.3.", "encoder.norm.", "decoder.")
            if not all(any(key.startswith(g) for key in changed) for g in groups):
                raise ValueError("Expected trainable groups did not change.")
            if scope == "last_block" and any(not key.startswith(groups) for key in changed):
                raise ValueError("Frozen encoder tensors changed.")
            if scope == "full" and not any(key.startswith("encoder.blocks.0.") for key in changed):
                raise ValueError("Full-backbone arm did not update its early encoder.")
            results[f"{scope}/{seed}"] = {"passed": True, "changed_tensors": changed}
    return results


def check_frozen_pair(row: dict[str, Any]) -> None:
    """Reject discrepancies in any paired frozen global or individual condition score."""
    if row["full"]["frozen_conditions"] != row["last_block"]["frozen_conditions"]:
        raise ValueError("Frozen condition scores differ.")
    for panel in map(str, PANELS):
        for key, value in row["full"]["panels"][panel].items():
            if key.startswith("frozen_") and abs(value - row["last_block"]["panels"][panel][key]) > 1e-10:
                raise ValueError("Frozen panel scores differ.")


def summarize(out: Path) -> dict[str, Any]:
    """Audit complete paired artifacts and emit primary and mandatory sensitivity analyses."""
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))["payload"]
    evaluation = json.loads((out / "evaluation_protocol.json").read_text(encoding="utf-8"))["payload"]
    well = Path(protocol["config"]["well"])
    if hashlib.sha256(well.read_bytes()).hexdigest() != protocol["checkpoint_sha256"]:
        raise ValueError("Warm checkpoint changed.")
    analyses = {}
    for name, eligible in (("primary", evaluation["eligible_conditions"]), ("all_conditions_secondary", None)):
        rows = []
        for seed in SEEDS:
            validate_pair(out, seed)
            row = {
                "seed": seed,
                **{scope: read_arm(out / scope / f"seed_{seed}", seed, eligible) for scope in ("full", "last_block")},
            }
            check_frozen_pair(row)
            rows.append(row)
        analyses[name] = reduce_rows(rows)
    write_json(out / "weight_audit.json", audit_weights(out, well))
    result = {"study": "Fresh-seed restricted-plasticity confirmation, 2026-09-28", **analyses}
    write_json(out / "summary.json", result)
    lines = [
        "# Fresh-seed last-block confirmation",
        "",
        "Five paired training seeds; three panels averaged within seed. Retention uses panel 31001.",
        "Intervals are conditional on the reused warm checkpoint and validation corpus.",
        "",
    ]
    for name, report in analyses.items():
        lines += [
            f"## {name}",
            "",
            f"Retention/accuracy rule passed: {report['retention_accuracy_confirmed']}",
            f"Frozen advantage: {report['frozen_advantage_verdict']}",
            "",
        ]
        for contrast, values in {
            **report["metrics"],
            "retention_last_minus_full": report["retention_last_minus_full"],
        }.items():
            lines += [
                f"### {contrast}",
                "",
                "| Metric | Mean difference (pp) | 95% interval |",
                "| --- | ---: | ---: |",
            ]
            for metric, value in values.items():
                low, high = value["ci95_pp"]
                lines.append(f"| {metric} | {value['mean_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Execute all ten online arms with immutable provenance and safe per-arm resume checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.summarize_only:
        summarize(out)
        return
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "cafl4ds/configs")):
        config = compose(config_name="adaptation_bdd_last_block_confirmation")
    config.well = (ROOT / str(config.well)).resolve().as_posix()
    if list(config.seeds) != SEEDS or [int(config.stream.canary_seed), *config.probe_panel_seeds] != PANELS:
        raise ValueError("Configuration no longer matches the fixed protocol.")
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "cafl4ds").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    lock_json(
        out / "protocol.json",
        {
            "config": OmegaConf.to_container(config, resolve=True),
            "checkpoint_sha256": hashlib.sha256(Path(str(config.well)).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
            "torch_version": torch.__version__,
        },
    )
    if not (out / "protocol.md").exists():
        (out / "protocol.md").write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    logger.add(out / "execution.log", level="INFO")
    logger.info("Loading data and locking count-only evaluation preflight before any training")
    preflight(config, out)
    for index, seed in enumerate(SEEDS):
        for scope in ("full", "last_block") if index % 2 == 0 else ("last_block", "full"):
            directory = out / scope / f"seed_{seed}"
            one = copy.deepcopy(config)
            one.seed = seed
            one.stream.seed = seed
            one.seeds = [seed]
            one.train_scope = scope
            if (directory / "comparison.json").exists():
                if OmegaConf.to_container(
                    OmegaConf.load(directory / "config.yaml"), resolve=True
                ) != OmegaConf.to_container(one, resolve=True):
                    raise ValueError("Completed arm config mismatch.")
                logger.info(f"Skipping completed {scope} seed {seed}")
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Partial arm at {directory}; preserve it and use a new output directory.")
            directory.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(one, directory / "config.yaml", resolve=True)
            logger.info(f"Starting {scope} seed {seed}")
            run_experiment(one, directory)
            logger.info(f"Completed {scope} seed {seed}")
    result = summarize(out)
    passed = result["primary"]["retention_accuracy_confirmed"]
    logger.info(f"All ten arms finished. Primary retention/accuracy confirmation: {passed}")


if __name__ == "__main__":
    main()
