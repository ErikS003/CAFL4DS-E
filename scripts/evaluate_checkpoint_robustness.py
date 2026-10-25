"""Fixed-pool BDD readouts, including genuine fine-tuning, of saved stream checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from cafl4ds.models.vit import TinyViTEncoder
from scripts.confirm_last_block import lock_json as _lock_json
from scripts.confirm_last_block import tensor_hash, write_json
from scripts.confirm_replay_advantage import paired_interval

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/adaptation-bdd/checkpoint-robustness/20260930"
PROTOCOL = ROOT / "docs/experiments/phase1/P1.2-checkpoint-robustness.md"
CONTROL = ROOT / "outputs/adaptation-bdd/last-block-confirmation/20260928"
REPLAY = ROOT / "outputs/adaptation-bdd/scope-replay/20260929"
SEEDS = [3001, 3011, 3019, 3023, 3037]
DRAWS = [41011, 41017, 41023]
BUDGETS = [5, 25, 100]
ARMS = ["full_no_replay", "full_replay", "last_block_no_replay", "last_block_replay"]
READOUTS = ["knn", "linear", "finetune"]
METRICS = ["global_micro", "global_balanced", "conditions", "all_conditions"]


def lock_json(path: Path, payload: dict[str, Any]) -> None:
    """Canonicalize integer mapping keys before comparing a resumed JSON lock."""
    _lock_json(path, json.loads(json.dumps(payload)))


def digest(path: Path) -> str:
    """Fingerprint read-only inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_pools(labels: torch.Tensor, regimes: torch.Tensor, seed: int = 41001) -> tuple[torch.Tensor, torch.Tensor]:
    """Fix disjoint query/support membership within regime/class cells."""
    generator = torch.Generator().manual_seed(seed)
    support, query = [], []
    for regime in regimes.unique(sorted=True):
        for label in labels.unique(sorted=True):
            indices = ((labels == label) & (regimes == regime)).nonzero().flatten()
            indices = indices[torch.randperm(len(indices), generator=generator)]
            cut = len(indices) // 2
            query.extend(indices[:cut].tolist())
            support.extend(indices[cut:].tolist())
    return torch.tensor(support), torch.tensor(query)


def support_draw(labels: torch.Tensor, pool: torch.Tensor, seed: int, shots: int) -> torch.Tensor:
    """Balanced nested support draws; never inspect query labels to fit a model."""
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in labels.unique(sorted=True):
        candidates = pool[labels[pool] == label]
        if len(candidates) < shots:
            raise ValueError(f"Insufficient support for class {label}: {len(candidates)} < {shots}")
        indices = candidates[torch.randperm(len(candidates), generator=generator)]
        selected.extend(indices[:shots].tolist())
    return torch.tensor(selected)


def qualified_conditions(labels: torch.Tensor, regimes: torch.Tensor) -> list[int]:
    """Choose eligible condition IDs from counts alone."""
    return [
        int(r)
        for r in regimes.unique(sorted=True)
        if int((regimes == r).sum()) >= 30 and int((labels[regimes == r].bincount() >= 5).sum()) >= 2
    ]


def scores(
    predictions: torch.Tensor, labels: torch.Tensor, regimes: torch.Tensor, eligible: list[int]
) -> dict[str, Any]:
    """Aggregate identical predictions under global and condition-balanced weights."""
    correct = predictions == labels
    per_condition = {str(int(r)): float(correct[regimes == r].float().mean()) for r in regimes.unique(sorted=True)}
    return {
        "global_micro": float(correct.float().mean()),
        "global_balanced": statistics.fmean(float(correct[labels == c].float().mean()) for c in labels.unique()),
        "conditions": statistics.fmean(per_condition[str(r)] for r in eligible),
        "all_conditions": statistics.fmean(per_condition.values()),
        "per_condition": per_condition,
    }


def features(encoder: TinyViTEncoder, images: torch.Tensor) -> torch.Tensor:
    """Bound memory and keep evaluation free of gradients."""
    encoder.eval()
    with torch.no_grad():
        return torch.cat([encoder.embed(batch).cpu() for batch in images.split(64)])


def probe_predictions(
    kind: str, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor
) -> torch.Tensor:
    """Match existing frozen-feature readout recipes, retaining query predictions."""
    xs, xq = support_x.numpy().copy(), query_x.numpy().copy()
    if kind == "linear":
        mean, std = xs.mean(0), xs.std(0)
        std[std == 0] = 1
        xs, xq = (xs - mean) / std, (xq - mean) / std
        classifier = LogisticRegression(C=1, max_iter=1000)
    elif kind == "knn":
        classifier = KNeighborsClassifier(n_neighbors=min(20, len(xs)), metric="cosine", weights="distance")
    else:
        raise ValueError(kind)
    classifier.fit(xs, support_y.numpy())
    return torch.from_numpy(classifier.predict(xq))


def finetune(
    encoder: TinyViTEncoder, images: torch.Tensor, labels: torch.Tensor, seed: int, steps: int = 100
) -> tuple[TinyViTEncoder, torch.nn.Linear, dict[str, Any]]:
    """Fit an independent encoder copy plus head using support data only."""
    model = copy.deepcopy(encoder)
    model.requires_grad_(True)
    model.train()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = torch.nn.Linear(model.embed_dim, int(labels.max()) + 1)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.05,
    )
    losses = []
    for _ in range(steps):
        indices = torch.randperm(len(labels), generator=generator)[:32]
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(model.embed(images[indices])), labels[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    delta = (
        sum(
            float((p.detach() - q.detach()).square().sum())
            for p, q in zip(model.parameters(), encoder.parameters(), strict=True)
        )
        ** 0.5
    )
    if delta <= 0 or not np.isfinite(losses).all():
        raise RuntimeError("Fine-tuning did not make finite backbone updates")
    with torch.no_grad():
        support_acc = float((head(features(model, images)).argmax(1) == labels).float().mean())
    return (
        model,
        head,
        {
            "backbone_delta_l2": delta,
            "support_accuracy": support_acc,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "steps": steps,
        },
    )


def checkpoint_inputs() -> tuple[DictConfig, dict[str, Path]]:
    """Locate the exact 20 completed models plus common warm starting point."""
    result = {}
    for arm in ARMS:
        base = CONTROL if arm.endswith("no_replay") else REPLAY
        scope = "last_block" if arm.startswith("last") else "full"
        for seed in SEEDS:
            directory = base / scope / f"seed_{seed}"
            result[f"{arm}_{seed}"] = directory / "checkpoints" / f"adapted_s{seed}.pt"
    config = OmegaConf.load(CONTROL / "full/seed_3001/config.yaml")
    result["frozen"] = Path(config.well)
    return config, result


def summarize(out: Path) -> None:
    """Average draws within training seed, never treating draws as independent seeds."""
    result: dict[str, Any] = {}
    for shots in BUDGETS:
        for readout in READOUTS:
            cells = {}
            for arm in ["frozen", *ARMS]:
                rows = []
                for seed in [None] if arm == "frozen" else SEEDS:
                    key = arm if seed is None else f"{arm}_{seed}"
                    draws = [
                        json.loads((out / "cells" / f"{key}_{shots}_{draw}.json").read_text())[readout]
                        for draw in DRAWS
                    ]
                    rows.append({m: statistics.fmean(d[m] for d in draws) for m in METRICS})
                cells[arm] = rows
            contrasts = {}
            pairs = [(f"{a}_minus_frozen", a, "frozen") for a in ARMS]
            pairs += [
                (f"{scope}_replay_minus_no_replay", f"{scope}_replay", f"{scope}_no_replay")
                for scope in ["full", "last_block"]
            ]
            for name, a, b in pairs:
                contrasts[name] = {
                    m: paired_interval([cells[a][i][m] - cells[b][0 if b == "frozen" else i][m] for i in range(5)])
                    for m in METRICS
                }
            result[f"{shots}_{readout}"] = {"contrasts": contrasts, "seed_means": cells}
    primary = result["25_finetune"]["contrasts"]["last_block_replay_minus_frozen"]
    report: dict[str, Any] = {
        "readouts": result,
        "primary_joint_positive": all(primary[m]["ci95_pp"][0] > 0 for m in ["global_balanced", "conditions"]),
        "caveat": "Diagnostic reuse; nominal seed intervals conditional on a shared fixed warm well and query pool.",
    }
    write_json(out / "summary.json", report)
    lines = [
        "# Checkpoint evaluation robustness",
        "",
        report["caveat"],
        "",
        "Gains vs warm frozen (pp; paired nominal 95% CI). Three draws averaged within each of five seeds.",
        "",
        "| Labels/class | Readout | Arm | Global balanced | Qualified conditions |",
        "|---|---|---|---|---|",
    ]
    for key, value in result.items():
        shots_label, readout = key.split("_")
        for arm in ARMS:
            row = value["contrasts"][f"{arm}_minus_frozen"]
            formatted = [
                f"{row[m]['mean_pp']:+.2f} [{row[m]['ci95_pp'][0]:+.2f}, {row[m]['ci95_pp'][1]:+.2f}]"
                for m in ["global_balanced", "conditions"]
            ]
            lines.append(f"| {shots_label} | {readout} | {arm} | {' | '.join(formatted)} |")
    lines += [
        "",
        f"Primary joint positive: {report['primary_joint_positive']}",
        "",
        "Not a new independent confirmation; no new forgetting/BWT estimate. Full results in summary.json.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_report(result, out)


def plot_report(result: dict[str, Any], out: Path) -> None:
    """Plot all budgets without selecting a winning readout."""
    fig = Figure(figsize=(14, 8), constrained_layout=True)
    axes = fig.subplots(2, 3)
    for col, readout in enumerate(READOUTS):
        for row, metric in enumerate(["global_balanced", "conditions"]):
            ax = axes[row, col]
            for offset, arm in enumerate(ARMS):
                values = [result[f"{s}_{readout}"]["contrasts"][f"{arm}_minus_frozen"][metric] for s in BUDGETS]
                means = np.array([v["mean_pp"] for v in values])
                bounds = np.array([v["ci95_pp"] for v in values]).T
                ax.errorbar(
                    np.arange(3) + (offset - 1.5) * 0.12,
                    means,
                    yerr=np.vstack([means - bounds[0], bounds[1] - means]),
                    marker="o",
                    capsize=2,
                    label=arm,
                    linewidth=1,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(3), BUDGETS)
            ax.set(xlabel="Labels per class", ylabel="Gain vs warm frozen (pp)", title=f"{readout}: {metric}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fixed-query checkpoint evaluation • nominal paired 95% seed intervals")
    fig.savefig(out / "robustness.png", dpi=150)


def main() -> None:  # noqa: C901 - explicit, resumable evaluation state machine
    """Run/resume immutable evaluation cells and maintain truthful progress."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cells").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    progress: dict[str, Any] = {"state": "preflight", "pid": os.getpid(), "completed_cells": 0, "total_cells": 189}

    def status(**kwargs: str | bool | int | float) -> None:
        progress.update(kwargs, elapsed_seconds=time.time() - started)
        write_json(OUTPUT / "progress.json", progress)
        print(json.dumps(progress), flush=True)

    try:
        status()
        config, inputs = checkpoint_inputs()
        source_files = [
            PROTOCOL,
            Path(__file__),
            ROOT / "cafl4ds/models/vit.py",
            ROOT / "cafl4ds/data/attributes.py",
            ROOT / "scripts/confirm_last_block.py",
            ROOT / "scripts/confirm_replay_advantage.py",
        ]
        input_hashes = {str(p): digest(p) for p in [*inputs.values(), *source_files]}
        lock_json(
            OUTPUT / "input_lock.json",
            {
                "hashes": input_hashes,
                "evaluation_config": OmegaConf.to_container(config.evaluation_data, resolve=True),
                "encoder_config": OmegaConf.to_container(config.encoder, resolve=True),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn_version,
            },
        )
        data = instantiate(config.evaluation_data).load()
        pool, query = split_pools(data.canary, data.era_key)
        eligible = qualified_conditions(data.canary[query], data.era_key[query])
        if not eligible:
            raise ValueError("No count-qualified conditions")
        draws = {f"{shots}_{draw}": support_draw(data.canary, pool, draw, shots) for shots in BUDGETS for draw in DRAWS}
        lock_json(
            OUTPUT / "evaluation_pool.json",
            {
                "support_pool": pool.tolist(),
                "query": query.tolist(),
                "draws": {k: v.tolist() for k, v in draws.items()},
                "eligible_conditions": eligible,
                "regime_names": data.regime_names,
                "class_names": data.canary_names,
                "images_sha256": tensor_hash(data.images),
                "labels_sha256": tensor_hash(data.canary),
                "regimes_sha256": tensor_hash(data.era_key),
                "query_counts": {
                    str(int(r)): torch.bincount(
                        data.canary[query][data.era_key[query] == r], minlength=len(data.canary_names)
                    ).tolist()
                    for r in data.era_key.unique()
                },
            },
        )
        query_images, query_y, query_r = data.images[query], data.canary[query], data.era_key[query]
        # Warm first; every learned cell is paired with these identical support/query readouts.
        for key in ["frozen", *[k for k in inputs if k != "frozen"]]:
            pending = [cell for cell in draws if not (OUTPUT / "cells" / f"{key}_{cell}.json").exists()]
            if not pending:
                progress["completed_cells"] += len(draws)
                continue
            status(state="encoding", model=key)
            encoder = instantiate(config.encoder)
            checkpoint = torch.load(inputs[key], map_location="cpu", weights_only=True)
            encoder.load_state_dict(
                {k.removeprefix("encoder."): v for k, v in checkpoint.items() if k.startswith("encoder.")}, strict=True
            )
            encoder.requires_grad_(False)
            encoded = features(encoder, data.images)
            for cell, indices in draws.items():
                path = OUTPUT / "cells" / f"{key}_{cell}.json"
                if path.exists():
                    progress["completed_cells"] += 1
                    continue
                status(state="evaluating", model=key, cell=cell)
                shots, draw = map(int, cell.split("_"))
                predictions, record = {}, {"model": key, "shots": shots, "draw": draw}
                for readout in ["knn", "linear"]:
                    predictions[readout] = probe_predictions(
                        readout, encoded[indices], data.canary[indices], encoded[query]
                    )
                tuned, head, audit = finetune(encoder, data.images[indices], data.canary[indices], draw + shots)
                with torch.no_grad():
                    predictions["finetune"] = head(features(tuned, query_images)).argmax(1)
                for readout, prediction in predictions.items():
                    record[readout] = scores(prediction, query_y, query_r, eligible)
                record["fine_tuning_audit"] = audit
                np.savez_compressed(path.with_suffix(".npz"), **{k: v.numpy() for k, v in predictions.items()})
                temporary = path.with_suffix(".tmp")
                write_json(temporary, record)
                temporary.replace(path)
                progress["completed_cells"] += 1
                status()
        if any(digest(Path(p)) != expected for p, expected in input_hashes.items()):
            raise ValueError("Read-only inputs changed during evaluation")
        summarize(OUTPUT)
        status(state="completed", inputs_unchanged=True)
    except Exception as error:
        status(state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
