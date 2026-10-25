"""Independently rescore saved recurrence predictions, intervals, budgets and input locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from cafl4ds.artifacts import atomic_json

ARMS = ["recurrence_no_replay", "recurrence_replay", "shuffled_no_replay", "shuffled_replay"]
KINDS = ["knn", "linear", "finetune"]


def digest(path: Path) -> str:
    """Hash an artifact without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float | list[float], expected: float | list[float], name: str) -> None:
    """Reject numerical disagreement rather than replace a failed result."""
    if not np.allclose(actual, expected, rtol=0, atol=1e-7):
        raise ValueError(f"Independent mismatch: {name}")


def scores(predicted: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict[str, float]:
    """Compute balanced recall directly from class predictions."""
    if predicted.shape != labels.shape or not np.isin(predicted, np.unique(labels)).all():
        raise ValueError("Prediction shape/classes differ from the query allocation")
    correct = (predicted == labels).astype(np.float32)
    result = {name: float(np.mean([
        float(correct[(domains == domain) & (labels == label)].mean()) for label in np.unique(labels)
    ])) for domain, name in enumerate(("A", "B"))}
    result.update(balanced=(result["A"] + result["B"]) / 2, micro=float(correct.mean()))
    return result


def metrics(trace: list[dict[str, Any]], visit: int, kind: str) -> dict[str, float]:
    """Compute absent-domain interference and endpoint changes independently."""
    r = {row["step"] // visit: row["scores"][kind] for row in trace if row["step"] % visit == 0}
    if set(r) != set(range(5)):
        raise ValueError("Incomplete recurrence boundary grid")
    a, b = r[1]["A"] - r[2]["A"], r[2]["B"] - r[3]["B"]
    return {
        "interference": (a + b) / 2, "absence_A": a, "absence_B": b,
        "bwt": ((r[4]["A"] - r[1]["A"]) + (r[4]["B"] - r[2]["B"])) / 2,
        "forgetting": ((max(r[i]["A"] for i in (1, 2, 3)) - r[4]["A"])
            + (max(r[i]["B"] for i in (2, 3)) - r[4]["B"])) / 2,
        "return_gain_A": r[3]["A"] - r[2]["A"], "return_gain_B": r[4]["B"] - r[3]["B"],
        "return_vs_first_A": r[3]["A"] - r[1]["A"], "return_vs_first_B": r[4]["B"] - r[2]["B"],
        "first_gain_A": r[1]["A"] - r[0]["A"], "first_gain_B": r[2]["B"] - r[0]["B"],
        "final_balanced": r[4]["balanced"], "final_vs_warm": r[4]["balanced"] - r[0]["balanced"],
    }


def interval(values: list[float]) -> dict[str, Any]:
    """Five-seed Student-t inference; support draws are averaged earlier."""
    if len(values) != 5:
        raise ValueError("Expected exactly five seeds")
    sample = np.asarray(values, dtype=float) * 100
    mean = float(sample.mean())
    width = float(t.ppf(0.975, 4) * sample.std(ddof=1) / np.sqrt(5))
    return {"mean_pp": mean, "ci95_pp": [mean - width, mean + width]}


def order_hash(order: list[int]) -> str:
    """Reproduce the locked int64 tensor shape/dtype/content fingerprint."""
    value = hashlib.sha256(f"(({len(order)},), torch.int64)".encode())
    value.update(np.asarray(order, dtype=np.int64).tobytes())
    return value.hexdigest()


def audit(out: Path) -> dict[str, Any]:  # noqa: C901 - explicit complete collection checks
    """Check all saved predictions, locked inputs, paired summaries and decision rules."""
    inputs = json.loads((out / "input_lock.json").read_text())["payload"]
    data = json.loads((out / "data_lock.json").read_text())["payload"]
    provenance = json.loads((out / "restart_provenance.json").read_text())
    for name, expected in {**inputs["hashes"], **provenance["prior_hashes"]}.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed input: {name}")
    seeds = inputs["config"]["seeds"]
    preflight = json.loads((out / "preflight.json").read_text())
    if not preflight["passed"] or preflight["evaluation_arrays_sha256"] != digest(out / "evaluation_arrays.npz"):
        raise ValueError("Evaluation allocation changed since preflight")
    arrays = np.load(out / "evaluation_arrays.npz", allow_pickle=False)
    query = arrays["query"]
    labels, domains = arrays["labels"][query], arrays["domains"][query]
    if query.tolist() != data["evaluation"]["query_positions"]:
        raise ValueError("Query identities differ")
    for support in data["evaluation"]["supports"].values():
        if set(support) & set(query.tolist()) or len(support) != 75:
            raise ValueError("Support/query leakage or label budget mismatch")
        if np.bincount(arrays["labels"][support]).tolist() != [25, 25, 25]:
            raise ValueError("Support class balance differs")
    rows: dict[str, list[dict[str, dict[str, float]]]] = {arm: [] for arm in ARMS}
    checked = 0
    for arm in ARMS:
        for seed in seeds:
            directory = out / arm / f"seed_{seed}"
            result = json.loads((directory / "result.json").read_text())
            order = data["orders"][str(seed)]["recurrence" if arm.startswith("recurrence") else "shuffled"]
            if len(order) != len(set(order)) or len(order) != 18432:
                raise ValueError("Unique single-pass arrival contract failed")
            # tensor_hash uses tensor bytes, independently reconstructed here.
            order_sha = order_hash(order)
            if result["arrival_order_sha256"] != order_sha:
                raise ValueError("Arrival order differs")
            expected_budget = {"arrivals": 18432, "updates": 576, "optimizer_images": 18432,
                "reservoir_seen": 0 if arm.endswith("no_replay") else 18432,
                "reservoir_size": 0 if arm.endswith("no_replay") else 256}
            if result["budget"] != expected_budget or result["encoder_delta_l2"] <= 0:
                raise ValueError("Optimization budget or encoder update failed")
            if set(order) != set(data["orders"][str(seed)]["shuffled"]):
                raise ValueError("Shuffled control uses a different population")
            if any(not (directory / "checkpoints" / f"boundary_{i}.pt").is_file() for i in range(1, 5)):
                raise ValueError("Missing saved boundary model")
            losses = json.loads((directory / "losses.json").read_text())["losses"]
            if [row["step"] for row in losses] != list(range(1, 577)):
                raise ValueError("Missing optimizer update")
            if not np.isfinite([[row["loss"], row["grad_norm"]] for row in losses]).all():
                raise ValueError("Nonfinite optimizer observations")
            trace = result["trajectory"]
            if [row["step"] for row in trace] != list(range(0, 577, 36)):
                raise ValueError("Incomplete observation grid")
            for point in trace:
                base = out / "frozen" if point["step"] == 0 else directory
                predictions = np.load(base / f"predictions_{point['step']:06d}.npz", allow_pickle=False)
                kinds = KINDS if point["step"] % 144 == 0 else KINDS[:2]
                for kind in kinds:
                    panel_scores = []
                    for draw in inputs["config"]["recurrence"]["draw_seeds"]:
                        recomputed = scores(predictions[f"{draw}_{kind}"], labels, domains)
                        panel_scores.append(recomputed)
                        for metric, value in recomputed.items():
                            close(point["panels"][str(draw)][kind][metric], value, "panel scoring")
                        checked += 1
                    for metric in ("A", "B", "balanced", "micro"):
                        close(
                            point["scores"][kind][metric],
                            float(np.mean([p[metric] for p in panel_scores])), "panel mean",
                        )
                if point["step"] % 144 == 0:
                    for record in point["fine_tuning_audits"].values():
                        if record["steps"] != 100 or record["backbone_delta_l2"] <= 0:
                            raise ValueError("Copied fine-tuning budget failed")
            recomputed_metrics = {kind: metrics(trace, 144, kind) for kind in KINDS}
            for kind in KINDS:
                for metric, value in recomputed_metrics[kind].items():
                    close(result["metrics"][kind][metric], value, "retention metrics")
            rows[arm].append(recomputed_metrics)
    summary = json.loads((out / "summary.json").read_text())
    pairs = [("replay_minus_no_replay", ARMS[1], ARMS[0]), ("order_effect_no_replay", ARMS[0], ARMS[2]),
        ("order_effect_replay", ARMS[1], ARMS[3]), ("shuffled_replay_effect", ARMS[3], ARMS[2])]
    for kind in KINDS:
        for name, first, second in pairs:
            for metric in ("interference", "final_balanced", "bwt", "forgetting"):
                independent = interval([a[kind][metric] - b[kind][metric]
                    for a, b in zip(rows[first], rows[second], strict=True)])
                saved = summary["readouts"][kind]["contrasts"][name][metric]
                if saved["n_seeds"] != 5:
                    raise ValueError("Seed pseudoreplication")
                for key, value in independent.items():
                    close(saved[key], value, "paired inference")
        for arm in ARMS:
            for metric in ("interference", "final_vs_warm"):
                independent = interval([r[kind][metric] for r in rows[arm]])
                saved = summary["readouts"][kind]["against_zero"][arm][metric]
                for key, value in independent.items():
                    close(saved[key], value, "absolute inference")
    primary = summary["readouts"]["finetune"]
    gates = {
        "degradation_present": primary["against_zero"][ARMS[0]]["interference"]["ci95_pp"][0] > 0,
        "replay_reduces_interference": primary["contrasts"][pairs[0][0]]["interference"]["ci95_pp"][1] < 0,
        "final_accuracy_noninferior": primary["contrasts"][pairs[0][0]]["final_balanced"]["ci95_pp"][0] > -1,
    }
    if gates != summary["primary_gates"] or all(gates.values()) != summary["retention_supported"]:
        raise ValueError("Primary conjunction differs")
    if summary["replay_final_advantage_vs_warm"] != (
        primary["against_zero"][ARMS[1]]["final_vs_warm"]["ci95_pp"][0] > 0
    ):
        raise ValueError("Frozen-advantage decision differs")
    result = {"passed": True, "arms_checked": 20, "prediction_panels_checked": checked,
        "input_and_prior_hashes_unchanged": True, "retention_supported": all(gates.values()),
        "scope": "Saved prediction scoring, paired inference and provenance; not independent model retraining.",
        "artifact_hashes": {str(p.relative_to(out)): digest(p) for p in out.rglob("*")
            if p.is_file() and (p.suffix in {".npz", ".pt"} or p.name in {"result.json", "summary.json"})},
    }
    atomic_json(out / "independent_verification.json", result)
    root = Path(__file__).resolve().parents[1]
    text = (out / "summary.md").read_text() + (
        "\nIndependent saved-prediction scoring, seed inference and input/provenance audit passed. "
        "This is a five-seed conditional study with reused BDD evaluation and warm initialization; "
        "it does not establish new-population transfer or health control.\n\n"
        f"Run artifacts: `{out.relative_to(root).as_posix()}`.\n"
    )
    for folder in ("phase1", "latest-results"):
        (root / "docs/experiments" / folder / "slow-edge-recurrence-restart.md").write_text(text, encoding="utf-8")
    return result


def main() -> None:
    """Audit a completed collection without training or selecting a recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
