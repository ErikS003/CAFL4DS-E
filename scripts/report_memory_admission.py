"""Independently reconstruct memory admission, rescore predictions and apply frozen comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure

from cafl4ds.artifacts import atomic_json
from scripts.audit_recurrence_retention import close, digest, interval, metrics, order_hash, scores

KINDS = ["knn", "linear", "finetune"]


def membership(  # noqa: C901 - independent complete causal reconstruction
    order: list[int],
    raw: np.ndarray,
    ledger: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, float | int]]:
    """Reconstruct all evictions with independent NumPy geometry and the declared private RNG."""
    if raw.shape != (len(order), 96) or not np.isfinite(raw).all() or len(ledger) != len(order) // 32:
        raise ValueError("Invalid descriptor/ledger budget")
    features = raw.astype(np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Invalid reference feature norm")
    features /= norms
    generator = torch.Generator().manual_seed(seed)
    identities: list[int] = []
    retained = np.empty((0, 96))
    similarity = np.empty((0, 0))
    reservoir_ids: list[int] = []
    reservoir_positions: list[int] = []
    positions = {identity: i for i, identity in enumerate(order)}
    diagnostics = []
    for step, row in enumerate(ledger):
        current = order[step * 32 : (step + 1) * 32]
        replay_slots = [] if not identities else torch.randperm(len(identities), generator=generator)[:16].tolist()
        current_slots = list(range(32)) if not replay_slots else torch.randperm(32, generator=generator)[:16].tolist()
        if row["step"] != step + 1 or row["current"] != current:
            raise ValueError("Ledger does not match causal arrival order")
        if row["replay"] != [identities[i] for i in replay_slots]:
            raise ValueError("Replay identities differ from declared private draw")
        if row["selected_current"] != [current[i] for i in current_slots]:
            raise ValueError("Current selection/private RNG schedule differs")
        events = []
        for offset, identity in enumerate(current):
            arrival = step * 32 + offset
            proposed = None if arrival < 256 else int(torch.randint(0, arrival + 1, (1,), generator=generator))
            if arrival < 256:
                reservoir_ids.append(identity)
                reservoir_positions.append(arrival)
            elif proposed is not None and proposed < 256:
                reservoir_ids[proposed] = identity
                reservoir_positions[proposed] = arrival
            n = len(identities)
            candidate = features[arrival]
            pair = np.round(retained @ candidate, 12)
            updated = np.empty((n + 1, n + 1))
            updated[:n, :n] = similarity
            updated[n, :n], updated[:n, n], updated[n, n] = pair, pair, -np.inf
            retained = np.concatenate([retained, candidate[None]])
            identities.append(identity)
            evicted = None
            if len(identities) > 256:
                nearest = updated.max(axis=1)
                victim = int(np.flatnonzero(nearest == nearest.max())[-1])
                evicted = identities.pop(victim)
                keep = np.arange(n + 1) != victim
                retained = retained[keep]
                updated = updated[keep][:, keep]
            similarity = updated
            events.append({"arrival": identity, "evicted": evicted})
        if events != row["events"] or identities != row["memory"]:
            raise ValueError(f"Independent memory admission differs at step{step + 1}")
        if (step + 1) % 144 == 0:
            coordinates = features[reservoir_positions]
            reservoir_similarity = np.round(coordinates @ coordinates.T, 12)
            np.fill_diagonal(reservoir_similarity, -np.inf)
            diagnostics.append(
                {
                    "step": step + 1,
                    "memory_overlap_fraction": len(set(identities) & set(reservoir_ids)) / 256,
                    "minred_mean_nearest_similarity": float(similarity.max(axis=1).mean()),
                    "reservoir_mean_nearest_similarity": float(reservoir_similarity.max(axis=1).mean()),
                    "minred_mean_age_arrivals": float(np.mean([arrival - positions[i] for i in identities])),
                    "reservoir_mean_age_arrivals": float(np.mean([arrival - positions[i] for i in reservoir_ids])),
                }
            )
    return diagnostics


def read_trajectory(
    directory: Path,
    frozen_directory: Path,
    labels: np.ndarray,
    domains: np.ndarray,
    draws: list[int],
) -> dict[str, Any]:
    """Verify all predictions and reconstruct scores before inference."""
    result: dict[str, Any] = json.loads((directory / "result.json").read_text())
    trace = result["trajectory"]
    if [p["step"] for p in trace] != list(range(0, 577, 36)):
        raise ValueError("Incomplete readout grid")
    for point in trace:
        base = frozen_directory if point["step"] == 0 else directory
        kinds = KINDS if point["step"] % 144 == 0 else KINDS[:2]
        with np.load(base / f"predictions_{point['step']:06d}.npz", allow_pickle=False) as predictions:
            for kind in kinds:
                panels = []
                for draw in draws:
                    computed = scores(predictions[f"{draw}_{kind}"], labels, domains)
                    panels.append(computed)
                    for metric, value in computed.items():
                        close(point["panels"][str(draw)][kind][metric], value, "per-panel scoring")
                for metric in ("A", "B", "balanced", "micro"):
                    value = float(np.mean([p[metric] for p in panels]))
                    close(point["scores"][kind][metric], value, "draw-averaged scoring")
                    point["scores"][kind][metric] = value
        if point["step"] % 144 == 0:
            for record in point["fine_tuning_audits"].values():
                if record["steps"] != 100 or record["backbone_delta_l2"] <= 0:
                    raise ValueError("Copied fine-tuning exposure differs")
    return result


def report(out: Path) -> dict[str, Any]:  # noqa: C901 - explicit full-grid and decision checks
    """Audit and report the complete prospective filter study without outcome selection."""
    locked = json.loads((out / "input_lock.json").read_text())
    for name, sha in locked["hashes"].items():
        if digest(Path(name)) != sha:
            raise ValueError(f"Locked input changed: {name}")
    if not json.loads((out / "compatibility.json").read_text())["passed"]:
        raise ValueError("Controls cannot be reused")
    prior = Path(locked["prior"])
    data = json.loads((out / "data_lock.json").read_text())
    seeds = locked["config"]["seeds"]
    draws = locked["config"]["recurrence"]["draw_seeds"]
    with np.load(out / "evaluation_arrays.npz", allow_pickle=False) as arrays:
        query = arrays["query"]
        labels, domains = arrays["labels"][query], arrays["domains"][query]
        if query.tolist() != data["evaluation"]["query_positions"]:
            raise ValueError("Query identities differ")
        for support in data["evaluation"]["supports"].values():
            if set(support) & set(query.tolist()) or np.bincount(arrays["labels"][support]).tolist() != [25, 25, 25]:
                raise ValueError("Support/query allocation differs")
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    geometry: dict[str, list[dict[str, float | int]]] = {}
    for schedule in ("recurrence", "shuffled"):
        rows[schedule] = {policy: [] for policy in ("minred", "reservoir", "current_only")}
        for seed in seeds:
            directory = out / f"{schedule}_minred" / f"seed_{seed}"
            result = read_trajectory(directory, out / "frozen", labels, domains, draws)
            if (
                result["budget"]
                != {
                    "arrivals": 18432,
                    "updates": 576,
                    "optimizer_images": 18432,
                    "reservoir_seen": 18432,
                    "reservoir_size": 256,
                }
                or not result["reference_unchanged"]
                or result["encoder_delta_l2"] <= 0
            ):
                raise ValueError("Model/memory budget differs")
            order = data["orders"][str(seed)][schedule]
            if len(set(order)) != 18432 or result["arrival_order_sha256"] != order_hash(order):
                raise ValueError("Single-pass arrival contract differs")
            ledger = json.loads((directory / "ledger.json").read_text())["steps"]
            raw = np.load(directory / "reference_features.npy", allow_pickle=False)
            geometry[f"{schedule}_{seed}"] = membership(order, raw, ledger, seed)
            losses = json.loads((directory / "losses.json").read_text())["losses"]
            if [p["step"] for p in losses] != list(range(1, 577)) or not np.isfinite(
                [[p["loss"], p["grad_norm"]] for p in losses]
            ).all():
                raise ValueError("Missing/nonfinite optimizer update")
            if any(not (directory / "checkpoints" / f"boundary_{i}.pt").exists() for i in range(1, 5)):
                raise ValueError("Missing boundary checkpoint")
            rows[schedule]["minred"].append(result)
            for policy, suffix in (("reservoir", "replay"), ("current_only", "no_replay")):
                base = prior / f"{schedule}_{suffix}" / f"seed_{seed}"
                rows[schedule][policy].append(read_trajectory(base, prior / "frozen", labels, domains, draws))
    contrasts: dict[str, Any] = {}
    for schedule, policies in rows.items():
        contrasts[schedule] = {}
        for kind in KINDS:
            differences: dict[str, Any] = {}
            for policy in ("reservoir", "current_only"):
                differences[policy] = {}
                for metric in ("final_balanced", "interference", "final_vs_warm", "bwt", "forgetting"):
                    values = [
                        metrics(a["trajectory"], 144, kind)[metric] - metrics(b["trajectory"], 144, kind)[metric]
                        for a, b in zip(policies["minred"], policies[policy], strict=True)
                    ]
                    differences[policy][metric] = interval(values)
            differences["versus_frozen"] = interval(
                [metrics(a["trajectory"], 144, kind)["final_vs_warm"] for a in policies["minred"]]
            )
            contrasts[schedule][kind] = differences
    current_guards = []
    for step, domain in ((144, "A"), (288, "B"), (432, "A"), (576, "B")):
        values = []
        for a, b in zip(rows["recurrence"]["minred"], rows["recurrence"]["reservoir"], strict=True):
            left = next(p for p in a["trajectory"] if p["step"] == step)["scores"]["finetune"][domain]
            right = next(p for p in b["trajectory"] if p["step"] == step)["scores"]["finetune"][domain]
            values.append(left - right)
        current_guards.append({"step": step, "domain": domain, **interval(values)})
    primary = contrasts["recurrence"]["finetune"]
    effect = primary["reservoir"]["final_balanced"]
    gates = {
        "practical_gain_at_least_1pp": effect["mean_pp"] >= 1,
        "positive_lower_bound": effect["ci95_pp"][0] > 0,
        "current_domain_noninferior": all(p["ci95_pp"][0] > -1 for p in current_guards),
        "positive_advantage_over_frozen": primary["versus_frozen"]["ci95_pp"][0] > 0,
    }
    result = {
        "decision": "memory_policy_lead" if all(gates.values()) else "stop_this_memory_recipe",
        "primary_gates": gates,
        "contrasts": contrasts,
        "current_domain_guards": current_guards,
        "advantage_over_current_only_established": primary["current_only"]["final_balanced"]["ci95_pp"][0] > 0,
        "work": {
            "new_collection_updates": 5760,
            "compatibility_updates": 576,
            "disposable_preflight_updates": 8,
            "copied_model_finetune_updates": 13500,
            "collection_reference_image_exposures": 184320,
        },
        "persistent_bytes_scope": "Tensor payloads; excludes Python/container overhead and research trace archives.",
        "memory_geometry_by_arm": geometry,
        "costs_by_arm": {
            f"{schedule}_{policy}": [
                {
                    k: r[k]
                    for k in ("seed", "elapsed_seconds", "encoding_seconds", "admission_seconds", "persistent_bytes")
                }
                for r in policies["minred"]
            ]
            for schedule, policies in rows.items()
            for policy in ("minred",)
        },
    }
    atomic_json(out / "report.json", result)
    lines = [
        "# Static memory admission under SSL",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "Primary: copied-model fine-tuning, 25 labels/class. Five paired seeds, support draws averaged within seed.",
        "Reused warm model, stream images and evaluation; conditional development, not new-population confirmation.",
        "",
        "| Schedule | Readout | MinRed minus comparator | Final balanced gain, pp [95% range] |",
        "|---|---|---|---|",
    ]
    for schedule, readouts in contrasts.items():
        for kind, differences in readouts.items():
            for policy in ("reservoir", "current_only"):
                value = differences[policy]["final_balanced"]
                lines.append(
                    f"| {schedule} | {kind} | {policy} | {value['mean_pp']:+.3f} "
                    f"[{value['ci95_pp'][0]:+.3f}, {value['ci95_pp'][1]:+.3f}] |"
                )
            value = differences["versus_frozen"]
            lines.append(
                f"| {schedule} | {kind} | frozen | {value['mean_pp']:+.3f} "
                f"[{value['ci95_pp'][0]:+.3f}, {value['ci95_pp'][1]:+.3f}] |"
            )
    lines += [
        "",
        f"Frozen primary gates: `{gates}`.",
        "",
        f"Advantage over current-only established: {result['advantage_over_current_only_established']}.",
        "",
        "Image memory and SSL exposures match; reference encoding, geometry and auxiliary bytes are reported.",
        "This tests a static filter, not health control. Greater geometric coverage alone is not success.",
        "Nominal intervals describe five-seed variability on reused data; guardrails are not simultaneous95% bands.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, domain in zip(axes, ("A", "B"), strict=True):
        for policy, runs in rows["recurrence"].items():
            curves = (
                np.array(
                    [
                        [p["scores"]["finetune"][domain] for p in r["trajectory"] if "finetune" in p["scores"]]
                        for r in runs
                    ]
                )
                * 100
            )
            axis.plot([0, 144, 288, 432, 576], curves.mean(axis=0), marker="o", label=policy)
        axis.set(title=f"{domain}: downstream fine-tuning", xlabel="SSL updates", ylabel="Balanced accuracy (%)")
        warm_score = rows["recurrence"]["minred"][0]["trajectory"][0]["scores"]["finetune"][domain]
        axis.axhline(100 * warm_score, color="black", linestyle="--", label="warm frozen")
        axis.legend()
    figure.savefig(out / "trajectories.png", dpi=150)
    atomic_json(
        out / "independent_verification.json",
        {
            "passed": True,
            "new_arms_checked": 10,
            "prior_controls_rescored": 20,
            "arrivals_and_evictions_reconstructed": 184320,
            "input_hashes_unchanged": True,
            "report_sha256": digest(out / "report.json"),
            "scope": "Saved-output scoring and independent membership reconstruction; not model retraining.",
            "artifact_hashes": {
                str(p.relative_to(out)): digest(p)
                for p in out.rglob("*")
                if p.is_file()
                and "source_snapshot" not in p.parts
                and p.suffix in {".json", ".npz", ".npy", ".pt", ".png", ".md"}
                and p.name not in {"status.json", "independent_verification.json"}
            },
        },
    )
    root = Path(__file__).resolve().parents[1]
    for folder in ("phase1", "latest-results"):
        (root / "docs/experiments" / folder / "slow-edge-memory-admission.md").write_text(
            (out / "report.md").read_text() + f"\nArtifacts: `{out.relative_to(root).as_posix()}`.\n", encoding="utf-8"
        )
    return result


def main() -> None:
    """Audit a completed prospective collection without training or tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
