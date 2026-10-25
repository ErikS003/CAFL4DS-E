"""Exercise complete independent scoring and frozen success/failure gates on synthetic artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cafl4ds.artifacts import atomic_json
from scripts import report_update_filtering as audit
from scripts.audit_recurrence_retention import scores


def test_complete_report_and_timing_gate(  # noqa: C901 - complete synthetic study fixture
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical accurate filtered models qualify only when the declared timing gate also passes."""
    out, prior = tmp_path / "outputs/run", tmp_path / "prior"
    out.mkdir(parents=True)
    for folder in ("phase1", "latest-results"):
        (tmp_path / "docs/experiments" / folder).mkdir(parents=True)
    monkeypatch.setattr(audit, "__file__", str(tmp_path / "scripts/report_update_filtering.py"))
    seeds = [5003, 5009, 5011, 5021, 5023]
    draws = [51011, 51017, 51023]
    query = np.arange(6)
    labels = np.concatenate([np.tile(np.arange(3), 2), np.tile(np.arange(3), 25)])
    domains = np.concatenate([np.repeat(np.arange(2), 3), np.zeros(75, dtype=int)])
    supports = {str(draw): list(range(6, 81)) for draw in draws}
    order = list(range(18432))
    atomic_json(
        out / "input_lock.json",
        {"hashes": {}, "prior": str(prior), "config": {"seeds": seeds, "recurrence": {"draw_seeds": draws}}},
    )
    atomic_json(
        out / "data_lock.json",
        {
            "evaluation": {"query_positions": query.tolist(), "supports": supports},
            "orders": {str(seed): dict.fromkeys(("recurrence", "shuffled"), order) for seed in seeds},
        },
    )
    np.savez(out / "evaluation_arrays.npz", query=query, labels=labels, domains=domains)
    atomic_json(out / "preflight.json", {"passed": True})
    manifests = {
        str(seed): {p: audit.independent_mask(p, seed) for p in ("fixed_half", "random_half")} for seed in seeds
    }
    atomic_json(out / "mask_manifest.json", manifests)
    trace = []
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for step in range(0, 577, 36):
        kinds = audit.KINDS if step % 144 == 0 else audit.KINDS[:2]
        predicted = np.zeros(6, dtype=int) if step == 0 else labels[query]
        computed = scores(predicted, labels[query], domains[query])
        point = {
            "step": step,
            "panels": {str(d): dict.fromkeys(kinds, computed) for d in draws},
            "scores": dict.fromkeys(kinds, computed),
            "fine_tuning_audits": {str(d): {"steps": 100, "backbone_delta_l2": 1} for d in draws},
        }
        trace.append(point)
        predictions[step] = {f"{d}_{k}": predicted for d in draws for k in kinds}
    for frozen in (out / "frozen", prior / "frozen"):
        frozen.mkdir(parents=True)
        np.savez(frozen / "predictions_000000.npz", **predictions[0])

    def artifacts(directory: Path, seed: int, policy: str, online: bool) -> dict[str, Any]:
        directory.mkdir(parents=True)
        mask = audit.independent_mask(policy, seed)
        count = sum(mask)
        result = {
            "seed": seed,
            "trajectory": copy.deepcopy(trace),
            "encoder_delta_l2": 1,
            "arrival_order_sha256": audit.order_hash(order),
            "budget": {
                "arrivals": 18432,
                "updates": count,
                "optimizer_images": count * 32,
                "reservoir_seen": 0,
                "reservoir_size": 0,
            },
            "online_seconds": 576 if policy == "always" else 288,
            "block_online_seconds": [144 if policy == "always" else 72] * 4,
            "elapsed_seconds": 1000,
        }
        atomic_json(directory / "result.json", result)
        for step in range(36, 577, 36):
            np.savez(directory / f"predictions_{step:06d}.npz", **predictions[step])
        if online:
            atomic_json(directory / "mask.json", {"seed": seed, "policy": policy, "accepted": mask})
            ledger, losses = [], []
            updates = 0
            for index, accepted in enumerate(mask):
                updates += int(accepted)
                ledger.append(
                    {
                        "step": index + 1,
                        "current": order[index * 32 : (index + 1) * 32],
                        "accepted": accepted,
                        "optimizer_steps": updates,
                        "optimizer_images": updates * 32,
                        "online_seconds": 1 if policy == "always" else 0.5,
                    }
                )
                if accepted:
                    losses.append({"step": index + 1, "optimizer_step": updates, "loss": 1, "grad_norm": 1})
            atomic_json(directory / "ledger.json", {"steps": ledger})
            atomic_json(directory / "losses.json", {"losses": losses})
            (directory / "checkpoints").mkdir()
            for i in range(1, 5):
                torch.save({"encoder": torch.ones(1)}, directory / "checkpoints" / f"boundary_{i}.pt")
            (directory / "decision_states").mkdir()
            for step in audit.SNAPSHOTS:
                torch.save(
                    {
                        "arrival_step": step,
                        "optimizer_steps": sum(mask[:step]),
                        "mask": mask,
                        "arrival_order_sha256": audit.order_hash(order),
                        "torch_rng": torch.get_rng_state(),
                        "model": {"encoder": torch.ones(1)},
                        "optimizer": {"state": {0: {"step": torch.tensor(sum(mask[:step]))}}},
                    },
                    directory / "decision_states" / f"arrival_{step:06d}.pt",
                )
        return result

    for schedule in ("recurrence", "shuffled"):
        for seed in seeds:
            artifacts(prior / f"{schedule}_no_replay/seed_{seed}", seed, "always", False)
            for policy in ("fixed_half", "random_half"):
                artifacts(out / f"{schedule}_{policy}/seed_{seed}", seed, policy, True)
    artifacts(out / "compatibility/recurrence_always/seed_5003", 5003, "always", True)
    atomic_json(out / "compatibility.json", {"passed": True, "optimizer_updates": 576, "online_seconds": 576})
    result = audit.report(out)
    assert result["decision"] == "half_rate_efficiency_lead"
    assert not result["fixed_timing_advantage_established"]
    assert (out / "independent_verification.json").is_file()
    directory = out / "recurrence_fixed_half/seed_5003"
    # Preserve scores and accepted-update counts; only remove the measured cost savings.
    record = json.loads((directory / "result.json").read_text())
    ledger = json.loads((directory / "ledger.json").read_text())
    for row in ledger["steps"]:
        row["online_seconds"] = 0.9
    record["online_seconds"] = sum(row["online_seconds"] for row in ledger["steps"])
    record["block_online_seconds"] = [
        sum(row["online_seconds"] for row in ledger["steps"][i : i + 144]) for i in range(0, 576, 144)
    ]
    atomic_json(directory / "ledger.json", ledger)
    atomic_json(directory / "result.json", record)
    result = audit.report(out)
    assert result["decision"] == "no_primary_efficiency_lead"
    assert result["primary_gates"]["final_noninferior"]
    assert not result["primary_gates"]["online_time_reduced_at_least_25pct"]
