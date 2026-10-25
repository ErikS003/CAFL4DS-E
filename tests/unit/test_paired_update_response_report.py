"""Exercise all 80 audited terminal outcomes and adjusted preference classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cafl4ds.artifacts import atomic_json
from scripts import report_paired_update_response as audit
from scripts.audit_recurrence_retention import scores


def test_complete_report_classifies_and_rejects_corrupt_binding(  # noqa: C901 - full synthetic scientific grid
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One safe phase and one harmful phase support a conditional diagnostic, never a fitted policy."""
    out, parent = tmp_path / "outputs/run", tmp_path / "parent"
    out.mkdir(parents=True)
    for folder in ("phase1", "latest-results"):
        (tmp_path / "docs/experiments" / folder).mkdir(parents=True)
    monkeypatch.setattr(audit, "__file__", str(tmp_path / "scripts/report_paired_update_response.py"))
    seeds, draws = [5003, 5009, 5011, 5021, 5023], [51011, 51017, 51023]
    order = list(range(18432))
    query = np.arange(6)
    labels = np.concatenate([np.tile(np.arange(3), 2), np.tile(np.arange(3), 25)])
    domains = np.concatenate([np.repeat(np.arange(2), 3), np.zeros(75, dtype=int)])
    config = {"seeds": seeds, "recurrence": {"draw_seeds": draws}}
    atomic_json(out / "input_lock.json", {"hashes": {}, "parent": str(parent), "config": config})
    atomic_json(
        out / "data_lock.json",
        {
            "evaluation": {"query_positions": query.tolist(), "supports": {str(d): list(range(6, 81)) for d in draws}},
            "orders": {str(s): {"recurrence": order} for s in seeds},
        },
    )
    np.savez(out / "evaluation_arrays.npz", query=query, labels=labels, domains=domains)
    atomic_json(
        out / "preflight.json", {"passed": True, "restored_continuation_exact": True, "continuation_updates": 36}
    )

    def state(offset: int, updates: int) -> dict[str, Any]:
        return {
            "model": {"encoder": torch.ones(1)},
            "optimizer": {"state": {0: {"step": torch.tensor(updates)}}},
            "torch_rng": torch.get_rng_state(),
            "model_training": True,
            "arrival_step": offset,
            "optimizer_steps": updates,
        }

    def window(offset: int, policy: str, horizon: int = 36) -> dict[str, Any]:
        ledger, losses = [], []
        updates = 0
        for arrival in range(offset + 1, offset + horizon + 1):
            accepted = policy == "full" or arrival % 2 == 0
            updates += int(accepted)
            ledger.append(
                {
                    "step": arrival,
                    "current": order[(arrival - 1) * 32 : arrival * 32],
                    "accepted": accepted,
                    "branch_updates": updates,
                    "optimizer_steps": offset // 2 + updates,
                    "online_seconds": 1 if policy == "full" else 0.5,
                }
            )
            if accepted:
                losses.append({"step": arrival, "optimizer_step": offset // 2 + updates, "loss": 1, "grad_norm": 1})
        return {
            "ledger": ledger,
            "losses": losses,
            "updates": updates,
            "optimizer_images": updates * 32,
            "arrivals": horizon * 32,
            "online_seconds": sum(p["online_seconds"] for p in ledger),
        }

    def terminal(directory: Path, step: int, harmful: bool) -> dict[str, Any]:
        predicted = labels[query].copy()
        if harmful:
            predicted[1] = 0
        computed = scores(predicted, labels[query], domains[query])
        point = {
            "step": step,
            "scores": dict.fromkeys(audit.KINDS, computed),
            "panels": {str(d): dict.fromkeys(audit.KINDS, computed) for d in draws},
            "fine_tuning_audits": {str(d): {"steps": 100, "backbone_delta_l2": 1} for d in draws},
        }
        np.savez(
            directory / f"predictions_{step:06d}.npz", **{f"{d}_{k}": predicted for d in draws for k in audit.KINDS}
        )
        return point

    for seed in seeds:
        first_dir = parent / f"recurrence_fixed_half/seed_{seed}"
        (first_dir / "decision_states").mkdir(parents=True)
        for offset in audit.OFFSETS:
            saved = {**state(offset, offset // 2), "arrival_order_sha256": audit.order_hash(order)}
            parent_file = first_dir / f"decision_states/arrival_{offset:06d}.pt"
            torch.save(saved, parent_file)
            for policy in ("full", "half"):
                directory = out / f"seed_{seed}/arrival_{offset:06d}" / policy
                directory.mkdir(parents=True)
                work = window(offset, policy)
                final = state(offset + 36, offset // 2 + work["updates"])
                torch.save(final, directory / "final_state.pt")
                result = {
                    "seed": seed,
                    "offset": offset,
                    "policy": policy,
                    "parent_sha256": audit.digest(parent_file),
                    "restored_fingerprint": audit.state_hash({k: saved[k] for k in state(offset, offset // 2)}),
                    "arrival_order_sha256": audit.order_hash(order),
                    "window": work,
                    "terminal": terminal(directory, offset + 36, policy == "half" and offset == 108),
                    "final_state_fingerprint": audit.state_hash(final),
                    "parent_unchanged": True,
                    "restoration_seconds": 0.001,
                    "elapsed_seconds": 100,
                }
                atomic_json(directory / "result.json", result)
    first_dir = parent / "recurrence_fixed_half/seed_5003"
    work = window(36, "half", 72)
    atomic_json(first_dir / "losses.json", {"losses": work["losses"]})
    compatibility = out / "compatibility"
    compatibility.mkdir()
    point = terminal(compatibility, 108, False)
    np.savez(
        first_dir / "predictions_000108.npz", **{f"{d}_{k}": labels[query] for d in draws for k in audit.KINDS[:2]}
    )
    atomic_json(compatibility / "result.json", {"window": work, "terminal": point})
    torch.save(state(108, 54), compatibility / "final_state.pt")
    result = audit.report(out)
    assert result["decision"] == "candidate_state_dependent_opportunity"
    assert 36 in result["half_preferred_offsets"]
    assert result["full_required_offsets"] == [108]
    path = out / "seed_5003/arrival_000036/half/result.json"
    corrupted = json.loads(path.read_text())
    corrupted["restored_fingerprint"] = "wrong-parent"
    atomic_json(path, corrupted)
    with pytest.raises(ValueError, match="Restored"):
        audit.report(out)
