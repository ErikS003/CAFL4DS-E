"""Audit regression checks using a small synthetic collection, without model training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cafl4ds.artifacts import atomic_json
from cafl4ds.recurrence import prediction_scores
from scripts.decode_bdd_temporal import sha256
from scripts.report_bdd_temporal_feasibility import report
from scripts.run_paired_update_response import fingerprint


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """Create auditable predictions, clocks and snapshots for twenty tiny arms."""
    query_labels = np.tile(np.repeat(np.arange(3), 100), 2)
    query_domains = np.repeat(np.arange(2), 300)
    query = np.arange(600)
    draws = [51011, 51017, 51023]
    labels = np.concatenate([query_labels, np.tile(np.repeat(np.arange(3), 25), 4)])
    np.savez_compressed(tmp_path / "evaluation_arrays.npz", labels=labels,
                        domains=np.concatenate([query_domains, np.zeros(300, dtype=int)]), query=query)
    supports = {str(draw): list(range(600 + i * 75, 675 + i * 75)) for i, draw in enumerate(draws)}
    atomic_json(tmp_path / "data_lock.json", {"arrivals": 6144,
                "evaluation": {"query_positions": query.tolist(), "supports": supports}})
    atomic_json(tmp_path / "orders.json", {"chronological": list(range(6144)), "shuffled": list(reversed(range(6144)))})
    seeds = [5003, 5009, 5011, 5021, 5023]
    atomic_json(tmp_path / "input_lock.json", {"hashes": {}, "seeds": seeds})
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    atomic_json(frozen / "result.json", {"scores": {"finetune": {"balanced": .5}}})
    metrics = prediction_scores(torch.from_numpy(query_labels), torch.from_numpy(query_labels),
                                torch.from_numpy(query_domains))
    point = {"scores": dict.fromkeys(("finetune", "linear", "knn"), metrics),
             "panels": {str(draw): dict.fromkeys(("finetune", "linear", "knn"), metrics) for draw in draws}}
    for order in ("chronological", "shuffled"):
        for seed in seeds:
            for policy in ("full", "half"):
                directory = tmp_path / order / policy / f"seed_{seed}"
                directory.mkdir(parents=True)
                count = 192 if policy == "full" else 96
                initial = {"model": {"weight": torch.tensor([1.0])}, "optimizer": {},
                           "torch_rng": torch.tensor([seed]), "training": True,
                           "arrivals": 0, "optimizer_updates": 0}
                torch.save(initial, directory / "initial_state.pt")
                torch.save({"arrivals": 6144, "optimizer_updates": count,
                            "optimizer": {"state": {0: {"step": torch.tensor(count)}}}}, directory / "final_state.pt")
                records = [{"arrival_batch": step, "accepted": policy == "full" or step % 2 == 0,
                            "loss_gradient": [1.0, .1] if policy == "full" or step % 2 == 0 else None,
                            "online_seconds": 1.0 if policy == "full" or step % 2 == 0 else 0.0}
                           for step in range(1, 193)]
                atomic_json(directory / "arrivals.json", {"records": records})
                atomic_json(directory / "result.json", {"outcome": point, "optimizer_updates": count,
                            "online_seconds": float(count), "initial_fingerprint": fingerprint(initial),
                            "initial_file_sha256": sha256(directory / "initial_state.pt"),
                            "readout_preserved_learning_state": True})
                np.savez_compressed(directory / "predictions_000192.npz", **{
                    f"{draw}_{kind}": query_labels for draw in draws for kind in ("finetune", "linear", "knn")})
    return tmp_path


def test_synthetic_qualified_static_point_remains_not_a_controller(collection: Path) -> None:
    """A quality-preserving cheaper point must not turn into a controller claim."""
    result = report(collection)
    assert result["decision"] == "useful_static_half_rate_point"
    assert result["controller_qualified"] is False
    verification = json.loads((collection / "independent_verification.json").read_text())
    assert verification["arms_checked"] == 20
    assert verification["optimizer_updates"] == 2880


def test_corrupted_prediction_rejected(collection: Path) -> None:
    """Raw predictions, rather than stored summaries, determine audit success."""
    path = collection / "chronological/full/seed_5003/predictions_000192.npz"
    with np.load(path) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    values["51011_finetune"][0] = 1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="raw predictions"):
        report(collection)


def test_wrong_optimizer_budget_rejected(collection: Path) -> None:
    """Counters in the serialized optimizer must agree with the declared update count."""
    path = collection / "chronological/half/seed_5003/final_state.pt"
    value = torch.load(path, weights_only=False)
    value["optimizer"]["state"][0]["step"] = torch.tensor(97)
    torch.save(value, path)
    with pytest.raises(ValueError, match="moment step"):
        report(collection)
