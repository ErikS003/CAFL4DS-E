"""Independent virtual-update and grouped-statistics verification for anchor risk."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import t

from cafl4ds.artifacts import atomic_json
from cafl4ds.retention_anchor import private_loss
from cafl4ds.short_buffer import capture, restore
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_short_buffer_experiment import digest
from scripts.run_retention_anchor_signal import gather, setup


def main() -> None:
    """Check full state effects and statistics with separately reconstructed loops."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    root = Path(__file__).resolve().parents[1]
    prior = root / "outputs/budget-reservation/20261013-gpu"
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    lock = read(out / "input_lock.json")
    assert all(digest(Path(p)) == h for p, h in lock["hashes"].items())
    observations = read(out / "observations.json")
    assert observations["complete"] and len(observations["records"]) == 54
    intake = read(root / "outputs/coverage-admission/20261013-frames/intake.json")
    streams = {s["name"]: s for s in intake["streams"]}
    clips = {c["clip_id"]: c for c in intake["clips"]}
    config = OmegaConf.create(lock["config"])
    setup()
    torch.manual_seed(11)
    initial = _method(config).cuda()
    for o in observations["records"]:
        generator = torch.Generator().manual_seed(o["seed"] + 1117)
        memory = list(range(64))
        for k in range(64, o["point"]):
            j = int(torch.randint(0, k + 1, (1,), generator=generator))
            if j < 64:
                memory[j] = k
        assert sorted(memory) == o["anchor_indices"] and max(memory) < o["point"]
        assert o["incoming_indices"] == list(range(o["point"], o["point"] + 32))
        anchors = gather(streams[o["stream"]], clips, sorted(memory))
        incoming = gather(streams[o["stream"]], clips, o["incoming_indices"])
        parent = torch.load(
            prior / "parents" / o["stream"] / o["checkpoint"] / f"seed_{o['seed']}/state.pt",
            map_location="cpu",
            weights_only=True,
        )
        model, optimizer = restore(initial, config.optim, parent)
        marker = fingerprint(capture(model, optimizer))
        assert marker == o["parent_fingerprint"]
        before = private_loss(model, anchors, o["mask_seed"])
        assert fingerprint(capture(model, optimizer)) == marker
        model.train()
        optimizer.zero_grad()
        loss = model.training_step(incoming.cuda())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assert [float(loss.detach()), float(norm)] == o["preview_loss_gradient"]
        assert fingerprint(capture(model, optimizer)) == o["virtual_fingerprint"]
        after = private_loss(model, anchors, o["mask_seed"])
        assert before == o["anchor_before"] and after == o["anchor_after"]
        assert (after - before) / max(before, 1e-12) == o["relative_anchor_change"]
        assert fingerprint(capture(model, optimizer)) == o["virtual_fingerprint"]
    summary = read(out / "summary.json")
    points = summary["points"]
    rows = read(prior / "collection.json")["records"]
    lookup = {(r["stream"], r["checkpoint"], r["seed"], r["arm"]): r for r in rows}
    with np.load(root / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz") as a:
        labels, domains = a["labels"][a["query"]], a["domains"][a["query"]]
    for p in points:
        obs = [o for o in observations["records"] if o["stream"] == p["stream"] and o["checkpoint"] == p["checkpoint"]]
        assert len(obs) == 3
        for feature in ("incoming_loss", "relative_anchor_change"):
            assert np.isclose(p[feature], np.mean([o[feature] for o in obs]), atol=1e-10, rtol=0)
        assert p["updates"] == obs[0]["point"] / 64
        for horizon in ("immediate", "terminal"):
            effects = []
            for mask in (domains == 0, domains == 1):
                changes = []
                for seed in (5003, 5009, 5011):
                    means = []
                    for arm in ("spend", "hold"):
                        with np.load(
                            prior / lookup[p["stream"], p["checkpoint"], seed, arm][horizon] / "predictions.npz"
                        ) as pred:
                            means.append(
                                np.mean(
                                    [
                                        100 * np.mean(pred[f"{draw}_finetune"][mask] == labels[mask])
                                        for draw in (51011, 51017, 51023)
                                    ]
                                )
                            )
                    changes.append(means[0] - means[1])
                effects.append(np.mean(changes))
            assert np.isclose(p[f"{horizon}_damage"], -min(effects), atol=1e-10, rtol=0)
            assert np.isclose(p[f"{horizon}_balanced_effect"], np.mean(effects), atol=1e-10, rtol=0)
    feature_sets = {
        "mean": [],
        "age": ["updates"],
        "loss": ["incoming_loss"],
        "age_loss": ["updates", "incoming_loss"],
        "anchor": ["relative_anchor_change"],
        "age_loss_anchor": ["updates", "incoming_loss", "relative_anchor_change"],
    }
    groups = sorted({p["stream"] for p in points})
    assert len(groups) == 6 and len(points) == 18
    errors, coefficients, checks = {}, {}, 0
    for horizon in ("immediate", "terminal"):
        y = np.array([p[f"{horizon}_damage"] for p in points])
        errors[horizon] = {}
        for name, features in feature_sets.items():
            x = np.array([[p[f] for f in features] for p in points]).reshape(18, len(features))
            prediction = np.zeros(18)
            for group in groups:
                test = np.array([p["stream"] == group for p in points])
                center = np.mean(x[~test], axis=0)
                scale = np.std(x[~test], axis=0)
                scale[scale == 0] = 1
                z = (x - center) / scale
                design = np.column_stack((z[~test], np.ones(15)))
                penalty = np.eye(len(features) + 1)
                penalty[-1, -1] = 0
                beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[~test])
                prediction[test] = np.column_stack((z[test], np.ones(3))) @ beta
                if horizon == "terminal" and name == "age_loss_anchor":
                    coefficients[group] = beta[-2]
            reported = summary["forecasts"][horizon][name]
            assert np.allclose(prediction, reported["predictions"], atol=1e-9, rtol=0)
            error = (prediction - y) ** 2
            errors[horizon][name] = {g: error[[p["stream"] == g for p in points]].mean() for g in groups}
            for g in groups:
                assert np.isclose(errors[horizon][name][g], reported["source_mse"][g], atol=1e-9, rtol=0)
            assert np.isclose(error.mean(), reported["mse"], atol=1e-9, rtol=0)
            checks += 1
    passed = True
    for baseline in ("age_loss", "mean"):
        delta = np.array([errors["terminal"][baseline][g] - errors["terminal"]["age_loss_anchor"][g] for g in groups])
        width = t.ppf(0.9875, 5) * delta.std(ddof=1) / np.sqrt(6)
        reduction = 1 - np.mean(list(errors["terminal"]["age_loss_anchor"].values())) / np.mean(
            list(errors["terminal"][baseline].values())
        )
        reported = summary["primary_contrasts"][baseline]
        assert np.allclose(reported["adjusted_ci95"], [delta.mean() - width, delta.mean() + width], atol=1e-9, rtol=0)
        assert np.isclose(reduction, reported["reduction_fraction"], atol=1e-9, rtol=0)
        passed &= reduction >= 0.1 and delta.mean() - width > 0
    sign = sum(v > 0 for v in coefficients.values()) >= 5
    support = any(p["terminal_damage"] >= 1 for p in points) and any(p["terminal_damage"] <= 0 for p in points)
    assert summary["signal_qualified"] == (passed and sign and support)
    result = {
        "passed": True,
        "independent_preview_updates": 54,
        "causal_reservoirs": 54,
        "outcome_points": 18,
        "grouped_models_checked": checks,
        "primary_contrasts_checked": 2,
    }
    atomic_json(out / "independent-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
