"""Independently audit batch-one forecasts, raw targets and physical continuation."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cafl4ds.artifacts import atomic_json
from cafl4ds.short_buffer import capture
from cafl4ds.single_image_forecast import designs, single_step
from scripts.run_adaptation_experiment import _method
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup
from scripts.run_short_buffer_experiment import digest


def independent_predict(x, y, prefix, online):
    """Direct augmented least squares, without the collector's RLS implementation."""
    center, scale = x[:prefix].mean(0), x[:prefix].std(0)
    scale[scale < 1e-12] = 1
    yc, ys = y[:prefix].mean(0), y[:prefix].std(0)
    ys[ys < 1e-12] = 1
    a = np.column_stack((np.ones(len(x)), (x - center) / scale))
    penalty = np.diag(np.sqrt(np.r_[1e-8, np.full(x.shape[1], 10.0)]))
    predictions = []
    for i in range(prefix, len(x)):
        stop = i if online else prefix
        beta = np.linalg.lstsq(
            np.vstack((a[:stop], penalty)),
            np.vstack(((y[:stop] - yc) / ys, np.zeros((len(penalty), y.shape[1])))),
            rcond=None,
        )[0]
        predictions.append(yc + a[i] @ beta * ys)
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    lock = json.loads((out / "input_lock.json").read_text())
    archived_verifier = None
    for path, expected in lock["hashes"].items():
        if digest(Path(path)) != expected:
            # Audit-only CE summation repair after collection. Retain and verify
            # the original locked verifier; every scientific input stays strict.
            if Path(path).resolve() != Path(__file__).resolve():
                raise ValueError(f"Locked input changed: {path}")
            snapshot = out / "source_snapshot" / Path(path).relative_to(Path(__file__).resolve().parents[1])
            if not snapshot.exists() or digest(snapshot) != expected:
                raise ValueError("Original locked verifier snapshot differs")
            archived_verifier = {
                "snapshot": str(snapshot),
                "locked_sha256": expected,
                "current_sha256": digest(Path(__file__)),
                "repair": "Reconstruct float32 CE one state at a time to preserve the original summation order",
            }
    prefix = lock["settings"]["prefix_images"]
    panel = np.load(out / "panel.npz", allow_pickle=False)
    labels, domains = panel["query_labels"], panel["query_domains"]
    maximum = 0.0
    transitions = 0
    physical = 0
    raw_scores = 0
    initial = None
    if args.physical:
        setup()
        config = OmegaConf.create(lock["learner_config"])
        initial = _method(config).cuda()
    paths = sorted((out / "trajectories").glob("*/seed_*"))
    for directory in paths:
        a = np.load(directory / "series.npz", allow_pickle=False)
        f = np.load(directory / "forecasts.npz", allow_pickle=False)
        obs = json.loads((directory / "observations.json").read_text())
        metrics = a["metrics"].tolist()
        y = a["after"] - a["before"]
        if len(y) != 192 or not np.array_equal(y[prefix:], f["targets"]):
            raise ValueError("Wrong target/transition count")
        # A next-state target must be the next row's measured current state, apart
        # from the explicitly current-image-dependent reconstruction contrast.
        if not np.array_equal(a["after"][:-1, :-1], a["before"][1:, :-1]):
            raise ValueError("Time series continuity failed")
        inputs = designs(a["before"], a["after"], a["image"], metrics, a["training"])
        for name, x in inputs.items():
            for online in (False, True):
                model = ("online_" if online else "offline_") + name
                predicted = independent_predict(x, y, prefix, online)
                error = float(np.max(np.abs(predicted - f[model])))
                maximum = max(maximum, error)
                if not np.allclose(predicted, f[model], rtol=2e-7, atol=1e-8):
                    raise ValueError(f"Independent regression mismatch: {directory}/{model}: {error}")
        with np.load(directory / "readouts.npz", allow_pickle=False) as r:
            for reader in r.files:
                probabilities = r[reader]
                for domain, mask in (
                    ("balanced", np.ones(len(labels), bool)),
                    ("day", domains == 0),
                    ("night", domains == 1),
                ):
                    correct = probabilities.argmax(-1)[:, mask] == labels[mask]
                    accuracies = 100 * correct.mean(1)
                    ce = np.array(
                        [
                            (-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)))[mask].mean()
                            for p in probabilities
                        ]
                    )
                    for kind, values in (("accuracy", accuracies), ("ce", ce)):
                        j = metrics.index(f"{reader}_{kind}_{domain}")
                        actual = np.r_[a["before"][0, j], a["after"][:, j]]
                        if not np.allclose(values, actual, rtol=0, atol=1e-12):
                            raise ValueError("Raw readout score reconstruction failed")
                        raw_scores += len(values)
        if args.physical:
            seed = int(directory.name.split("_")[-1])
            model = copy.deepcopy(initial)
            torch.manual_seed(seed)
            optimizer = instantiate(config.optim, params=[p for p in model.parameters() if p.requires_grad])
            stream = lock["selected"][directory.parent.name]["frames"]
            intake_path = next(
                Path(p)
                for p in lock["hashes"]
                if p.endswith("20261013-frames\\intake.json") or p.endswith("20261013-frames/intake.json")
            )
            intake = json.loads(intake_path.read_text())
            clips = {c["clip_id"]: c for c in intake["clips"]}
            frames = {k: np.load(clips[k]["frames_path"], mmap_mode="r") for k in {s["clip_id"] for s in stream}}
            if fingerprint(capture(model, optimizer)) != obs["initial_fingerprint"]:
                raise ValueError("Initial state replay mismatch")
            for i, row in enumerate(stream):
                image = (
                    torch.from_numpy(frames[row["clip_id"]][row["frame"]].copy()).permute(2, 0, 1).float()[None] / 255
                )
                train = single_step(model, optimizer, image.cuda())
                if not np.array_equal(train, a["training"][i]):
                    raise ValueError("Exact single-image loss/gradient replay failed")
                if (
                    str(i + 1) in obs["checkpoints"]
                    and fingerprint(capture(model, optimizer)) != obs["checkpoints"][str(i + 1)]
                ):
                    raise ValueError("Complete learner checkpoint replay failed")
                physical += 1
            saved = torch.load(directory / "final_state.pt", weights_only=True, map_location="cuda")
            if fingerprint(capture(model, optimizer)) != fingerprint(saved):
                raise ValueError("Final complete learner differs")
        transitions += len(y)
        print(
            json.dumps({"verified": str(directory), "transitions": transitions, "physical_replay_updates": physical}),
            flush=True,
        )
    # Independently reconstruct every saved aggregate's point errors.
    summary_path = out / "summary.json"
    summary_errors = 0
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for j, metric in enumerate(metrics):
            for name, score in summary["metrics"][metric]["models"].items():
                errors = []
                for directory in paths:
                    with np.load(directory / "forecasts.npz") as f:
                        errors.extend((f[name][:, j] - f["targets"][:, j]).tolist())
                e = np.array(errors)
                if not np.isclose(score["rmse"], np.sqrt(np.mean(e**2)), rtol=1e-12) or not np.isclose(
                    score["mae"], np.mean(abs(e)), rtol=1e-12
                ):
                    raise ValueError("Aggregate RMSE/MAE reconstruction failed")
                summary_errors += 1
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "trajectories": len(paths),
            "scientific_transitions": transitions,
            "physical_replay_updates": physical,
            "raw_readout_values_reconstructed": raw_scores,
            "independent_forecast_max_absolute_difference": maximum,
            "aggregate_error_pairs_checked": summary_errors,
            "audit_only_source_repair": archived_verifier,
            "prefix_scaling_only": True,
            "online_fit_ends_before_target": True,
            "limitations": "No independent reconstruction of every fine-tuning fit; preflight exact repeated fits and observer invariance checks cover collection mechanics.",
        },
    )


if __name__ == "__main__":
    main()
