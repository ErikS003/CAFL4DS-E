"""Run a cached, regression-only measured-health and forecast-health utility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_regression_gate import crossfit, observations, response, summarize
from cafl4ds.prediction_benchmark import health_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/health-warning/20261013-gpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_readout(path):
    directory = Path(path)
    if (directory / "predictions.npz").exists():
        return directory
    alias = read(directory / "readout-alias.json")
    candidate = Path(alias["directory"])
    if not candidate.is_absolute():
        candidate = ROOT / "outputs/regime-windows/20261013-gpu" / candidate
    if not (candidate / "predictions.npz").exists():
        raise ValueError("Cannot resolve cached readout")
    return candidate


def collect(out):
    """Freeze all used sources before constructing the new retrospective targets."""
    if not all(read(SOURCE / n)["passed"] for n in ("verification.json", "independent-verification.json")):
        raise ValueError("Original numerical and replay audits required")
    collection = read(SOURCE / "collection.json")
    rows = [r for r in collection["records"] if r["role"] == "six_stream"]
    if not collection["complete"] or len(rows) != 54 or not all(r["exact_replay"] for r in rows):
        raise ValueError("Complete cached six-history cohort required")
    if len({r["stream"] for r in rows}) != 6 or {r["seed"] for r in rows} != {5003, 5009, 5011}:
        raise ValueError("Frozen source/seed cohort required")
    arrays = ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz"
    scientific = [
        Path(__file__).resolve(),
        ROOT / "scripts/verify_health_regression_gate.py",
        ROOT / "scripts/verify_prediction_pilots.py",
        ROOT / "cafl4ds/health_regression_gate.py",
        ROOT / "cafl4ds/prediction_benchmark.py",
        ROOT / "cafl4ds/artifacts.py",
        ROOT / "cafl4ds/jsonio.py",
        ROOT / "tests/unit/test_health_regression_gate.py",
        ROOT / "docs/project-plan/health-regression-gate-frozen.md",
    ]
    paths = scientific + [
        SOURCE / n for n in ("collection.json", "verification.json", "independent-verification.json", "input_lock.json")
    ]
    paths += [arrays]
    for row in rows:
        paths.append(Path(row["embeddings_path"]))
        for landmark in (8, 16):
            directory = resolve_readout(row["landmarks"][str(landmark)]["readout"])
            paths.extend(directory / n for n in ("predictions.npz", "readout.json"))
    lock = {
        "hashes": {str(p): digest(p) for p in paths},
        "role": "retrospective_reused_development",
        "decision_update": 8,
        "target_update": 16,
        "health_reference": "fixed local continuation parent at update zero",
        "new_ssl_updates": 0,
        "new_classifier_fits": 0,
        "health_gradients_or_virtual_updates": False,
    }
    atomic_json(out / "input_lock.json", lock)
    for p in scientific:
        destination = out / "source_snapshot" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    with np.load(arrays) as a:
        labels = a["labels"][a["query"]]
        domains = a["domains"][a["query"]]
    data = {
        "identities": [],
        "groups": [],
        "x": [],
        "health_response": [],
        "utility": [],
        "health": [],
        "panel_utility": [],
        "reader_utility": [],
    }
    for row in rows:
        with np.load(row["embeddings_path"]) as a:
            reference = a["at_0"]
            health = [list(health_values(a[f"at_{t}"], reference).values()) for t in (0, 4, 8, 16)]
        h = np.array(health)
        for t, vector in zip((0, 4, 8, 16), health, strict=True):
            if abs(vector[3] - row["landmarks"][str(t)]["geometry"]["cka_drift"]) > 1e-6:
                raise ValueError("Local-parent reference differs from original geometry")
        quality = {}
        for t in (8, 16):
            directory = resolve_readout(row["landmarks"][str(t)]["readout"])
            quality[t] = {}
            with np.load(directory / "predictions.npz") as p:
                for reader in ("finetune", "linear", "knn"):
                    scores = []
                    for draw in (51011, 51017, 51023):
                        correct = p[f"{draw}_{reader}"] == labels
                        scores.append(
                            [
                                100 * float(correct[mask].mean())
                                for mask in (np.ones(len(labels), bool), domains == 0, domains == 1)
                            ]
                        )
                    quality[t][reader] = np.array(scores)
                    old = row["landmarks"][str(t)]["scores"][reader]
                    if not np.allclose(
                        np.mean(scores, axis=0), [old[k] for k in ("balanced", "A", "B")], atol=1e-9, rtol=1e-9
                    ):
                        raise ValueError("Raw readout score reconstruction differs")
        panel = quality[16]["finetune"] - quality[8]["finetune"]
        data["identities"].append({k: row[k] for k in ("stream", "checkpoint", "seed")})
        data["groups"].append(row["stream"])
        data["x"].append(observations(h[:3], row["parent_updates"], [r[0] for r in row["trace"][:8]]).tolist())
        data["health_response"].append(response(h[2], h[3]).tolist())
        data["utility"].append(panel.mean(0).tolist())
        data["panel_utility"].append(panel.tolist())
        data["reader_utility"].append(
            {reader: (quality[16][reader] - quality[8][reader]).mean(0).tolist() for reader in ("linear", "knn")}
        )
        data["health"].append(health)
    if len(set((i["stream"], i["checkpoint"], i["seed"]) for i in data["identities"])) != 54:
        raise ValueError("Duplicate cohort identity")
    if not all(np.isfinite(np.array(data[k])).all() for k in ("x", "health_response", "utility")):
        raise ValueError("Nonfinite observations")
    atomic_json(out / "data.json", data)
    return data, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if out.exists():
        raise ValueError("Use a new directory; locked regression runs are immutable")
    out.mkdir(parents=True)
    start = time.perf_counter()
    data, lock = collect(out)
    fitted = crossfit(data["x"], data["health_response"], data["utility"], data["groups"])
    summary = summarize(data, fitted)
    atomic_json(out / "regressions.json", fitted)
    atomic_json(out / "summary.json", summary)
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Locked evidence changed")
    if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 16 * 2**20:
        raise ValueError("16MiB output cap")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "cohort_rows": 54,
            "source_groups": 6,
            "health_regression_fits": 36,
            "utility_regression_fits": 36,
            "total_multioutput_fits": 72,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
            "health_derivatives": False,
            "elapsed_seconds": time.perf_counter() - start,
            "hashes": {str(out / n): digest(out / n) for n in ("data.json", "regressions.json", "summary.json")},
        },
    )
    print({"complete": True, "output": str(out), "gates": summary["gates"], "seconds": time.perf_counter() - start})


if __name__ == "__main__":
    main()
