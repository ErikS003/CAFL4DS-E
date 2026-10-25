"""Audit cached stopping-choice headroom without new model execution."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.stopping_headroom import DOSES, analysis, query_halves
from scripts.run_short_buffer_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "outputs/health-warning/20261013-gpu"
READERS = ("finetune", "linear", "knn")
DRAWS = (51011, 51017, 51023)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Immutable audit; no automatic retry")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 256 * 2**20:
        raise ValueError("Insufficient storage reserve")
    tick = time.perf_counter()
    old = read(PRIOR / "input_lock.json")
    for name in ("verification.json", "independent-verification.json"):
        if not read(PRIOR / name)["passed"]:
            raise ValueError("Prior warning audit failed")
    collection = read(PRIOR / "collection.json")
    rows = [r for r in collection["records"] if r["role"] == "six_stream"]
    if not collection["complete"] or len(rows) != 54:
        raise ValueError("Expected fifty-four six-history continuations")
    identities = sorted({(r["stream"], r["checkpoint"]) for r in rows})
    paths = [
        Path(__file__),
        ROOT / "scripts/verify_stopping_headroom.py",
        ROOT / "cafl4ds/stopping_headroom.py",
        ROOT / "tests/unit/test_stopping_headroom.py",
        ROOT / "docs/project-plan/slow-edge-stopping-headroom-frozen.md",
        PRIOR / "collection.json",
        PRIOR / "verification.json",
        PRIOR / "independent-verification.json",
        ROOT / "outputs/regime-windows/20261013-gpu/evaluation_arrays.npz",
    ]
    for row in rows:
        for landmark in row["landmarks"].values():
            paths.extend(Path(landmark["readout"]) / name for name in ("predictions.npz", "readout.json"))
    hashes = {**old["hashes"], **{str(p): digest(p) for p in paths}}
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Inherited source changed")
    lock = {
        "hashes": hashes,
        "case_identities": identities,
        "doses": DOSES.tolist(),
        "prior": str(PRIOR),
        "role": "retrospective_reused_development",
        "new_ssl_updates": 0,
        "new_finetune_fits": 0,
    }
    atomic_json(out / "input_lock.json", lock)
    for path in paths[:5]:
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    with np.load(paths[8], allow_pickle=False) as allocation:
        query = allocation["query"]
        labels, domains = allocation["labels"][query], allocation["domains"][query]
    a, b = query_halves(query, labels, domains)
    np.savez_compressed(out / "query_split.npz", query=query, labels=labels, domains=domains, A=a, B=b)
    tables = {reader: {part: np.zeros((18, 5, 3)) for part in ("all", "A", "B")} for reader in READERS}
    for index, (stream, checkpoint) in enumerate(identities):
        selected = [r for r in rows if (r["stream"], r["checkpoint"]) == (stream, checkpoint)]
        if len(selected) != 3:
            raise ValueError("Expected three optimizer repeats per case")
        for k, dose in enumerate(DOSES):
            panels = {reader: [] for reader in READERS}
            for row in selected:
                directory = Path(row["landmarks"][str(dose)]["readout"])
                with np.load(directory / "predictions.npz", allow_pickle=False) as prediction:
                    for reader in READERS:
                        panels[reader].extend(prediction[f"{draw}_{reader}"] == labels for draw in DRAWS)
            for reader in READERS:
                correct = np.stack(panels[reader])
                for part, mask in (("all", np.ones(400, bool)), ("A", a), ("B", b)):
                    for m, domain in enumerate((None, 0, 1)):
                        membership = mask if domain is None else mask & (domains == domain)
                        tables[reader][part][index, k, m] = 100 * correct[:, membership].mean()
    np.savez_compressed(
        out / "score_tables.npz", **{f"{r}_{p}": v for r, parts in tables.items() for p, v in parts.items()}
    )
    results = {reader: analysis(tables[reader]) for reader in READERS}
    summary = {
        "primary_reader": "finetune",
        "primary": results["finetune"],
        "secondary": {r: v for r, v in results.items() if r != "finetune"},
        "case_identities": identities,
        "new_ssl_updates": 0,
        "new_finetune_fits": 0,
        "elapsed_seconds": time.perf_counter() - tick,
    }
    atomic_json(out / "summary.json", summary)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("Final locked source changed")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > 32 * 2**20:
        raise ValueError("Output cap exceeded")
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "cases": 18,
            "history_groups": 6,
            "prediction_sets": 2430,
            "readout_landmarks": 270,
            "locked_inputs": len(hashes),
            "output_mib": size / 2**20,
            "new_ssl_updates": 0,
            "new_finetune_fits": 0,
        },
    )
    atomic_json(
        out / "status.json",
        {"stage": "complete", "investigation_screen_passed": summary["primary"]["investigation_screen_passed"]},
    )
    print(
        json.dumps(
            {"stage": "complete", "investigation_screen_passed": summary["primary"]["investigation_screen_passed"]}
        )
    )


if __name__ == "__main__":
    main()
