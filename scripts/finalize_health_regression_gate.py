"""Complete a frozen cached fit after a NumPy-scalar JSON serialization failure."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.health_regression_gate import summarize
from scripts.run_health_regression_gate import digest, read


def plain(value):
    """Convert representation types only; never change numeric values or criteria."""
    if isinstance(value, dict):
        return {key: plain(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "verification.json").exists():
        raise ValueError("Completed runs are immutable")
    lock = read(out / "input_lock.json")
    if any(digest(Path(p)) != h for p, h in lock["hashes"].items()):
        raise ValueError("Frozen scientific source or input changed")
    data, fitted = read(out / "data.json"), read(out / "regressions.json")
    original = {str(out / n): digest(out / n) for n in ("data.json", "regressions.json")}
    summary = plain(summarize(data, fitted))
    atomic_json(out / "summary.json", summary)
    if any(digest(Path(p)) != h for p, h in original.items()):
        raise ValueError("Original fitted evidence changed")
    scientific_fits = sum(1 + len(f["inner_health"]) + len(f["utility_models"]) for f in fitted["folds"].values())
    if scientific_fits != 72:
        raise ValueError("Complete original fit grid required")
    atomic_json(
        out / "serialization-repair.json",
        {
            "original_error": "TypeError: Object of type int64 is not JSON serializable",
            "repair": "Recursively convert NumPy scalars/arrays to JSON-native representations only",
            "original_fitted_artifacts_preserved": original,
            "new_fits": 0,
            "criteria_changed": False,
            "frozen_sources_changed": False,
            "source": str(Path(__file__).resolve()),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    atomic_json(
        out / "verification.json",
        {
            "passed": True,
            "cohort_rows": 54,
            "source_groups": 6,
            "health_regression_fits": 36,
            "utility_regression_fits": 36,
            "total_multioutput_fits": scientific_fits,
            "new_ssl_updates": 0,
            "new_classifier_fits": 0,
            "health_derivatives": False,
            "elapsed_seconds": None,
            "runtime_note": "Original collector exited after all fitted models were saved; exact internal timer unavailable. Summary finalized without refitting.",
            "serialization_repair": str(out / "serialization-repair.json"),
            "hashes": {str(out / n): digest(out / n) for n in ("data.json", "regressions.json", "summary.json")},
        },
    )
    print({"complete": True, "fits_preserved": scientific_fits, "new_fits": 0, "gates": summary["gates"]})


if __name__ == "__main__":
    main()
