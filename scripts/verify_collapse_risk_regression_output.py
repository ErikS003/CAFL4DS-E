"""Run the frozen numerical verifier with NumPy scalars normalized only at JSON output."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from cafl4ds.artifacts import atomic_json
from scripts import verify_collapse_risk_regression as frozen


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def write_normalized(path, value):
    """The scientific assertions and fits remain in the unchanged locked verifier."""
    atomic_json(path, plain(value))
    lock = frozen.read(Path(path).parent / "input_lock.json")
    source = Path(frozen.__file__).resolve()
    atomic_json(
        Path(path).parent / "verification-output-adapter.json",
        {
            "frozen_verifier_sha256": frozen.sha(source),
            "frozen_verifier_matches_input_lock": frozen.sha(source) == lock["hashes"][str(source)],
            "output_adapter_sha256": frozen.sha(Path(__file__)),
            "change": "NumPy scalar to equivalent Python scalar at JSON serialization only; no assertions, criteria or fits changed",
        },
    )


if __name__ == "__main__":
    frozen.atomic_json = write_normalized
    with threadpool_limits(limits=1):
        frozen.main()
