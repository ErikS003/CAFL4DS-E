"""Check the intended diet manipulation without training or reading outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cafl4ds.artifacts import atomic_json
from cafl4ds.sustained_replay import ARMS, SEEDS, batches, exposure_summary
from scripts.run_short_buffer_experiment import digest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if (out / "input_lock.json").exists():
        raise ValueError("Do not rewrite a locked manipulation preflight")
    out.mkdir(parents=True, exist_ok=True)
    intake = ROOT / "outputs/coverage-admission/20261013-frames/intake.json"
    records, ratios, distances = [], [], []
    for stream in json.loads(intake.read_text())["streams"]:
        for seed in SEEDS:
            plans = {a: batches(stream["name"], seed, a) for a in ARMS}
            metrics = {a: exposure_summary(p) for a, p in plans.items()}
            for dose in (25, 50):
                u, r = metrics[f"uniform{dose}"], metrics[f"recent{dose}"]
                ratios.append(r["mean_old_age"] / u["mean_old_age"])
                up = np.array(u["old_exposures_by_clip"]) / u["old_exposures"]
                rp = np.array(r["old_exposures_by_clip"]) / r["old_exposures"]
                distances.append(float(np.abs(up - rp).sum() / 2))
            records.append({"stream": stream["name"], "seed": seed, "arms": metrics})
    passed = max(ratios) <= 0.6 and min(distances) >= 0.15
    result = {
        "passed": passed,
        "outcomes_consulted": False,
        "training_updates": 0,
        "criterion": "Every source/seed/dose recent-to-uniform mean-age ratio <=0.6 and clip-share TV >=0.15",
        "max_age_ratio": max(ratios),
        "min_clip_share_tv": min(distances),
        "records": records,
        "hashes": {str(p): digest(p) for p in (intake, ROOT / "cafl4ds/sustained_replay.py", Path(__file__))},
    }
    atomic_json(out / "diet-preflight.json", result)
    print({k: v for k, v in result.items() if k not in ("records", "hashes")})
    if not passed:
        raise ValueError("Diet contrast too small; no scientific outcomes may be collected")


if __name__ == "__main__":
    main()
