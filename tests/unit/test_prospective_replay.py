"""Frozen policies use original pool exclusion and ignore outcome fields."""

import json
from pathlib import Path

import pandas as pd

from cafl4ds.prospective_replay import decisions


def test_original_pool_selects_frozen_predictor_without_outcomes(tmp_path: Path) -> None:
    """Changing the new order name cannot select a model trained on its pool."""
    directory = tmp_path / "outputs/image-utility/20261008-domain-replay-health/analysis"
    directory.mkdir(parents=True)
    models, constants = [], []
    for group in ("stream1", "stream2"):
        for horizon in (4, 12):
            for target in ("linear_error_balanced", "linear_error_B", "refit_ce_balanced"):
                constants.append(
                    {"horizon": horizon, "heldout_stream": group, "target": target, "training_constant": 16}
                )
                for recipe in (
                    "action_history",
                    "reconstruction",
                    "reconstruction_health",
                    "reconstruction_domain_contrast",
                ):
                    models.append(
                        {
                            "horizon": horizon,
                            "heldout_stream": group,
                            "target": target,
                            "recipe": recipe,
                            "fields": ["step"],
                            "mean": [4.0],
                            "scale": [1.0],
                            "weight": [0.1, 0.0, 0.0, 0.0] if group == "stream1" else [0.0, 0.0, 0.1, 0.0],
                            "regularization": 1.0,
                        }
                    )
    (directory / "response_models.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    pd.DataFrame(constants).to_csv(directory / "policy_choices.csv", index=False)
    rows = [
        {
            "state": "stream6_88106_4",
            "stream": "stream6",
            "pool_group": "stream1",
            "seed": 88106,
            "prefix": 4,
            "action": action,
            "features": {"step": 4.0},
        }
        for action in (8, 16, 24)
    ]
    original = decisions(rows, tmp_path)
    assert {r["action"] for r in original} == {8}
    assert all(r["predictions"][16] == 0 for r in original)
    assert decisions([{**r, "target": -999.0} for r in rows], tmp_path) == original
    other = decisions([{**r, "pool_group": "stream2"} for r in rows], tmp_path)
    assert {r["action"] for r in other} == {24}
