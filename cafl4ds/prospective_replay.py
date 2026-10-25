"""Use previously frozen response models without learning from new outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cafl4ds.replay_response import ResponseModel, choose


def decisions(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Predict one parent using models trained outside its original image pool."""
    if len(rows) != 3 or len({r["state"] for r in rows}) != 1:
        raise ValueError("Require three actions from a single parent")
    group = rows[0]["pool_group"]
    recipes = {
        "level": ("action_history", "reconstruction", "reconstruction_health", "reconstruction_domain_contrast"),
        "temporal": ("contrast_change", "contrast_level_change", "domain_level_change"),
    }
    output = []
    for family, selected in recipes.items():
        if family == "temporal" and rows[0]["prefix"] != 12:
            continue
        directory = (
            root
            / "outputs/image-utility"
            / ("20261008-domain-replay-health/analysis" if family == "level" else "20261008-replay-health-dynamics")
        )
        models = json.loads((directory / "response_models.json").read_text())["models"]
        controls = pd.read_csv(directory / "policy_choices.csv")
        for horizon in (4, 12):
            for recipe in selected:
                predictions = {}
                for target in ("linear_error_balanced", "linear_error_B", "refit_ce_balanced"):
                    saved = next(
                        m
                        for m in models
                        if m["horizon"] == horizon
                        and m["heldout_stream"] == group
                        and m["recipe"] == recipe
                        and m["target"] == target
                    )
                    model = ResponseModel(
                        tuple(saved["fields"]),
                        np.array(saved["mean"]),
                        np.array(saved["scale"]),
                        np.array(saved["weight"]),
                        saved["regularization"],
                    )
                    predictions[target] = dict(zip([r["action"] for r in rows], model.predict(rows), strict=True))
                for target in ("linear_error_balanced", "refit_ce_balanced"):
                    constant = controls[
                        (controls.horizon == horizon) & (controls.heldout_stream == group) & (controls.target == target)
                    ].training_constant.unique()
                    if len(constant) != 1:
                        raise ValueError("Ambiguous frozen constant")
                    identity = {k: rows[0][k] for k in ("state", "stream", "pool_group", "seed", "prefix")}
                    identity.update(
                        horizon=horizon, family=family, recipe=recipe, target=target, training_constant=int(constant[0])
                    )
                    pred = predictions[target]
                    for gate in ("best_gain", "night_guard") if target == "linear_error_balanced" else ("best_gain",):
                        night = predictions["linear_error_B"] if gate == "night_guard" else None
                        output.append(
                            {
                                **identity,
                                "gate": gate,
                                "action": choose(pred, night),
                                "predictions": pred,
                                "night_predictions": predictions["linear_error_B"],
                            }
                        )
    return output
