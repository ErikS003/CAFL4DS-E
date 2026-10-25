"""Matched eight-image utility interventions and frozen pool-held-out predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.replay_ratio import ReplayRatio

CHOICES = ("uniform", "replay_a", "replay_b", "current_a")
FIELDS = {
    "constant": [],
    "reconstruction": ["reconstruction", "virtual_ssl_loss"],
    "geometry": ["reconstruction", "coverage", "variance", "novelty"],
    "health_ssl": [
        "reconstruction",
        "coverage",
        "variance",
        "novelty",
        "virtual_ssl_loss",
        "gradient_norm",
        "current_cosine",
        "old_ssl_benefit",
        "health_rankme",
        "health_variance",
        "health_cka_drift",
        "health_reconstruction",
    ],
    "labeled_gradient": ["task_benefit0", "task_benefit1", "task_benefit2"],
}


def choices(current: list[int], candidates: list[int]) -> dict[str, dict[str, list[int]]]:
    """Protect24 baseline slots; replace exactly8 current or replay images."""
    if len(current) != 16 or len(set(current)) != 16 or len(candidates) != 64 or len(set(candidates)) != 64:
        raise ValueError("Require16 current and64 distinct old positions")
    if min(current) < 0 or max(current) >= 32 or min(candidates) < 0:
        raise ValueError("Invalid image positions")
    unused = [i for i in range(32) if i not in current][:8]
    return {
        "uniform": {"current": current.copy(), "replay": candidates[:16]},
        "replay_a": {"current": current.copy(), "replay": candidates[:8] + candidates[16:24]},
        "replay_b": {"current": current.copy(), "replay": candidates[:8] + candidates[24:32]},
        "current_a": {"current": current[:8] + unused, "replay": candidates[:16]},
    }


class FixedTrainingChoice(ReplayRatio):
    """Replace one complete matched batch before action-independent ingestion."""

    def __init__(self, selection: dict[str, list[int]], first_step: int, seed: int) -> None:
        """Validate sixteen images from each source and retain ordinary future RNG."""
        super().__init__(16, seed=seed)
        a, b = selection["current"], selection["replay"]
        if len(a) != 16 or len(b) != 16 or len(set(a)) != 16 or len(set(b)) != 16:
            raise ValueError("Require16 distinct current and old images")
        if min(a) < 0 or max(a) >= 32 or min(b) < 0:
            raise ValueError("Invalid selection")
        self.selection, self.first_step = selection, first_step

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        """Later updates use common uniform draws; all arrivals enter memory."""
        replacement = None
        if ctx.step == self.first_step:
            old = torch.stack([self.reservoir._buffer[i] for i in self.selection["replay"]])
            replacement = torch.cat((batch.images[self.selection["current"]], old.to(batch.images.device)))
        selected = super().select(batch, ctx)
        if replacement is not None:
            self.last["current_indices"] = self.selection["current"].copy()
            self.last["replay_indices"] = self.selection["replay"].copy()
            return replacement
        return selected


def freeze_predictors(features: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Train only four-update development draw0 from other entire original pools."""
    rows = pd.DataFrame(scores)
    primary = rows.loc[
        (rows.draw == 0)
        & (rows.horizon == 4)
        & (rows.panel == "pooled")
        & (rows.role == "development")
        & (rows.tolerance == 1e-10)
    ]
    pools = sorted({r["pool_group"] for r in features})
    if len(pools) != 3:
        return {"models": [], "choices": [], "forecasts": [], "reason": "Smoke has fewer than three pools"}
    lookup = {(r["state"], r["rule"]): r["features"] for r in features}
    data: list[dict[str, Any]] = []
    for state, frame in primary.groupby("state"):
        values = frame.set_index("rule")
        base = lookup[(str(state), "uniform")]
        for rule in CHOICES[1:]:
            observed = lookup[(str(state), rule)]
            data.append(
                {
                    "state": str(state),
                    "pool_group": str(values.iloc[0].pool_group),
                    "rule": rule,
                    "x": {k: observed[k] if k.startswith("health_") else observed[k] - base[k] for k in base},
                    "y": float(values.loc[rule, "accuracy_balanced"] - values.loc["uniform", "accuracy_balanced"]),
                }
            )
    models, actions, forecasts = [], [], []
    for heldout in pools:
        training = [r for r in data if r["pool_group"] != heldout]
        testing = [r for r in data if r["pool_group"] == heldout]
        for recipe, fields in FIELDS.items():
            x = np.array([[r["x"][k] for k in fields] for r in training])
            y = np.array([r["y"] for r in training])
            mean, scale = x.mean(0), x.std(0)
            scale[scale < 1e-12] = 1
            constants = {rule: float(np.mean([r["y"] for r in training if r["rule"] == rule])) for rule in CHOICES[1:]}
            residual = y - np.array([constants[r["rule"]] for r in training])
            model = Ridge(alpha=100.0).fit((x - mean) / scale, residual) if fields else None
            models.append(
                {
                    "heldout_pool": heldout,
                    "recipe": recipe,
                    "fields": fields,
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                    "constant_by_rule": constants,
                    "residualized_by_training_choice": True,
                    "weights": model.coef_.tolist() if model else [],
                    "intercept": float(model.intercept_) if model else 0.0,
                    "training_states": sorted({r["state"] for r in training}),
                }
            )
            for state in sorted({r["state"] for r in testing}):
                predictions = {"uniform": 0.0}
                for row in [r for r in testing if r["state"] == state]:
                    prediction = (
                        constants[row["rule"]]
                        + float(model.predict((np.array([[row["x"][k] for k in fields]]) - mean) / scale)[0])
                        if model
                        else constants[row["rule"]]
                    )
                    predictions[row["rule"]] = prediction
                    forecasts.append(
                        {
                            "state": state,
                            "pool_group": heldout,
                            "recipe": recipe,
                            "rule": row["rule"],
                            "predicted_gain": prediction,
                        }
                    )
                best = max(CHOICES, key=lambda r: predictions[r])
                actions.append(
                    {
                        "state": state,
                        "pool_group": heldout,
                        "recipe": recipe,
                        "rule": best if predictions[best] >= 0.001 else "uniform",
                        "predictions": predictions,
                    }
                )
    return {
        "models": models,
        "choices": actions,
        "forecasts": forecasts,
        "target": "four-update pooled development draw0 accuracy gain",
        "minimum_gain": 0.001,
        "Ridge_alpha": 100.0,
        "evaluation_draws": [1, 2],
        "labeled_recipes": ["labeled_gradient"],
        "query_labels_runtime": False,
    }
