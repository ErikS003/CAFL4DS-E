"""Support-trained replay models and explicit-precision linear probes."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cafl4ds.image_utility import FixedProbe, balanced_losses
from cafl4ds.replay_local import fit
from cafl4ds.replay_response import ResponseModel, choose

RECIPES_USED = ("composition", "reconstruction", "health", "gradients", "health_gradients")


def precise_probe(z: torch.Tensor, labels: torch.Tensor, fitting: torch.Tensor, c: float,
                  tolerance: float = 1e-6) -> FixedProbe:
    """Fit and normalize exclusively on fitting labels with explicit solver settings."""
    x = z[fitting].numpy().astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    classifier = LogisticRegression(C=c, tol=tolerance, max_iter=10000, random_state=63000)
    classifier.fit((x - mean) / scale, labels[fitting].numpy())
    if int(classifier.n_iter_.max()) >= 10000:
        raise ValueError("Head solver reached its iteration limit")
    return FixedProbe(mean, scale, classifier)


def metrics(probe: FixedProbe, z: torch.Tensor, labels: torch.Tensor,
            domains: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    """Return accuracy and CE without using scoring labels in a fit."""
    x = (z[indices].numpy().astype(np.float64) - probe.mean) / probe.scale
    prediction = torch.from_numpy(probe.classifier.predict(x))
    values = {"accuracy": (prediction == labels[indices]).double(),
              "ce": probe.losses(z[indices], labels[indices])}
    return {f"{name}_{zone}": value for name, v in values.items()
            for zone, value in balanced_losses(v, labels[indices], domains[indices]).items()}


def train_choices(rows: list[dict[str, Any]], features: list[dict[str, Any]], heldout: str,
                  target: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every related trajectory of the evaluation pool out of model fitting."""
    training = [r for r in rows if r["pool_group"] != heldout and r["target_name"] == target]
    if not training or any(r["pool_group"] == heldout for r in training):
        raise ValueError("Missing training pools or group leakage")
    choices, saved = [], {}
    for recipe in RECIPES_USED:
        model = fit(training, recipe, 100.0)
        saved[recipe] = model.as_dict()
        for state in sorted({r["base_state"] for r in features if r["pool_group"] == heldout}):
            incoming = [r for r in features if r["base_state"] == state and r["action"] != 16]
            predicted = dict(zip([r["action"] for r in incoming], model.predict(incoming).tolist(), strict=True))
            predicted[16] = 0.0
            choices.append({"base_state": state, "pool_group": heldout, "target": target,
                            "recipe": recipe, "action": choose(predicted), "predictions": predicted,
                            "training_states": sorted({r["base_state"] for r in training})})
    values = {a: float(np.mean([r["target"] for r in training if r["action"] == a])) for a in (8, 24)}
    values[16] = 0.0
    for state in sorted({r["base_state"] for r in features if r["pool_group"] == heldout}):
        choices.append({"base_state": state, "pool_group": heldout, "target": target,
                        "recipe": "trained_constant", "action": choose(values), "predictions": values,
                        "training_states": sorted({r["base_state"] for r in training})})
    return choices, saved


def restore_model(saved: dict[str, Any]) -> ResponseModel:
    """Replay a frozen health model without targets or labels."""
    return ResponseModel(tuple(saved["fields"]), np.array(saved["mean"]), np.array(saved["scale"]),
                         np.array(saved["weight"]), saved["regularization"])
