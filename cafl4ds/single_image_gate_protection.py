"""Bounded clock-free calibration and development-only transactional guards."""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.special import expit

from cafl4ds.single_image_filter import FrozenRegressor
from cafl4ds.single_image_filter_factorial import admission_threshold

ARMS = ("corrected_J", "corrected_covariance", "corrected_development")
GROUPS = ("balanced", "day", "night")


class BoundedForecaster:
    """Refit without either raw clock; never extrapolate retained coordinates."""

    def __init__(self, family, x, y, indices, settings):
        self.indices = np.asarray(indices, int)
        values = np.asarray(x, float)[:, self.indices]
        self.minimum, self.maximum = values.min(0), values.max(0)
        self.regressor = FrozenRegressor(family, values, np.asarray(y, float), settings)

    def transform(self, x):
        values = np.atleast_2d(x)[:, self.indices]
        if not np.isfinite(values).all():
            raise ValueError("Finite gate coordinates required")
        return np.clip(values, self.minimum, self.maximum)

    def predict(self, x):
        return self.regressor.predict(self.transform(x))

    def clipped_fraction(self, x):
        values = np.asarray(x)[self.indices]
        return float(np.mean((values < self.minimum) | (values > self.maximum)))


def fit_corrected(arrays, names, settings, excluded, seed):
    """History-held-out family choice and threshold, with no N73 outcomes."""
    indices = [i for i, name in enumerate(names) if name not in excluded]
    if len(indices) != len(names) - len(excluded):
        raise ValueError("Every excluded feature must exist exactly once")
    take = arrays["seeds"] == seed
    x, y = arrays["rich"][take], arrays["targets"][take, 0]
    valid = np.isin(arrays["histories"][take], settings["validation_histories"])
    if not valid.any() or valid.all():
        raise ValueError("Disjoint fit and validation histories required")
    losses, predictions = {}, {}
    for family in settings["regressors"]:
        model = BoundedForecaster(family, x[~valid], y[~valid], indices, settings)
        predictions[family] = model.predict(x[valid])
        losses[family] = float(np.mean((predictions[family] - y[valid]) ** 2))
    winner = min(losses, key=losses.get)
    p = predictions[winner]
    threshold = admission_threshold(p, settings["max_validation_admission"])
    model = BoundedForecaster(winner, x, y, indices, settings)
    return model, dict(
        family=winner,
        threshold=threshold,
        excluded_features=excluded,
        retained_features=[names[i] for i in indices],
        validation_mse=losses,
        validation_predictions=p.tolist(),
        validation_targets=y[valid].tolist(),
        validation_admission=float(np.mean(p < threshold)),
        validation_rows=int(valid.sum()),
        fit_rows=int((~valid).sum()),
        final_refit_rows=len(y),
        objective="unchanged mean three-view delta J",
        bounded_live_coordinates=True,
        final_query_outcomes_used=False,
    )


def development_scores(probe, panel, temperature):
    """Support-fitted probe accuracy and boundary-aware error on development only."""
    p = np.asarray(probe["probabilities"], float)[:, 0]
    labels, domains = panel["query_labels"], panel["query_domains"]
    correct = p.argmax(-1) == labels[None, :]
    true_p = p[:, np.arange(len(labels)), labels]
    masks = [np.ones(len(labels), bool), domains == 0, domains == 1]
    if any(not mask.any() for mask in masks):
        raise ValueError("All development groups must be represented")
    accuracy = np.array([correct[:, mask].mean() for mask in masks])
    soft_error = np.array([expit((0.5 - true_p[:, mask]) / temperature).mean() for mask in masks])
    return np.r_[accuracy, soft_error]


def protection_passes(arm, covariance, scores, anchor, covariance_cap, tolerance):
    if arm not in ARMS:
        raise ValueError("Unknown protection arm")
    if not np.isfinite(np.r_[covariance, scores, anchor, covariance_cap, tolerance]).all():
        raise ValueError("Nonfinite protection input")
    if arm == "corrected_covariance":
        return bool(covariance <= covariance_cap + 1e-12)
    if arm == "corrected_development":
        return bool(np.all(np.asarray(scores)[:3] >= np.asarray(anchor) - tolerance - 1e-12))
    return True


def confirmed_anchor(anchor, previous_checkpoint, current_checkpoint):
    """Raise each floor only after it is supported by two committed checkpoints."""
    return np.maximum(anchor, np.minimum(previous_checkpoint, current_checkpoint))


def rollback(model, optimizer, state):
    """Undo trial weights, optimizer buffers, modes and every captured RNG."""
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    for name, module in model.named_modules():
        module.training = state["modes"][name]
    torch.set_rng_state(state["cpu_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
