"""Label-free arrival/replay observations and leakage-safe chronological ranking experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.image_utility import embed, reconstruction_draws
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.utility_prediction import feature_matrix

ARRIVAL = ["context_reconstruction_gap", "context_anchor_tv", "context_anchor_entropy", "context_embedding_gap"]
HISTORY = ["context_gap_change", "context_anchor_turnover"]
IMAGE_CONTEXT = ["context_image_reconstruction_z", "context_image_cluster_excess"]


@dataclass
class ArrivalHistory:
    """Only measurements available at a previous observation boundary."""

    reconstruction_gap: float
    anchors: torch.Tensor
    step: int


def arrival_context(
    method: SSLMethod,
    actuator: SteerableReplay,
    batch: torch.Tensor,
    candidates: list[int],
    signals: list[dict[str, float]],
    step: int,
    previous: ArrivalHistory | None,
    draws: int,
) -> ArrivalHistory:
    """Append pre-decision signals; never read labels, future arrivals or outcome pools."""
    if len(batch) != 32 or len(candidates) != len(signals) or not actuator.reservoir._buffer:
        raise ValueError("Need a full arrival batch, matching candidates and historical memory")
    if previous is not None and step <= previous.step:
        raise ValueError("Context history must precede this decision")
    memory = actuator.reservoir._buffer
    indices = torch.randperm(len(memory), generator=torch.Generator().manual_seed(76003))[:32]
    replay = torch.stack([memory[int(index)] for index in indices])
    current_losses = reconstruction_draws(method, batch, draws, 76000).mean(0)
    replay_losses = reconstruction_draws(method, replay, draws, 77000).mean(0)
    mean, scale = current_losses.mean(), current_losses.std(unbiased=False).clamp_min(1e-8)
    gap = float(mean - replay_losses.mean())
    count = len(actuator.target)
    assignments = actuator._assign(batch)
    current_hist = assignments.bincount(minlength=count).float() / len(batch)
    replay_hist = actuator._assign(replay).bincount(minlength=count).float() / len(replay)
    positive = current_hist[current_hist > 0]
    current_z = F.normalize(embed(method, batch), dim=1).mean(0)
    replay_z = F.normalize(embed(method, replay), dim=1).mean(0)
    context = {
        "context_reconstruction_gap": gap,
        "context_anchor_tv": float((current_hist - replay_hist).abs().sum() / 2),
        "context_anchor_entropy": float(-(positive * positive.log()).sum() / math.log(count)),
        "context_embedding_gap": float(1 - F.cosine_similarity(current_z[None], replay_z[None]).item()),
        "context_gap_change": 0.0 if previous is None else (gap - previous.reconstruction_gap) / (step - previous.step),
        "context_anchor_turnover": 0.0
        if previous is None
        else float((current_hist - previous.anchors).abs().sum() / 2),
    }
    for candidate, signal in zip(candidates, signals, strict=True):
        signal.update(context)
        signal["context_image_reconstruction_z"] = (signal["reconstruction_mean"] - float(mean)) / float(scale)
        cluster = assignments[candidate]
        signal["context_image_cluster_excess"] = float(current_hist[cluster] - replay_hist[cluster])
    if any(not math.isfinite(value) for signal in signals for value in signal.values()):
        raise ValueError("Nonfinite context signal")
    return ArrivalHistory(gap, current_hist.clone(), step)


def context_matrix(examples: list[dict[str, Any]], kind: str) -> tuple[np.ndarray, list[str]]:
    """Matched compact arrival/history additions; old image baselines retain their eight inputs."""
    if kind in ("loss_only", "loss_coverage", "loss_coverage_drift", "image_only", "image_health_interactions"):
        return feature_matrix(examples, kind)
    configurations = {
        "loss_coverage_relative": ("loss_coverage", []),
        "loss_coverage_arrival": ("loss_coverage", ARRIVAL),
        "loss_coverage_history": ("loss_coverage", HISTORY),
        "loss_coverage_context": ("loss_coverage", ARRIVAL + HISTORY),
        "image_context": ("image_only", ARRIVAL + HISTORY),
    }
    if kind not in configurations:
        raise ValueError(f"Unknown context variant: {kind}")
    baseline, context_keys = configurations[kind]
    matrix, names = feature_matrix(examples, baseline)
    extra_keys = [*IMAGE_CONTEXT, *context_keys]
    extra = np.array([[row["features"][name] for name in extra_keys] for row in examples])
    cross_names = [
        f"{image}*{state}" for image in ("reconstruction_mean", "coverage_deficit") for state in context_keys
    ]
    cross = np.array(
        [
            [
                row["features"][image] * row["features"][state]
                for image in ("reconstruction_mean", "coverage_deficit")
                for state in context_keys
            ]
            for row in examples
        ]
    )
    result = np.concatenate([matrix, extra, cross], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite context matrix")
    return result, names + extra_keys + cross_names


def center_states(examples: list[dict[str, Any]], values: np.ndarray) -> np.ndarray:
    """Remove common-state offsets without sharing information across states or seeds."""
    if len(examples) != len(values):
        raise ValueError("Invalid state-centering dimensions")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["seed"], row["step"]].append(index)
    centered = np.asarray(values, dtype=float).copy()
    for indices in groups.values():
        centered[indices] -= centered[indices].mean(axis=0)
    return centered


def rolling_split(  # noqa: C901 - explicit chronological availability and role checks
    examples: list[dict[str, Any]],
    seeds: list[int],
    origin: int,
    stop: int,
    horizon: int,
) -> dict[str, list[int]]:
    """Purge outcome windows, tune on past states, and separate seed/time test observations."""
    if len(seeds) < 4 or len(set(seeds)) != len(seeds) or horizon < 1 or stop <= origin:
        raise ValueError("Need four distinct seeds and a valid chronological test window")
    available = sorted({row["step"] for row in examples if row["step"] + horizon <= origin})
    if not available:
        raise ValueError("No outcome labels available before origin")
    validation_steps, fit_steps = [], set()
    for count in (2, 1):
        validation_steps = available[-count:]
        fit_steps = {step for step in available if step + horizon <= min(validation_steps)}
        if fit_steps:
            break
    if not fit_steps:
        raise ValueError("Insufficient purged fitting history")
    roles: dict[str, list[int]] = {name: [] for name in ("fit", "validation", "test_seed", "test_time")}
    for index, row in enumerate(examples):
        seed, step = row["seed"], row["step"]
        if seed not in seeds:
            raise ValueError("Unregistered learning seed")
        if origin <= step and step + horizon <= stop:
            roles["test_time"].append(index)
        elif seed == seeds[-1] and step in available:
            roles["test_seed"].append(index)
        elif seed == seeds[-2] and step in validation_steps:
            roles["validation"].append(index)
        elif seed in seeds[:-2] and step in fit_steps:
            roles["fit"].append(index)
    if not all(roles.values()):
        raise ValueError("Empty chronological prediction split")
    if len({index for indices in roles.values() for index in indices}) != sum(map(len, roles.values())):
        raise ValueError("Overlapping rolling split roles")
    return roles
