"""Read-only utility features and paired, equal-work image replacement experiments."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from cafl4ds import measurements
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod

Outcome = TypeVar("Outcome")


@contextmanager
def observation(method: SSLMethod, seed: int) -> Iterator[None]:
    """Isolate CPU observation randomness and restore every module's original mode."""
    modes = {module: module.training for module in method.modules()}
    try:
        method.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        for module, training in modes.items():
            module.training = training


def embed(method: SSLMethod, images: torch.Tensor) -> torch.Tensor:
    """Encode a bounded CPU pool without changing the learning trajectory."""
    with observation(method, 70001), torch.no_grad():
        return torch.cat(
            [method.encoder.embed(images[start : start + 64]).detach() for start in range(0, len(images), 64)]
        )


def reconstruction_draws(method: SSLMethod, images: torch.Tensor, draws: int, seed: int) -> torch.Tensor:
    """Use reproducible masks/augmentations and fixed chunking before and after updates."""
    if draws < 1:
        raise ValueError("At least one observation draw required")
    rows = []
    for draw in range(draws):
        with observation(method, seed + 1009 * draw), torch.no_grad():
            rows.append(
                torch.cat([method.per_sample_loss(images[start : start + 32]) for start in range(0, len(images), 32)])
            )
    result = torch.stack(rows)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite reconstruction observation")
    return result


def split_query_roles(evaluation: Evaluation, seed: int = 63000) -> dict[str, torch.Tensor]:
    """Separate monitor, development outcomes and held-out outcomes in every domain/class."""
    generator = torch.Generator().manual_seed(seed)
    roles: dict[str, list[int]] = {name: [] for name in ("monitor", "development", "holdout")}
    for domain in torch.unique(evaluation.domains, sorted=True):
        for label in torch.unique(evaluation.labels, sorted=True):
            query = evaluation.query
            indices = query[(evaluation.domains[query] == domain) & (evaluation.labels[query] == label)]
            indices = indices[torch.randperm(len(indices), generator=generator)]
            if len(indices) < 9:
                raise ValueError("Need at least nine query images per domain/class for three roles")
            for name, part in zip(roles, indices.tensor_split(3), strict=True):
                roles[name].extend(part.tolist())
    result = {name: torch.tensor(indices) for name, indices in roles.items()}
    support = set(torch.cat(list(evaluation.supports.values())).tolist())
    if len(set(torch.cat(list(result.values())).tolist())) != sum(map(len, result.values())):
        raise ValueError("Overlapping query roles")
    if support & set(torch.cat(list(result.values())).tolist()):
        raise ValueError("Support/query overlap")
    return result


def balanced_losses(values: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> dict[str, float]:
    """Average classes within domains, then give the two declared domains equal weight."""
    result = {}
    for domain, name in ((0, "A"), (1, "B")):
        cells = [values[(domains == domain) & (labels == label)].mean() for label in torch.unique(labels)]
        if any(not torch.isfinite(cell) for cell in cells):
            raise ValueError("Missing/nonfinite outcome cell")
        result[name] = float(torch.stack(cells).mean())
    result["balanced"] = (result["A"] + result["B"]) / 2
    return result


@dataclass
class FixedProbe:
    """A warm support-fitted classifier whose parameters and normalization never adapt."""

    mean: np.ndarray
    scale: np.ndarray
    classifier: LogisticRegression

    @classmethod
    def fit(cls, method: SSLMethod, evaluation: Evaluation) -> FixedProbe:
        """Use labeled support only, before the first stream update."""
        support = next(iter(evaluation.supports.values()))
        x = embed(method, evaluation.images[support]).numpy().astype(np.float64)
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1
        classifier = LogisticRegression(C=1, max_iter=1000, random_state=63000)
        classifier.fit((x - mean) / scale, evaluation.labels[support].numpy())
        return cls(mean, scale, classifier)

    def losses(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Score fixed-head cross entropy, retaining per-image outcomes."""
        probabilities = self.classifier.predict_proba((features.numpy() - self.mean) / self.scale)
        lookup = {int(label): i for i, label in enumerate(self.classifier.classes_)}
        indices = [lookup[int(label)] for label in labels]
        return torch.from_numpy(-np.log(probabilities[np.arange(len(labels)), indices].clip(1e-12, 1)))


def outcome_losses(
    method: SSLMethod,
    evaluation: Evaluation,
    roles: dict[str, torch.Tensor],
    probe: FixedProbe,
    draws: int,
    *,
    reconstruction: bool = True,
) -> dict[str, dict[str, float]]:
    """Read outcomes on pools that are never inputs to the utility feature extractor."""
    result = {}
    for name in ("development", "holdout"):
        indices = roles[name]
        images, labels, domains = evaluation.images[indices], evaluation.labels[indices], evaluation.domains[indices]
        ce = probe.losses(embed(method, images), labels)
        result[name] = {f"fixed_ce_{k}": v for k, v in balanced_losses(ce, labels, domains).items()}
        if reconstruction:
            recon = reconstruction_draws(method, images, draws, 71000).mean(0)
            result[name].update({f"reconstruction_{k}": v for k, v in balanced_losses(recon, labels, domains).items()})
    return result


def gradient_vector(method: SSLMethod, images: torch.Tensor, seed: int) -> torch.Tensor:
    """Diagnostic last-block SSL gradient; no labels, backward mutation or optimizer step."""
    parameters = tuple(method.encoder.blocks[-1].parameters())
    with observation(method, seed), torch.enable_grad():
        loss = method.training_step(images)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).detach().flatten()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def utility_features(
    method: SSLMethod,
    actuator: SteerableReplay,
    images: torch.Tensor,
    monitor_images: torch.Tensor,
    warm_monitor_features: torch.Tensor,
    *,
    draws: int = 3,
    gradient_signals: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Extract only pre-decision image signals and label-free monitor health."""
    losses = reconstruction_draws(method, images, draws, 72000)
    z, z_monitor = embed(method, images), embed(method, monitor_images)
    unit, monitor_unit = F.normalize(z, dim=1), F.normalize(z_monitor, dim=1)
    health = {
        "health_rankme": measurements.rankme(z_monitor),
        "health_variance": float(monitor_unit.var(dim=0).mean()),
        "health_covariance": measurements.offdiag_covariance(monitor_unit),
        "health_uniformity": measurements.uniformity(z_monitor),
        "health_cosine_drift": measurements.cosine_drift(warm_monitor_features, z_monitor),
        "health_cka_drift": measurements.cka_drift(warm_monitor_features, z_monitor),
        "health_reconstruction": float(reconstruction_draws(method, monitor_images, draws, 73000).mean()),
        "health_occupancy_tv": float((actuator.target - actuator.occupancy).abs().sum() / 2),
    }
    reference = actuator.reference_encoder
    with torch.no_grad():
        warm_z = reference.embed(images)
        clusters = actuator._assign(images)
        deficit = (actuator.target - actuator.occupancy).clamp_min(0)[clusters]
        drift = 1 - (unit * F.normalize(warm_z, dim=1)).sum(1)
        memory = torch.stack(actuator.reservoir._buffer)
        memory_z = F.normalize(embed(method, memory), dim=1)
        similarity = (unit @ memory_z.T).max(1).values
    with observation(method, 74000), torch.no_grad():
        views = method.make_views(images)
        if views is None:
            raise ValueError("This experiment requires the MAE positive-view reader")
        alignment = (
            (F.normalize(method.encoder.embed(views[0]), dim=1) - F.normalize(method.encoder.embed(views[1]), dim=1))
            .square()
            .sum(1)
        )
    values = {
        "reconstruction_mean": losses.mean(0),
        "reconstruction_std": losses.std(0, unbiased=False),
        "coverage_deficit": deficit,
        "embedding_drift": drift,
        "replay_similarity": similarity,
        "view_disagreement": alignment,
        "pixel_variance": images.flatten(1).var(dim=1),
        "feature_norm": z.norm(dim=1),
    }
    if gradient_signals:
        monitor_gradient = gradient_vector(method, monitor_images, 75000)
        gradients = torch.stack([gradient_vector(method, image[None], 75001) for image in images])
        values["gradient_norm"] = gradients.norm(dim=1)
        values["gradient_monitor_cosine"] = F.cosine_similarity(gradients, monitor_gradient[None], dim=1)
        health["health_gradient_norm"] = float(monitor_gradient.norm())
    rows = [{**{key: float(value[i]) for key, value in values.items()}, **health} for i in range(len(images))]
    if not all(np.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("Nonfinite utility feature")
    return rows, health


def replacement_batches(
    arrivals: torch.Tensor, replay: torch.Tensor, candidate_indices: list[int], seed: int
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """Keep 31 images fixed; substitute one candidate for a random noncandidate arrival."""
    if len(arrivals) != 32 or len(replay) != 16 or not candidate_indices:
        raise ValueError("Require 32 arrivals, 16 replay images and candidates")
    if len(set(candidate_indices)) != len(candidate_indices) or any(i not in range(32) for i in candidate_indices):
        raise ValueError("Candidate indices must be distinct batch indices")
    others = [i for i in range(32) if i not in candidate_indices]
    if len(others) < 16:
        raise ValueError("At most 16 candidate images per snapshot")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(others), generator=generator).tolist()
    background, reference = [others[i] for i in order[:15]], others[order[15]]
    context = torch.cat((arrivals[background], replay))
    baseline = torch.cat((context[:15], arrivals[reference : reference + 1], context[15:]))
    branches = [torch.cat((context[:15], arrivals[index : index + 1], context[15:])) for index in candidate_indices]
    return baseline, branches, reference


def branch_outcomes(
    method: SSLMethod,
    optimizer: torch.optim.Optimizer,
    first_batch: torch.Tensor,
    continuation: list[torch.Tensor],
    horizons: list[int],
    rng: torch.Tensor,
    evaluate: Callable[[SSLMethod], Outcome],
) -> dict[int, Outcome]:
    """Restore exact AdamW/model/RNG state for an equal-work intervention trajectory."""
    if not horizons or min(horizons) < 1 or max(horizons) > len(continuation) + 1:
        raise ValueError("Invalid intervention horizons")
    trial = copy.deepcopy(method)
    trial_optimizer = torch.optim.AdamW(trial.parameters())
    trial_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    result = {}
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        for step, images in enumerate([first_batch, *continuation], start=1):
            if len(images) != 32:
                raise ValueError("Every branch must use exactly 32 optimizer images per update")
            trial.train()
            trial_optimizer.zero_grad(set_to_none=True)
            loss = trial.training_step(images)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite intervention loss")
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(trial.parameters(), 1.0, error_if_nonfinite=True)
            trial_optimizer.step()
            if step in horizons:
                result[step] = evaluate(trial)
            if step == max(horizons):
                break
    return result
