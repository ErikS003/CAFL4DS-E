"""Verify counterfactual isolation, equal work, outcome direction and predictor leakage controls."""

from __future__ import annotations

import copy
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

import cafl4ds.image_utility as utility
from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.image_utility import branch_outcomes, observation, replacement_batches, split_query_roles
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.recurrence import Evaluation
from cafl4ds.ssl.base import SSLMethod
from cafl4ds.ssl.factory import build_mae
from cafl4ds.utility_prediction import (
    aggregate_repeats,
    feature_matrix,
    ranking_metrics,
    repeat_reliability,
    split_examples,
)


class ScalarMethod(SSLMethod):
    """An analytically interpretable plant with stochastic observations."""

    def __init__(self) -> None:
        """Initialize one adjustable scalar and an independently configured module."""
        super().__init__(cast(TinyViTEncoder, nn.Identity()))
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.extra = nn.Dropout()

    @property
    def name(self) -> str:
        """Identify the test plant."""
        return "scalar"

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Fit the batch mean, drawing randomness to check trajectory isolation."""
        torch.rand(1)
        return (self.weight - imgs.mean()).square()

    def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        """Expose a noisy read-only observation."""
        return (self.weight.detach() - imgs.flatten(1).mean(1)).square() + torch.rand(len(imgs))


def test_observation_restores_mixed_modes_and_rng_on_error() -> None:
    """A failed instrument cannot change subsequent learning randomness or module modes."""
    method = ScalarMethod().train()
    method.extra.eval()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="reader"):
        with observation(method, 42):
            assert not method.training
            torch.rand(5)
            raise RuntimeError("reader")
    assert method.training and not method.extra.training
    assert torch.equal(rng, torch.get_rng_state())


def test_only_one_slot_changes_and_reference_is_not_a_candidate() -> None:
    """Each candidate displaces a random control while context and historical replay stay fixed."""
    arrivals = torch.arange(32.0).reshape(32, 1, 1, 1)
    replay = torch.arange(100.0, 116.0).reshape(16, 1, 1, 1)
    baseline, branches, reference = replacement_batches(arrivals, replay, [0, 4, 8, 12], 3)
    assert reference not in [0, 4, 8, 12]
    for index, branch in zip([0, 4, 8, 12], branches, strict=True):
        assert len(branch) == 32
        assert (baseline.flatten() != branch.flatten()).nonzero().flatten().tolist() == [15]
        assert branch[15].item() == index
        assert torch.equal(branch[16:], replay)


def test_paired_branches_restore_optimizer_model_rng_and_measure_positive_benefit() -> None:
    """Helpful intervention lowers held-out loss without touching the prefix state."""
    method = ScalarMethod()
    optimizer = torch.optim.AdamW(method.parameters(), lr=0.01)
    before = copy.deepcopy(method.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    zero = torch.zeros(32, 1, 1, 1)
    candidate = zero.clone()
    candidate[15] = 1

    def evaluate(trial: ScalarMethod) -> float:
        return float((trial.weight.detach() - 1).square())

    baseline = branch_outcomes(method, optimizer, zero, [zero], [1, 2], rng, evaluate)
    actual = branch_outcomes(method, optimizer, candidate, [zero], [1, 2], rng, evaluate)
    duplicate = branch_outcomes(method, optimizer, candidate, [zero], [1, 2], rng, evaluate)
    assert actual == duplicate
    assert baseline[1] - actual[1] > 0
    assert all(torch.equal(before[key], value) for key, value in method.state_dict().items())
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="32 optimizer"):
        branch_outcomes(method, optimizer, zero[:16], [], [1], rng, evaluate)
    assert torch.equal(torch.get_rng_state(), rng)


def test_monitor_development_and_holdout_are_disjoint_balanced_pools() -> None:
    """Every query role has each domain/class and excludes labeled support."""
    labels = torch.tensor([label for domain in range(2) for label in range(3) for _ in range(12)])
    domains = torch.tensor([domain for domain in range(2) for label in range(3) for _ in range(12)])
    supports = torch.tensor([offset + i for offset in range(0, 72, 12) for i in range(3)])
    query = torch.tensor([offset + i for offset in range(0, 72, 12) for i in range(3, 12)])
    evaluation = Evaluation(torch.zeros(72, 3, 2, 2), labels, domains, query, {1: supports}, {})
    roles = split_query_roles(evaluation)
    assert len(set(torch.cat(list(roles.values())).tolist())) == len(query)
    assert not set(supports.tolist()) & set(torch.cat(list(roles.values())).tolist())
    for indices in roles.values():
        assert all(
            int(((domains[indices] == d) & (labels[indices] == c)).sum()) == 3 for d in range(2) for c in range(3)
        )


def examples() -> list[dict]:
    """Give all split roles early/final snapshots with distinct future arrival identities."""
    return [
        {
            "seed": seed,
            "step": step,
            "visit": visit,
            "candidate": c,
            "arrival_id": step * 32 + c,
            "development": float(c),
            "holdout": float(c),
            "features": {
                "reconstruction_mean": float(c),
                "coverage_deficit": float(1 - c),
                "health_drift": float(visit),
            },
        }
        for seed in [1, 2, 3]
        for step, visit in [(1, 0), (8, 3)]
        for c in [0, 1]
    ]


def test_split_groups_seeds_and_future_images_without_row_leakage() -> None:
    """Future candidates and test seed are excluded from fitting and validation."""
    rows = examples()
    splits = split_examples(rows, [1, 2, 3])
    assert {rows[i]["seed"] for i in splits["fit"]} == {1}
    assert {rows[i]["seed"] for i in splits["validation"]} == {2}
    assert {rows[i]["seed"] for i in splits["test_seed"]} == {3}
    assert all(rows[i]["visit"] == 3 for i in splits["test_time"])
    rows[2]["arrival_id"] = rows[0]["arrival_id"]
    with pytest.raises(ValueError, match="leaked"):
        split_examples(rows, [1, 2, 3])


def test_health_needs_interactions_to_change_order_and_regret_uses_snapshot_groups() -> None:
    """Health-only scores tie within a snapshot, while image-health products can reverse rankings."""
    rows = examples()
    health, _ = feature_matrix(rows, "health_only")
    assert health[0].tolist() == health[1].tolist()
    matrix, keys = feature_matrix(rows, "image_health_interactions")
    column = keys.index("reconstruction_mean*health_drift")
    assert matrix[0, column] == matrix[1, column] == 0
    assert matrix[3, column] > matrix[2, column]
    scores = np.array([row["holdout"] for row in rows])
    result = ranking_metrics(rows, scores, "holdout")
    assert result["snapshots"] == 6
    assert result["regret"] == 0
    assert result["gain_over_random"] == 0.5
    tied = ranking_metrics(rows, np.zeros(len(rows)), "holdout")
    assert tied["gain_over_random"] == 0
    assert tied["mean_rank_correlation"] is None
    assert tied["regret"] == 0.5


def test_repeats_are_measurements_not_extra_training_examples() -> None:
    """Context repeats yield one candidate label with an explicit noise estimate."""
    base = examples()[0]
    rows = [
        {**base, "repeat": repeat, "gain": {"1": {"development": {"metric": value}, "holdout": {"metric": value * 2}}}}
        for repeat, value in enumerate([-1.0, 3.0])
    ]
    result = aggregate_repeats(rows, 1, "metric")
    assert len(result) == 1
    assert result[0]["development"] == 1
    assert result[0]["holdout"] == 2
    assert result[0]["repeat_std"] == 2
    assert result[0]["repeat_sign_agreement"] == 0.5


def test_repeat_ranking_ignores_the_shared_random_reference_offset() -> None:
    """A reference shift changes utility levels but supplies no noise in candidate ordering."""
    rows = [
        {
            "seed": 1,
            "step": 1,
            "candidate": candidate,
            "repeat": repeat,
            "gain": {"1": {"development": {"metric": offset + candidate}}},
        }
        for repeat, offset in enumerate([0, 100])
        for candidate in range(4)
    ]
    reliability = repeat_reliability(rows, 1, "metric")
    assert reliability["mean_centered_repeat_std"] == 0
    assert reliability["mean_pairwise_repeat_rank_correlation"] == 1


def test_sparse_health_and_gradient_ablation_has_matched_baselines() -> None:
    """Gradient instrumentation cannot silently enrich the image-only comparison."""
    rows = examples()
    for row in rows:
        row["features"].update(
            health_cosine_drift=0.25, gradient_norm=2.0, gradient_monitor_cosine=-0.5, health_gradient_norm=3.0
        )
    image, image_keys = feature_matrix(rows, "image_only")
    _, health_keys = feature_matrix(rows, "health_only")
    sparse, sparse_keys = feature_matrix(rows, "loss_coverage_drift")
    extended, extended_keys = feature_matrix(rows, "loss_coverage_drift_gradient")
    assert not any(key.startswith("gradient_") for key in image_keys)
    assert "health_gradient_norm" not in health_keys
    assert image.shape[1] == 2
    assert sparse.shape[1] == 5
    assert extended.shape[1] == 7
    assert "gradient_monitor_cosine" in extended_keys
    assert sparse[1, sparse_keys.index("reconstruction_mean*health_cosine_drift")] == 0.25


def test_primary_only_outcome_keeps_identical_ce_and_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting secondary reconstruction reads cannot alter the primary target."""
    labels = torch.tensor([label for domain in range(2) for label in range(2) for _ in range(3)])
    domains = torch.tensor([domain for domain in range(2) for label in range(2) for _ in range(3)])
    images = torch.arange(12.0).reshape(12, 1, 1, 1)
    evaluation = Evaluation(images, labels, domains, torch.arange(12), {}, {})
    roles = {"development": torch.tensor([0, 3, 6, 9]), "holdout": torch.tensor([1, 4, 7, 10])}
    monkeypatch.setattr(utility, "embed", lambda method, items: items.flatten(1))

    class Probe:
        """Return a deterministic readout for the isolation test."""

        def losses(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Keep the fixed readout independent of reconstruction randomness."""
            return features[:, 0] + labels

    method = ScalarMethod()
    probe = cast(utility.FixedProbe, Probe())
    rng = torch.get_rng_state().clone()
    full = utility.outcome_losses(method, evaluation, roles, probe, 3)
    primary = utility.outcome_losses(method, evaluation, roles, probe, 3, reconstruction=False)
    for role in roles:
        assert primary[role] == {key: value for key, value in full[role].items() if key.startswith("fixed_ce")}
        assert len(primary[role]) == 3
    assert torch.equal(rng, torch.get_rng_state())


def test_real_gradient_reader_uses_full_monitor_and_preserves_existing_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grouped monitor must not be truncated, and reading gradients must not mutate learning state."""
    torch.set_num_threads(2)
    method = build_mae(TinyViTEncoder(img_size=16, embed_dim=12, depth=1, num_heads=3), decoder_dim=12)
    method.train()
    method.encoder.blocks[-1].eval()
    monitor = torch.cat((torch.zeros(32, 3, 16, 16), torch.ones(32, 3, 16, 16)))
    warm = utility.embed(method, monitor)
    actuator = SteerableReplay(method.encoder, torch.randn(2, 12), torch.tensor([0.5, 0.5]))
    for image in torch.rand(32, 3, 16, 16):
        actuator.reservoir._ingest(image)
    for parameter in method.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {key: value.clone() for key, value in method.state_dict().items()}
    rng = torch.get_rng_state().clone()
    sizes = []
    original = utility.gradient_vector

    def read(current: SSLMethod, images: torch.Tensor, seed: int) -> torch.Tensor:
        sizes.append(len(images))
        return original(current, images, seed)

    monkeypatch.setattr(utility, "gradient_vector", read)
    rows, _ = utility.utility_features(
        method, actuator, monitor[[0, 40]], monitor, warm, draws=1, gradient_signals=True
    )
    assert sizes == [64, 1, 1]
    assert all(np.isfinite(row["gradient_monitor_cosine"]) for row in rows)
    assert all(torch.equal(before[key], value) for key, value in method.state_dict().items())
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in method.parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert method.training and not method.encoder.blocks[-1].training
