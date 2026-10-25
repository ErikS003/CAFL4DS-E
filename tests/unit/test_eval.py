"""Tests for the downstream / forgetting evaluation (probe-on-past, BWT, Forgetting, B5).

The forgetting metrics are pinned with a hand-built accuracy matrix (known answer); the
per-era probe and the B5 comparison are exercised end-to-end on separable synthetic clusters
with a trivial flatten "encoder", and the loop hook is checked to populate the matrix on a
class-blocked stream.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EraStream, EvalSet, EvalSets
from cafl4ds.eval import PerEraProbe, adaptation_report, backward_transfer, few_shot_probe_curve, forgetting_measure
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.loop import StreamingLoop
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.monitor import HealthMonitor
from cafl4ds.run_log import RunLogger
from cafl4ds.ssl.factory import build_simsiam

# A 3-era matrix that forgets: era 0 decays 1.0 -> 0.5 -> 0.2, era 1 decays 1.0 -> 0.6.
_MATRIX = {0: {0: 1.0}, 1: {0: 0.5, 1: 1.0}, 2: {0: 0.2, 1: 0.6, 2: 1.0}}


def _flatten(images: torch.Tensor) -> torch.Tensor:
    """A trivial 'encoder': flatten pixels to a feature vector (separable clusters stay separable)."""
    return images.reshape(images.shape[0], -1)


def _zeros_encode(images: torch.Tensor) -> torch.Tensor:
    """A degenerate encoder mapping everything to the same point (chance-level probe)."""
    return torch.zeros(images.shape[0], 8)


def test_backward_transfer_known_answer() -> None:
    """BWT = mean over past eras of (final acc - first-learned acc)."""
    # ((0.2 - 1.0) + (0.6 - 1.0)) / 2 = -0.6
    assert backward_transfer(_MATRIX) == pytest.approx(-0.6)


def test_forgetting_measure_known_answer() -> None:
    """FM = mean over past eras of (best-ever acc - final acc)."""
    # era0: max(1.0, 0.5) - 0.2 = 0.8 ; era1: 1.0 - 0.6 = 0.4 ; mean = 0.6
    assert forgetting_measure(_MATRIX) == pytest.approx(0.6)


def test_positive_backward_transfer() -> None:
    """When later learning *raises* past-era accuracy, BWT is positive and FM ~ 0."""
    grew = {0: {0: 0.4}, 1: {0: 0.7, 1: 0.8}}
    assert backward_transfer(grew) == pytest.approx(0.3)
    assert forgetting_measure(grew) == pytest.approx(-0.3)  # "negative forgetting" = improvement


def test_forgetting_undefined_for_single_era() -> None:
    """With one recorded era there is no past to forget: metrics are None."""
    assert backward_transfer({0: {0: 1.0}}) is None
    assert forgetting_measure({}) is None


def _synthetic_stream() -> EraStream:
    """A small class-blocked synthetic stream with per-era held-out eval sets."""
    return EraStream(
        SyntheticSource(num_classes=3, per_class=48, img_size=16),
        batch_size=12,
        order="class_blocked",
        support_per_class=8,
        query_per_class=8,
        era_eval_per_class=6,
    )


def test_per_era_probe_builds_a_lower_triangular_matrix() -> None:
    """Recording a row per era yields accuracy on every era seen so far, then a summary."""
    stream = _synthetic_stream()
    probe = PerEraProbe(stream.eval_sets, probe="knn", knn_k=5)
    for era in range(3):
        row = probe.record(_flatten, era)
        assert set(row) == set(range(era + 1))  # eras 0..era scored
    summary = probe.summary()
    assert summary["num_eras"] == 3
    assert set(summary["per_era_final"]) == {0, 1, 2}
    assert isinstance(summary["backward_transfer"], float)
    assert isinstance(summary["forgetting_measure"], float)


def test_adaptation_report_prefers_the_informative_encoder() -> None:
    """A separable-feature encoder beats the degenerate (B5-style) one on the downstream probe."""
    stream = _synthetic_stream()
    report = adaptation_report(_flatten, _zeros_encode, stream.eval_sets, probe="knn", knn_k=5)
    assert report["adapted_acc"] > report["b5_acc"]
    assert report["gain"] == pytest.approx(report["adapted_acc"] - report["b5_acc"])


def test_few_shot_probe_curve_uses_balanced_nested_support() -> None:
    """Every point contains exactly the requested number of examples from each class."""
    support = EvalSet(
        images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
        labels=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    query = EvalSet(images=torch.tensor([[0.95, 0.05], [0.05, 0.95]]), labels=torch.tensor([0, 1]))
    curve = few_shot_probe_curve(_flatten, EvalSets(support, query), shots=[1, 2, 3], probe="knn", knn_k=1)
    assert curve == {1: 1.0, 2: 1.0, 3: 1.0}


def test_few_shot_probe_curve_rejects_unavailable_shots() -> None:
    """A requested support size must exist for every class."""
    support = EvalSet(images=torch.tensor([[0.0], [1.0]]), labels=torch.tensor([0, 1]))
    query = EvalSet(images=torch.tensor([[0.0], [1.0]]), labels=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="fewer than requested 2-shot"):
        few_shot_probe_curve(_flatten, EvalSets(support, query), shots=[2])


def test_loop_populates_probe_on_past(tmp_path: Path) -> None:
    """The loop's era hook fills the accuracy matrix across a class-blocked run."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=32, depth=2, num_heads=2)
    stream = _synthetic_stream()
    evaluator = PerEraProbe(stream.eval_sets, probe="knn", knn_k=5)
    StreamingLoop(
        stream=stream,
        method=build_simsiam(encoder),
        optimizer=torch.optim.AdamW(encoder.parameters(), lr=1e-3),
        selection_filter=AcceptAll(),
        monitor=HealthMonitor(stream.eval_sets, knn_k=5, run_knn=False, run_linear=False),
        run_logger=RunLogger(str(tmp_path / "loop.jsonl"), run_name="probe"),
        eval_every=3,
        era_evaluator=evaluator,
    ).run()
    # Three class-blocked eras -> three recorded rows, and the forgetting metrics are defined.
    assert set(evaluator.matrix) == {0, 1, 2}
    assert evaluator.summary()["backward_transfer"] is not None
