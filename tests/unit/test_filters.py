"""Tests for the Phase-0 selection knobs (the ``A`` factor) and their composition.

The knobs are deliberately *stock* parts, so the tests pin their known-answer behaviour:
SemDeDup greedy near-duplicate removal, the loss-gate's top-fraction selection, the reservoir's
Algorithm-R uniformity + replay, and the composite's admission-then-buffer ordering. A tiny stub
stands in for the live model so the embedding / per-sample-loss inputs are controlled exactly.
"""

from __future__ import annotations

import pytest
import torch

from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.accept_all import AcceptAll
from cafl4ds.filters.base import Filter, FilterContext
from cafl4ds.filters.composite import CompositeSelector
from cafl4ds.filters.dedup import SemDeDup, semantic_dedup_keep
from cafl4ds.filters.loss_gate import LossGate
from cafl4ds.filters.reservoir import FeatureDistillationReservoirReplay, ReservoirReplay


class _StubMethod:
    """Stand-in for an SSLMethod with fixed ``encode`` / ``per_sample_loss`` outputs."""

    def __init__(self, embeddings: torch.Tensor | None = None, losses: torch.Tensor | None = None) -> None:
        self._embeddings = embeddings
        self._losses = losses

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self._embeddings if self._embeddings is not None else images.reshape(images.shape[0], -1)

    def per_sample_loss(self, images: torch.Tensor) -> torch.Tensor:
        assert self._losses is not None
        return self._losses


def _ctx(method: object) -> FilterContext:
    """A filter context wrapping an arbitrary (duck-typed) method stub."""
    return FilterContext(method=method, step=0)  # type: ignore[arg-type]


def _tagged_batch(n: int, era: int = 0) -> StreamBatch:
    """A batch whose every pixel equals the row index, so a row's identity is recoverable."""
    imgs = torch.stack([torch.full((3, 4, 4), float(i)) for i in range(n)])
    return StreamBatch(images=imgs, era=era, step=0)


# --- SemDeDup ------------------------------------------------------------------------------


def test_semantic_dedup_keep_drops_only_near_duplicates() -> None:
    """Greedy dedup keeps the earliest of each near-duplicate group; distinct points survive."""
    a = torch.tensor([1.0, 0.0, 0.0])
    near = torch.tensor([1.0, 0.02, 0.0])  # cosine ~1 with a -> dropped
    b = torch.tensor([0.0, 1.0, 0.0])  # orthogonal to a -> kept
    z = torch.stack([a, near, b])
    keep = semantic_dedup_keep(z, threshold=0.9)
    assert keep.tolist() == [0, 2]


def test_semantic_dedup_keep_edges() -> None:
    """All-identical collapses to one; all-orthogonal keeps everything; n<=1 is a no-op."""
    identical = torch.ones(4, 3)
    assert semantic_dedup_keep(identical, threshold=0.9).tolist() == [0]
    ortho = torch.eye(3)
    assert semantic_dedup_keep(ortho, threshold=0.9).tolist() == [0, 1, 2]
    assert semantic_dedup_keep(torch.ones(1, 3), threshold=0.9).tolist() == [0]


def test_semdedup_filter_returns_deduplicated_images() -> None:
    """The filter maps its embedding-space keep-set back onto the incoming images."""
    embeddings = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.01]), torch.tensor([0.0, 1.0])])
    batch = _tagged_batch(3)
    dedup = SemDeDup(threshold=0.9)
    kept = dedup.select(batch, _ctx(_StubMethod(embeddings=embeddings)))
    assert kept.shape[0] == 2
    assert [int(img[0, 0, 0]) for img in kept] == [0, 2]  # rows 0 and 2 survived, in order
    assert dedup.stats() == {"seen_images": 3, "kept_images": 2, "keep_fraction": 2 / 3}


# --- loss-gate -----------------------------------------------------------------------------


def test_loss_gate_keeps_highest_loss_fraction_in_order() -> None:
    """keep_frac=0.5 keeps the top-half by per-sample loss, restored to stream order."""
    losses = torch.tensor([0.1, 0.9, 0.5, 0.2])  # top-2 are rows 1 and 2
    kept = LossGate(keep_frac=0.5).select(_tagged_batch(4), _ctx(_StubMethod(losses=losses)))
    assert [int(img[0, 0, 0]) for img in kept] == [1, 2]


def test_loss_gate_keep_all_is_a_noop_without_scoring() -> None:
    """keep_frac=1.0 returns the whole batch without consulting the model (no losses needed)."""
    batch = _tagged_batch(5)
    kept = LossGate(keep_frac=1.0).select(batch, _ctx(_StubMethod()))  # no losses set
    assert torch.equal(kept, batch.images)


def test_loss_gate_rejects_bad_fraction() -> None:
    """keep_frac must be in (0, 1]."""
    for bad in (0.0, 1.5, -0.1):
        try:
            LossGate(keep_frac=bad)
        except ValueError:
            continue
        raise AssertionError(f"keep_frac={bad} should have raised")


# --- reservoir -----------------------------------------------------------------------------


def test_reservoir_replays_past_and_keeps_incoming_first() -> None:
    """observe() returns incoming frames plus a replay draw from prior history."""
    res = ReservoirReplay(capacity=8, replay_batch=2, seed=0)
    first = res.observe(_tagged_batch(3).images, _ctx(_StubMethod()))
    assert first.shape[0] == 3  # buffer was empty -> no replay on the first step
    second = res.observe(_tagged_batch(2).images, _ctx(_StubMethod()))
    assert second.shape[0] == 4  # 2 incoming + 2 replayed
    assert [int(img[0, 0, 0]) for img in second[:2]] == [0, 1]  # incoming kept first, in order
    assert all(int(img[0, 0, 0]) in {0, 1, 2} for img in second[2:])  # replay drawn from history


def test_reservoir_algorithm_r_is_uniform() -> None:
    """Over many trials each of N frames survives a size-C reservoir with frequency ~ C/N."""
    capacity, n_items, trials = 5, 50, 400
    counts = torch.zeros(n_items)
    for seed in range(trials):
        res = ReservoirReplay(capacity=capacity, replay_batch=1, seed=seed)
        for i in range(n_items):
            res._ingest(torch.full((3, 2, 2), float(i)))
        for item in res._buffer:
            counts[int(item[0, 0, 0])] += 1
    assert int(counts.sum()) == capacity * trials  # buffer is always exactly full
    expected = capacity * trials / n_items  # = 40
    # Deterministic (seeds 0..trials-1); a generous band around the uniform expectation.
    assert counts.min() >= 0.55 * expected
    assert counts.max() <= 1.45 * expected


def test_reservoir_compute_cap_displaces_current_but_ingests_everything() -> None:
    """Compute-matched replay holds training size fixed without hiding arrivals from storage."""
    res = ReservoirReplay(capacity=16, replay_batch=2, max_train_batch=4, seed=0)
    first = res.observe(_tagged_batch(4).images, _ctx(_StubMethod()))
    assert first.shape[0] == 4
    second = res.observe(_tagged_batch(4).images + 10, _ctx(_StubMethod()))
    assert second.shape[0] == 4  # two current + two replay, rather than four + two
    assert all(int(img[0, 0, 0]) >= 10 for img in second[:2])
    assert all(int(img[0, 0, 0]) < 10 for img in second[2:])
    assert len(res._buffer) == 8  # all new arrivals were ingested, not only the trained two


def test_reservoir_compute_cap_rejects_impossible_budget() -> None:
    """The replay allocation cannot exceed the total compute-matched batch budget."""
    try:
        ReservoirReplay(capacity=8, replay_batch=5, max_train_batch=4)
    except ValueError:
        return
    raise AssertionError("replay_batch > max_train_batch should have raised")


class _EmbeddingEncoder(torch.nn.Module):  # type: ignore[misc]
    """Small differentiable encoder for cached-feature replay tests."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images.flatten(1))


class _FeatureMethod:
    def __init__(self) -> None:
        self.encoder = _EmbeddingEncoder()


def test_feature_distillation_replay_caches_aligned_targets_and_gradients() -> None:
    """Replay targets stay aligned with images and produce a differentiable drift penalty."""
    torch.manual_seed(4)
    method = _FeatureMethod()
    replay = FeatureDistillationReservoirReplay(
        capacity=8,
        replay_batch=2,
        max_train_batch=4,
        distill_weight=2.0,
        seed=3,
    )
    first = torch.randn(4, 1, 2, 2)
    assert replay.observe(first, _ctx(method)).shape[0] == 4
    second = torch.randn(4, 1, 2, 2)
    training = replay.observe(second, _ctx(method))
    assert training.shape[0] == 4

    with torch.no_grad():
        method.encoder.proj.weight[0, 0] += 0.5
    penalty = replay.auxiliary_loss(training, _ctx(method))
    assert penalty is not None and float(penalty.detach()) > 0.0
    penalty.backward()
    assert method.encoder.proj.weight.grad is not None
    stats = replay.stats()
    assert stats["distill_steps"] == 1
    assert stats["replayed_images"] == 2


def test_feature_distillation_replay_rejects_negative_weight() -> None:
    """A negative regularization weight is invalid rather than silently destabilizing training."""
    with pytest.raises(ValueError, match="distill_weight"):
        FeatureDistillationReservoirReplay(distill_weight=-0.1)


# --- composition ---------------------------------------------------------------------------


class _KeepIndices(Filter):
    """A trivial admission filter that keeps a fixed set of row indices (test double)."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def select(self, batch: StreamBatch, ctx: FilterContext) -> torch.Tensor:
        del ctx
        return batch.images[torch.tensor(self.indices, dtype=torch.long)]


def test_composite_empty_is_b_floor() -> None:
    """No admission + no buffer is the B-floor: the batch passes through unchanged."""
    batch = _tagged_batch(4)
    out = CompositeSelector().select(batch, _ctx(_StubMethod()))
    assert torch.equal(out, batch.images)


def test_composite_chains_admission_then_buffer() -> None:
    """Admission narrows first; the buffer then stores/replays only the admitted frames."""
    res = ReservoirReplay(capacity=8, replay_batch=2, seed=0)
    comp = CompositeSelector(admission=[_KeepIndices([0, 2])], buffer=res)
    out = comp.select(_tagged_batch(4), _ctx(_StubMethod()))
    assert [int(img[0, 0, 0]) for img in out] == [0, 2]  # admitted rows only, no replay yet
    # The buffer stored the admitted frames (0 and 2), not the dropped ones.
    assert {int(x[0, 0, 0]) for x in res._buffer} == {0, 2}


def test_composite_admission_composes_in_order() -> None:
    """Two admission filters apply in sequence, each narrowing what the next sees."""
    comp = CompositeSelector(admission=[_KeepIndices([0, 1, 2]), _KeepIndices([1])])
    out = comp.select(_tagged_batch(4), _ctx(_StubMethod()))
    # first keeps rows {0,1,2}; second keeps local index 1 of that -> original row 1.
    assert [int(img[0, 0, 0]) for img in out] == [1]


def test_accept_all_is_identity_admission() -> None:
    """The B-floor AcceptAll composes as a pass-through admission filter."""
    batch = _tagged_batch(3)
    out = CompositeSelector(admission=[AcceptAll()]).select(batch, _ctx(_StubMethod()))
    assert torch.equal(out, batch.images)
