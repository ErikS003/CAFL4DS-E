"""Check action leverage, historical replay, paired state and robust artifact replacement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from PIL import Image
from torch import nn

from cafl4ds.artifacts import atomic_json
from cafl4ds.data.attributes import BDD100KSource
from cafl4ds.data.bdd_subset import BDDSubsetIndex
from cafl4ds.data.streams import StreamBatch
from cafl4ds.filters.base import FilterContext
from cafl4ds.filters.steerable import SteerableReplay, fit_anchors, midranks
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.base import SSLMethod


class ToyEncoder(nn.Module):
    """Read two orthogonal clusters with identifiable sample tags."""

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Return the first two channels, preserving cluster membership."""
        return images[:, :2, 0, 0]


class ToyMethod(SSLMethod):
    """Prefer the first cluster while drawing stochastic scoring noise."""

    def __init__(self) -> None:
        """Construct a controllable novelty proxy."""
        super().__init__(cast(TinyViTEncoder, ToyEncoder()))

    @property
    def name(self) -> str:
        """Identify the deterministic test plant."""
        return "toy"

    def training_step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Unused training interface."""
        return imgs.mean()

    def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        """Make novelty oppose the under-covered cluster."""
        return imgs[:, 0, 0, 0] + 0.01 * torch.rand(len(imgs))


def arrivals(offset: int = 0) -> torch.Tensor:
    """Two equal clusters, unique sample tags in channel three."""
    images = torch.zeros(32, 3, 1, 1)
    images[:16, 0] = 1
    images[16:, 1] = 1
    images[:, 2, 0, 0] = torch.arange(32) + offset
    return images


def selector() -> SteerableReplay:
    """Construct the documented exact-work actuator."""
    return SteerableReplay(cast(TinyViTEncoder, ToyEncoder()), torch.eye(2), torch.tensor([0.5, 0.5]), seed=7)


def test_endpoints_reach_distinct_diets_with_identical_memory_and_work() -> None:
    """Coverage and novelty select opposite clusters; every rejected arrival is stored."""
    method = ToyMethod()
    choices = [selector(), selector()]
    for choice in choices:
        first = choice.select(StreamBatch(arrivals(), 0, 0), FilterContext(method, 0))
        assert len(first) == 32
        assert choice.last["replay_count"] == 0
        choice.occupancy = torch.tensor([0.9, 0.1])
    rng = torch.get_rng_state().clone()
    outputs = []
    for alpha, choice in zip((0, 1), choices, strict=True):
        choice.set_action(alpha)
        outputs.append(choice.select(StreamBatch(arrivals(100), 0, 1), FilterContext(method, 1)))
        assert torch.equal(rng, torch.get_rng_state())
        assert choice.reservoir._seen == 64
        assert len(choice.reservoir._buffer) == 64
        assert choice.optimizer_images == 64
        assert choice.last["current_count"] == choice.last["replay_count"] == 16
        assert (outputs[-1][16:, 2] < 32).all()  # strictly historical
    assert choices[0].last["current_indices"] == list(range(16))
    assert choices[1].last["current_indices"] == list(range(16, 32))
    assert torch.equal(outputs[0][16:], outputs[1][16:])
    assert torch.equal(torch.stack(choices[0].reservoir._buffer), torch.stack(choices[1].reservoir._buffer))
    assert method.training and method.encoder.training


def test_snapshot_reproduces_next_selection_and_is_detached() -> None:
    """Mutable memory, occupancy and RNG round-trip without aliases."""
    method, choice = ToyMethod(), selector()
    choice.select(StreamBatch(arrivals(), 0, 0), FilterContext(method, 0))
    snapshot = choice.state_dict()
    restored = selector()
    restored.load_state_dict(snapshot)
    snapshot["buffer"][0].zero_()
    expected = choice.select(StreamBatch(arrivals(100), 0, 1), FilterContext(method, 1))
    actual = restored.select(StreamBatch(arrivals(100), 0, 1), FilterContext(method, 1))
    assert torch.equal(expected, actual)
    assert torch.equal(choice.occupancy, restored.occupancy)
    for action in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            choice.set_action(action)
    with pytest.raises(ValueError, match="full incoming"):
        choice.select(StreamBatch(arrivals()[:5], 0, 2), FilterContext(method, 2))


def test_midranks_and_anchor_assignments() -> None:
    """Equal components cannot invent an alpha preference; anchor targets match cosine routing."""
    assert torch.allclose(midranks(torch.tensor([0.0, 0.0, 1.0])), torch.tensor([0.25, 0.25, 1.0]))
    assert torch.equal(midranks(torch.ones(3)), torch.full((3,), 0.5))
    assert torch.equal(midranks(torch.ones(1)), torch.tensor([0.5]))
    centers, target = fit_anchors(torch.eye(2).repeat(12, 1), count=2)
    assert torch.equal(target, torch.tensor([0.5, 0.5]))
    assert centers.shape == (2, 2)
    with pytest.raises(ValueError):
        midranks(torch.tensor([float("nan")]))


def test_reservoir_remains_paired_after_capacity_and_snapshot_rejects_wrong_spec() -> None:
    """Rejected arrivals remain eligible for replacement once the reservoir is full."""
    first, second, method = selector(), selector(), ToyMethod()
    first.set_action(0)
    second.set_action(1)
    for step in range(12):
        batch = StreamBatch(arrivals(100 * step), 0, step)
        for choice in (first, second):
            choice.select(batch, FilterContext(method, step))
    assert first.reservoir._seen == second.reservoir._seen == 384
    assert len(first.reservoir._buffer) == len(second.reservoir._buffer) == 256
    assert torch.equal(torch.stack(first.reservoir._buffer), torch.stack(second.reservoir._buffer))
    wrong = first.state_dict()
    wrong["specification"]["seed"] += 1
    with pytest.raises(ValueError, match="specification mismatch"):
        second.load_state_dict(wrong)


def test_scoring_failure_restores_rng_and_model_modes() -> None:
    """A bad novelty signal cannot consume global draws or leave the learner in eval mode."""

    class FailingMethod(ToyMethod):
        def per_sample_loss(self, imgs: torch.Tensor) -> torch.Tensor:
            """Consume private draws before simulating a scoring failure."""
            torch.rand(10)
            raise RuntimeError("failed score")

    method = FailingMethod()
    method.train()
    method.encoder.eval()  # deliberately mixed modes
    choice = selector()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="failed score"):
        choice.select(StreamBatch(arrivals(), 0, 0), FilterContext(method, 0))
    assert torch.equal(rng, torch.get_rng_state())
    assert method.training and not method.encoder.training
    assert choice.reservoir._seen == 0 and choice.optimizer_images == 0


def test_atomic_json_retries_and_preserves_previous_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient sharing locks retry, persistent failures leave the old artifact intact."""
    path = tmp_path / "progress.json"
    atomic_json(path, {"step": 0})
    original = Path.replace
    attempts = 0

    def transient(self: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("sharing violation")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr("cafl4ds.artifacts.time.sleep", lambda _: None)
    atomic_json(path, {"step": 1})
    assert attempts == 3 and json.loads(path.read_text())["step"] == 1

    def persistent(self: Path, target: Path) -> Path:
        raise PermissionError("persistent lock")

    monkeypatch.setattr(Path, "replace", persistent)
    with pytest.raises(PermissionError):
        atomic_json(path, {"step": 2})
    assert json.loads(path.read_text())["step"] == 1
    assert list(tmp_path.iterdir()) == [path]


def test_nonfinite_json_never_replaces_previous_artifact(tmp_path: Path) -> None:
    """Scientific failures cannot be serialized as apparently valid numeric results."""
    path = tmp_path / "result.json"
    atomic_json(path, {"value": 0})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text())["value"] == 0


def test_bdd_subset_preserves_full_partition_and_rare_count_rules(tmp_path: Path) -> None:
    """Decode fewer images without changing warm membership or filtering sparse subsets twice."""
    images = tmp_path / "images/100k/train"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    records = []
    for domain in ("daytime", "night"):
        for scene in ("city street", "highway"):
            for _ in range(10):
                name = f"{len(records)}.png"
                Image.new("RGB", (4, 4), (len(records), 0, 0)).save(images / name)
                records.append({"name": name, "attributes": {"timeofday": domain, "weather": "clear", "scene": scene}})
    (labels / "bdd100k_labels_images_train.json").write_text(json.dumps(records), encoding="utf-8")
    kwargs = {
        "bdd_root": tmp_path,
        "split": "train",
        "img_size": 16,
        "min_canary_count": 8,
        "warm_fraction": 0.2,
        "partition_seed": 19,
    }
    warm_source = BDD100KSource(**kwargs, partition_role="warm")
    stream_source = BDD100KSource(**kwargs, partition_role="stream")
    index = BDDSubsetIndex()
    warm, warm_paths = index.load(warm_source, seed=31, total=8)
    subset, subset_paths = index.load(stream_source, seed=31, per_cell=4)
    full_warm = warm_source.load()
    full_stream = stream_source.load()
    assert len(warm.images) == len(full_warm.images) == 8
    assert len(subset.images) == 16 < len(full_stream.images)
    assert not set(warm_paths) & set(subset_paths)
    assert set(warm.images[:, 0, 0, 0].tolist()) == set(full_warm.images[:, 0, 0, 0].tolist())
    assert set(subset.images[:, 0, 0, 0].tolist()) <= set(full_stream.images[:, 0, 0, 0].tolist())
    assert len(index.records) == 1  # warm and stream share complete train metadata
    fresh, fresh_paths = index.load(stream_source, seed=37, per_cell=1, excluded_paths=set(subset_paths))
    assert len(fresh.images) == 4
    assert not set(fresh_paths) & (set(subset_paths) | set(warm_paths))
    stream_source.max_images = 20
    with pytest.raises(ValueError, match="uncapped"):
        index.load(stream_source, seed=31, per_cell=4)
