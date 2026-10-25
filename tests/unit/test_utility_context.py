"""Verify read-only context instruments, matched features and chronological label availability."""

import numpy as np
import torch

from cafl4ds.filters.steerable import SteerableReplay
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_mae
from cafl4ds.utility_context import (
    ARRIVAL,
    HISTORY,
    IMAGE_CONTEXT,
    arrival_context,
    center_states,
    context_matrix,
    rolling_split,
)
from cafl4ds.utility_prediction import feature_matrix


def test_context_reader_preserves_learning_memory_rng_modes_and_gradients() -> None:
    """A context measurement cannot alter the prefix subsequently used for paired interventions."""
    torch.set_num_threads(2)
    method = build_mae(TinyViTEncoder(img_size=16, embed_dim=12, depth=1, num_heads=3), decoder_dim=12)
    method.train()
    method.encoder.blocks[-1].eval()
    actuator = SteerableReplay(method.encoder, torch.randn(2, 12), torch.tensor([0.5, 0.5]))
    for image in torch.rand(48, 3, 16, 16):
        actuator.reservoir._ingest(image)
    for parameter in method.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {key: value.clone() for key, value in method.state_dict().items()}
    memory = [image.clone() for image in actuator.reservoir._buffer]
    generator = actuator.reservoir._gen.get_state().clone()
    occupancy, seen = actuator.occupancy.clone(), actuator.reservoir._seen
    batch = torch.rand(32, 3, 16, 16)
    rng = torch.get_rng_state().clone()
    signals = [{"reconstruction_mean": 0.2}, {"reconstruction_mean": 0.4}]
    first = arrival_context(method, actuator, batch, [0, 17], signals, 1, None, 1)
    assert signals[0]["context_gap_change"] == signals[0]["context_anchor_turnover"] == 0
    second_signals = [{"reconstruction_mean": 0.2}, {"reconstruction_mean": 0.4}]
    second = arrival_context(method, actuator, batch * 0.8, [0, 17], second_signals, 5, first, 1)
    assert np.isclose(
        second_signals[0]["context_gap_change"], (second.reconstruction_gap - first.reconstruction_gap) / 4
    )
    assert all(signals[0][name] == signals[1][name] for name in ARRIVAL + HISTORY)
    assert signals[0]["context_image_reconstruction_z"] != signals[1]["context_image_reconstruction_z"]
    assert all(torch.equal(before[key], value) for key, value in method.state_dict().items())
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in method.parameters())
    assert all(torch.equal(a, b) for a, b in zip(memory, actuator.reservoir._buffer, strict=True))
    assert torch.equal(generator, actuator.reservoir._gen.get_state())
    assert torch.equal(occupancy, actuator.occupancy) and seen == actuator.reservoir._seen
    assert torch.equal(rng, torch.get_rng_state())
    assert method.training and not method.encoder.blocks[-1].training


def test_context_columns_cannot_silently_enrich_matched_image_baselines() -> None:
    """Relative image inputs, arrival context and history have separate matched contrasts."""
    features = dict.fromkeys(ARRIVAL + HISTORY + IMAGE_CONTEXT, 0.2)
    features.update(reconstruction_mean=0.4, coverage_deficit=0.1, embedding_drift=0.3, health_cosine_drift=0.05)
    rows = [{"seed": 1, "step": 1, "features": features}]
    _, image_names = feature_matrix(rows, "image_only")
    assert not any(name.startswith("context_") for name in image_names)
    relative, names = context_matrix(rows, "loss_coverage_relative")
    assert relative.shape == (1, 4) and names[-2:] == IMAGE_CONTEXT
    _, arrival_names = context_matrix(rows, "loss_coverage_arrival")
    _, history_names = context_matrix(rows, "loss_coverage_history")
    assert all(name not in arrival_names for name in HISTORY)
    assert all(name not in history_names for name in ARRIVAL)


def test_centering_is_invariant_to_arbitrary_state_offsets_and_does_not_mix_seeds() -> None:
    """Shared reference losses and state-only features cannot masquerade as image ordering signals."""
    rows = [{"seed": seed, "step": 1} for seed in (1, 1, 2, 2)]
    values = np.array([[1.0, 10.0], [3.0, 10.0], [4.0, 90.0], [8.0, 90.0]])
    shifted = values + np.array([[100.0, 2.0], [100.0, 2.0], [-50.0, 9.0], [-50.0, 9.0]])
    assert np.array_equal(center_states(rows, values), center_states(rows, shifted))
    assert np.array_equal(center_states(rows, values)[:, 1], np.zeros(4))
    assert np.array_equal(center_states(rows, np.array([1.0, 3.0, 4.0, 8.0])), [-1.0, 1.0, -2.0, 2.0])


def test_rolling_split_purges_future_outcomes_and_holds_out_seed_and_new_images() -> None:
    """A label whose update window ends after validation/test begins is unavailable for fitting."""
    steps = [1, 4, 8, 10, 12, 16, 20, 22, 24, 28, 32, 34, 36, 40, 44]
    rows = [
        {"seed": seed, "step": step, "arrival_id": step * 32 + candidate}
        for seed in (1, 2, 3, 4)
        for step in steps
        for candidate in range(2)
    ]
    for origin in (12, 24, 36):
        split = rolling_split(rows, [1, 2, 3, 4], origin, origin + 12, 4)
        first_validation = min(rows[i]["step"] for i in split["validation"])
        assert all(rows[i]["step"] + 4 <= first_validation for i in split["fit"])
        assert all(rows[i]["step"] + 4 <= origin for i in split["validation"])
        assert all(origin <= rows[i]["step"] and rows[i]["step"] + 4 <= origin + 12 for i in split["test_time"])
        assert all(rows[i]["seed"] == 4 for i in split["test_seed"])
        assert not {rows[i]["arrival_id"] for i in split["test_time"]} & {
            rows[i]["arrival_id"] for i in split["fit"] + split["validation"]
        }
        assert not set(split["test_seed"]) & set(split["test_time"])
