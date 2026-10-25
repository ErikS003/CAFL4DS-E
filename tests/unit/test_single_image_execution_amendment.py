"""Removing independent baselines must preserve retained scientific policies."""

import json

import pytest

from cafl4ds.single_image_execution_amendment import active_specs, capped_histories, validate_hashes
from cafl4ds.single_image_long_gate import policy_specs
from scripts.run_short_buffer_experiment import digest


def test_retirement_keeps_every_retained_policy_object(tmp_path):
    specs = {name: {"kind": "independent"} for name in ("independent_10", "independent_25", "independent_50")}
    specs.update(gate={"kind": "gate", "threshold": -0.1}, matched={"kind": "clip"})
    (tmp_path / "execution-amendment.json").write_text(
        json.dumps({"retired_policies": ["independent_10", "independent_50"]})
    )
    active = active_specs(tmp_path, specs)
    assert set(active) == {"independent_25", "gate", "matched"}
    assert all(active[name] is specs[name] for name in active)


def test_retirement_cannot_remove_a_gate(tmp_path):
    (tmp_path / "execution-amendment.json").write_text(json.dumps({"retired_policies": ["gate"]}))
    with pytest.raises(ValueError):
        active_specs(tmp_path, {"gate": {"kind": "gate"}})


def test_second_retirement_keeps_exactly_gates_full_and_25(tmp_path):
    specs = policy_specs(dict(threshold_factors=[1, 0.5, 0], independent_probabilities=[0.1, 0.25, 0.5]), -0.01)
    keep = {"gate_original", "gate_half", "gate_zero", "full", "independent_25"}
    (tmp_path / "execution-amendment.json").write_text(json.dumps({"retired_policies": sorted(set(specs) - keep)}))
    result = active_specs(tmp_path, specs)
    assert set(result) == keep
    assert all(result[name] is specs[name] for name in result)


def test_source_amendment_cannot_override_data_hashes(tmp_path):
    data = tmp_path / "data.npy"
    data.write_bytes(b"unchanged data")
    original = digest(data)
    data.write_bytes(b"modified data")
    (tmp_path / "execution-amendment.json").write_text(
        json.dumps({"source_overrides": {str(data): {"original_sha256": original, "amended_sha256": digest(data)}}})
    )
    with pytest.raises(ValueError):
        validate_hashes(tmp_path, {str(data): original})


def test_cap_30_preserves_five_policies_both_seeds_and_three_histories(tmp_path):
    histories = {f"history_{i}": {"frames": i} for i in range(6)}
    (tmp_path / "execution-amendment.json").write_text(json.dumps({"max_completed_trajectories": 30}))
    result = capped_histories(tmp_path, histories, policy_count=5, seed_count=2)
    assert list(result) == ["history_0", "history_1", "history_2"]
    assert all(result[name] is histories[name] for name in result)
    assert len(result) * 5 * 2 == 30


@pytest.mark.parametrize("cap", [0, 29, 31, 70])
def test_cap_rejects_partial_or_excess_history_comparisons(tmp_path, cap):
    (tmp_path / "execution-amendment.json").write_text(json.dumps({"max_completed_trajectories": cap}))
    with pytest.raises(ValueError):
        capped_histories(tmp_path, dict.fromkeys(range(6)), policy_count=5, seed_count=2)
