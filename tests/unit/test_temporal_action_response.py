"""Protect complete parent isolation, fixed arrivals and independent action auditing."""

import copy
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from cafl4ds.temporal_action_response import restore, run_action, snapshot
from scripts.report_paired_update_response import preference
from scripts.report_temporal_action_response import verify_rows
from scripts.run_paired_update_response import fingerprint


def test_restore_does_not_alias_parent_adam_or_initial_weights() -> None:
    """A real Adam step in one branch cannot mutate its sibling or parent."""
    initial = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(initial.parameters(), lr=0.01)
    initial(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    initial.eval()
    parent = snapshot(initial, optimizer, 1, 64)
    marker = fingerprint(parent)
    config = OmegaConf.create({"optim": {"_target_": "torch.optim.AdamW", "lr": 0.01}})
    model, restored = restore(initial, config, parent)
    assert fingerprint(snapshot(model, restored, 1, 64)) == marker
    sibling, sibling_optim = restore(initial, config, parent)
    model(torch.ones(1, 2)).sum().backward()
    restored.step()
    assert fingerprint(parent) == marker
    assert fingerprint(snapshot(sibling, sibling_optim, 1, 64)) == marker
    assert fingerprint(snapshot(model, restored, 1, 64)) != marker


@pytest.mark.parametrize("action", ["full", "half"])
def test_same_future_frames_and_exact_action_work(action: str) -> None:
    """Actions consume the same arrival horizon with independent reconstructible masks."""

    def observe(
        _model: object,
        _optimizer: object,
        policy: str,
        _images: torch.Tensor,
        start: int,
        _generator: torch.Generator,
    ) -> dict[str, object]:
        positions = list(range(64)) if policy == "full" else list(range(1, 64, 2))
        updates = 2 if policy == "full" else 1
        return {
            "start_frame": start,
            "end_frame_exclusive": start + 64,
            "selected_positions": positions,
            "selected_global_indices": [start + p for p in positions],
            "accepted_frames": len(positions),
            "optimizer_updates": updates,
            "loss_gradients": [[0.2, 0.3]] * updates,
            "online_seconds": 0.1,
            "selection_seconds": 0.01,
            "ssl_seconds": 0.08,
        }

    with patch("cafl4ds.temporal_action_response.optimize_window", side_effect=observe):
        rows = run_action(None, None, torch.empty(640, 3, 1, 1), 2, action, horizon=8)
    assert [row["start_frame"] for row in rows] == list(range(128, 640, 64))
    assert verify_rows(rows, 2, action, 8) == (16 if action == "full" else 8)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["selected_global_indices"][0] += 1
    with pytest.raises(ValueError, match="positions"):
        verify_rows(corrupted, 2, action, 8)
    corrupted = copy.deepcopy(rows)
    corrupted[-1]["optimizer_updates"] += 1
    with pytest.raises(ValueError, match="work"):
        verify_rows(corrupted, 2, action, 8)


def test_horizon_cannot_extend_beyond_available_arrivals() -> None:
    """Insufficient future input is rejected instead of silently truncating work."""
    with pytest.raises(ValueError, match="invalid"):
        run_action(None, None, torch.empty(64, 3, 1, 1), 0, "full", 8)


def test_adjusted_night_bound_and_cost_remain_mandatory() -> None:
    """Nominal or balanced success cannot promote an unresolved domain."""
    primary = {metric: {"simultaneous95_pp": [-0.2, 0.2]} for metric in ("balanced", "A", "B")}
    assert preference(primary, 0.7) == "half_efficiency_preferred"
    primary["B"]["simultaneous95_pp"] = [-1.1, 0.2]
    assert preference(primary, 0.7) == "unresolved"
    primary["B"]["simultaneous95_pp"] = [-1.3, -1.1]
    assert preference(primary, 0.7) == "full_quality_required"
