"""Crash recovery accepts complete committed prefixes, never duplicate or missing evidence."""

import pytest

from scripts.resume_health_utility_bridge import ACTIONS, preserve_json, validate_progress


def test_recovery_accepts_partial_parent_and_committed_prefix() -> None:
    """A crash between action readouts can resume without repeating earlier readouts."""
    choices = [{"parent": p} for p in range(2)]
    rows = [{"parent": 0, "draw": 0, "action": a} for a in ACTIONS]
    rows.append({"parent": 1, "draw": 0, "action": "uniform"})
    validate_progress(choices, rows)


def test_recovery_rejects_duplicates_and_out_of_order_results() -> None:
    """A corrupt checkpoint cannot silently masquerade as a finished branch."""
    row = {"parent": 0, "draw": 0, "action": "uniform"}
    with pytest.raises(ValueError, match="unique ordered"):
        validate_progress([{"parent": 0}], [row, row])
    with pytest.raises(ValueError, match="unique ordered"):
        validate_progress([{"parent": 0}], [{**row, "action": "rank_high"}])


def test_recovery_requires_previously_frozen_menu() -> None:
    """Completed labels without their pre-outcome image choices cannot be reused."""
    with pytest.raises(ValueError, match="lacks its frozen"):
        validate_progress([], [{"parent": 0, "draw": 0, "action": "uniform"}])


def test_recovery_cannot_replace_frozen_decisions(tmp_path) -> None:
    """An existing decision remains byte-identical and cannot be retuned after a crash."""
    path = tmp_path / "decisions.json"
    preserve_json(path, {"choice": "uniform"})
    initial = path.read_bytes()
    preserve_json(path, {"choice": "uniform"})
    assert path.read_bytes() == initial
    with pytest.raises(ValueError, match="artifact differs"):
        preserve_json(path, {"choice": "rank_high"})
    assert path.read_bytes() == initial
