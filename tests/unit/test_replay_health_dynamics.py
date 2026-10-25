"""Temporal pairing prevents future or cross-trajectory observations."""

import pytest

from cafl4ds.domain_replay_health import CONTRAST, DOMAIN
from cafl4ds.replay_health_dynamics import changes


def test_changes_require_same_past_trajectory() -> None:
    """Changes use only past readings, with known direction and elapsed work."""
    early = {"state": "stream1_77041_4", **dict.fromkeys((*CONTRAST, *DOMAIN), 2.0)}
    late = {"state": "stream1_77041_12", **dict.fromkeys((*CONTRAST, *DOMAIN), 6.0)}
    assert set(changes(early, late).values()) == {0.5}
    with pytest.raises(ValueError):
        changes(late, early)
    with pytest.raises(ValueError):
        changes(early, {**late, "state": "stream2_77041_12"})
