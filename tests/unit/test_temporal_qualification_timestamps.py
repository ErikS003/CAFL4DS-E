"""Real-frame intake must not confuse repeated timestamps with temporal coverage."""

import pytest

from scripts.decode_temporal_qualification import timestamps


def trace(seconds: list[float]) -> str:
    """Construct a minimal native-frame FFmpeg trace."""
    return "\n".join(f"n: {i} pts: {i} pts_time:{t} fmt:rgb24 s:1280x720" for i, t in enumerate(seconds))


def test_nonadvancing_frames_do_not_fill_slots() -> None:
    """Repeated timestamps cannot stand in for advancing frames."""
    result = timestamps(trace([0, 0, 0.05, 0.125, 0.125, 0.25]), count=3)
    assert [row["source_frame_index"] for row in result] == [0, 3, 5]


def test_missing_temporal_horizon_is_rejected() -> None:
    """A short clip must fail rather than silently shorten the horizon."""
    with pytest.raises(ValueError, match="full 32-second"):
        timestamps(trace([0, 0.125]), count=3)


def test_large_timestamp_gap_is_rejected() -> None:
    """Large gaps cannot masquerade as the desired arrival rate."""
    with pytest.raises(ValueError, match="gap"):
        timestamps(trace([0, 0.4]), count=2)
