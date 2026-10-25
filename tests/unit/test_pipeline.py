"""Tests for cafl4ds/pipeline.py."""

import pytest

from cafl4ds.pipeline import Pipeline, Repeat, Transform, Upper


@pytest.mark.parametrize(
    ("transform", "text", "expected"),
    [
        (Upper(), "hello", "HELLO"),
        (Repeat(times=3), "hi", "hi hi hi"),
    ],
)
def test_transform(transform: Transform, text: str, expected: str) -> None:
    """Each transform maps its input string as documented."""
    assert transform(text) == expected


def test_pipeline_composes_steps_in_order() -> None:
    """The pipeline applies its steps left to right over the starting message."""
    result = Pipeline(steps=[Upper(), Repeat(times=2)], message="hi").run()
    assert result == "HI HI"
