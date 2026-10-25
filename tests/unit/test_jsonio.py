"""Tests for spec-valid JSON serialization (``cafl4ds.jsonio``)."""

from __future__ import annotations

import json
import math

from cafl4ds.jsonio import dumps_valid, sanitize_nonfinite


def test_sanitize_replaces_non_finite_with_none() -> None:
    """Inf / -inf / nan become None; finite values and non-floats are untouched."""
    obj = {
        "inf": float("inf"),
        "ninf": float("-inf"),
        "nan": float("nan"),
        "finite": 1.5,
        "int": 3,
        "str": "x",
        "none": None,
        "nested": {"a": [float("inf"), 0.0, float("nan")], "b": (float("-inf"), 2)},
    }
    # Compare by equality (the function returns `object`) — non-finite -> None, tuples -> lists.
    assert sanitize_nonfinite(obj) == {
        "inf": None,
        "ninf": None,
        "nan": None,
        "finite": 1.5,
        "int": 3,
        "str": "x",
        "none": None,
        "nested": {"a": [None, 0.0, None], "b": [None, 2]},
    }


def test_dumps_valid_is_strict_parseable() -> None:
    """The output has no Infinity/NaN tokens and parses under a strict (constant-rejecting) parser."""
    payload = {"gate": {"forget_ratio": float("inf")}, "recon": [float("nan"), 0.4]}
    text = dumps_valid(payload)
    assert "Infinity" not in text and "NaN" not in text
    assert '"forget_ratio": null' in text
    # Strict parse: reject any JSON constant (Infinity/-Infinity/NaN) — must NOT be invoked.
    strict = json.loads(text, parse_constant=_reject)
    assert strict["gate"]["forget_ratio"] is None
    assert strict["recon"][0] is None and strict["recon"][1] == 0.4


def test_dumps_valid_indent_matches_pretty_format_json() -> None:
    """Default indent is 2 spaces, matching the repo's pretty-format-json hook (no reformat churn)."""
    assert dumps_valid({"a": 1}) == '{\n  "a": 1\n}'


def test_dumps_valid_round_trips_finite_data() -> None:
    """Finite structures are unchanged in value after a dump/load round-trip."""
    payload = {"x": 1, "y": [0.1, 0.2], "z": {"k": True, "s": "v"}}
    assert json.loads(dumps_valid(payload)) == payload


def _reject(token: str) -> float:
    """A ``parse_constant`` that raises — strict JSON has no Infinity/-Infinity/NaN."""
    raise AssertionError(f"non-finite JSON constant {token!r} present")


def test_finite_check_is_exhaustive() -> None:
    """A sanity check that math.isfinite drives the mapping (guards against a future signbit bug)."""
    assert sanitize_nonfinite(math.inf) is None
    assert sanitize_nonfinite(-math.inf) is None
    assert sanitize_nonfinite(math.nan) is None
    assert sanitize_nonfinite(0.0) == 0.0
    assert sanitize_nonfinite(-0.0) == -0.0
