"""Spec-valid JSON serialization for experiment artifacts.

Python's ``json`` emits the bare tokens ``Infinity`` / ``-Infinity`` / ``NaN`` for non-finite floats
(``allow_nan=True`` by default). Those are **not** valid JSON — strict parsers (browsers' ``JSON.parse``,
many linters, other-language tooling) reject them — even though Python's own ``json.load`` and ``jq``
read them back leniently. Our positive-control harnesses legitimately produce non-finite values
(divide-by-zero ratios like ``forget_ratio``, or ``NaN`` for readouts that don't apply to a vehicle),
so a raw ``json.dumps`` writes spec-invalid artifacts.

This module maps non-finite floats to ``null`` (the same choice JavaScript's ``JSON.stringify`` makes)
and serializes with ``allow_nan=False`` so any non-finite value that slips the sanitizer *raises* rather
than silently writing an invalid token. ``null`` is faithful here: every non-finite our harnesses emit is
a "not defined / degenerate" marker (an undefined ratio, an inapplicable readout), not a magnitude worth
preserving. Python readers get ``None`` instead of ``float('inf')`` / ``nan``.
"""

from __future__ import annotations

import json
import math


def sanitize_nonfinite(obj: object) -> object:
    """Recursively replace non-finite floats (``inf`` / ``-inf`` / ``nan``) with ``None``.

    Walks dicts and lists/tuples; every other value is returned unchanged. The result contains no
    non-finite floats, so it serializes to spec-valid JSON.

    Args:
        obj: Any JSON-serializable structure (dicts, lists, scalars).

    Returns:
        The same structure with non-finite floats replaced by ``None`` (lists returned for tuples).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize_nonfinite(v) for v in obj]
    return obj


def dumps_valid(obj: object, *, indent: int = 2) -> str:
    """Serialize ``obj`` to strict, spec-valid JSON — non-finite floats become ``null``.

    ``allow_nan=False`` is a belt-and-suspenders guard: the sanitizer removes every non-finite value,
    so if one survives (a non-float carrier, say) the dump raises ``ValueError`` instead of writing an
    invalid ``Infinity`` / ``NaN`` token.

    Args:
        obj: The structure to serialize.
        indent: Indentation width (matches the ``pretty-format-json`` pre-commit hook's 2 spaces).

    Returns:
        A JSON string with no non-finite tokens.
    """
    return json.dumps(sanitize_nonfinite(obj), indent=indent, allow_nan=False)
