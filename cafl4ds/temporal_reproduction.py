"""Exact JSON-canonical comparisons for live and persisted loss records."""

from __future__ import annotations

import json
from typing import Any


def same_losses(live: Any, persisted: Any) -> bool:  # noqa: ANN401 - nested loss records
    """Normalize tuple/list representation without rounding or numeric tolerance."""
    return bool(json.loads(json.dumps(live, allow_nan=False)) == json.loads(json.dumps(persisted, allow_nan=False)))
