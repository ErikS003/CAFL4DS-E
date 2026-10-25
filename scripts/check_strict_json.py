#!/usr/bin/env python3
"""Pre-commit hook: reject spec-invalid JSON (bare ``Infinity`` / ``-Infinity`` / ``NaN`` tokens).

The stock ``check-json`` hook uses Python's lenient ``json.load``, which *accepts* non-finite
constants — so a spec-invalid artifact (the kind our positive-control harnesses used to emit) passes
it silently. This complements ``check-json`` with a strict parse that rejects those constants, so
artifacts stay portable to strict parsers (``JSON.parse``, other-language tooling). New artifacts are
already clean because the harnesses serialize via :func:`cafl4ds.jsonio.dumps_valid` (non-finite →
``null``); this hook guards against regressions and hand-edited files.

Usage: ``check_strict_json.py FILE [FILE ...]`` — exits non-zero listing any file with a non-finite
constant (or a genuine parse error).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class _NonFinite(ValueError):
    """Raised by the strict ``parse_constant`` hook when a non-finite JSON constant is seen."""


def _reject(token: str) -> float:
    raise _NonFinite(token)


def _check(path: Path) -> str | None:
    """Return an error message if ``path`` is not strict-valid JSON, else ``None``."""
    try:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    except _NonFinite as exc:
        return f"{path}: contains non-finite JSON constant {str(exc)!r} (not valid JSON — use null)"
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return f"{path}: {exc}"
    return None


def main(argv: list[str]) -> int:
    """Check each path argument; print all failures and return 1 if any, else 0."""
    failures = [msg for arg in argv if (msg := _check(Path(arg))) is not None]
    for msg in failures:
        print(msg, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
