"""Operational retry shim for transient Windows locks; execute unchanged scientific code."""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    """Retry only atomic latest-checkpoint replacement within the requested output root."""
    arguments = sys.argv[1:]
    position = arguments.index("--output")
    root = Path(arguments[position + 1]).resolve()
    original = os.replace

    def replace(source: str | Path, destination: str | Path) -> None:
        source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
        eligible = (
            destination_path.is_relative_to(root)
            and destination_path.name == "latest.pt"
            and source_path.name == "latest.partial"
            and source_path.parent == destination_path.parent
        )
        for attempt in range(41):
            try:
                original(source, destination)
                return
            except PermissionError as error:
                if not eligible or getattr(error, "winerror", None) != 5 or attempt == 40:
                    raise
                print(f"Checkpoint access retry {attempt + 1}: {destination_path}", file=sys.stderr, flush=True)
                time.sleep(0.25)

    os.replace = replace
    try:
        sys.argv = ["scripts.run_temporal_state_development", *arguments]
        runpy.run_module("scripts.run_temporal_state_development", run_name="__main__")
    finally:
        os.replace = original


if __name__ == "__main__":
    main()
