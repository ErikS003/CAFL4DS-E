"""Supervise the authorized update-filtering collection and automatic audit/report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cafl4ds.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Launch one new output directory and stop automatically on failed compatibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/update-filtering/20261010-half-rate")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = out.parent / f"{out.name}.supervisor.json"
    with status.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    state = {"stage": "starting", "supervisor_pid": os.getpid()}
    try:
        with (out.parent / f"{out.name}.stdout.log").open("x", encoding="utf-8") as stdout:
            with (out.parent / f"{out.name}.stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(  # noqa: S603 - fixed local interpreter/module
                    [sys.executable, "-u", "-m", "scripts.run_update_filtering", "--output", str(out)],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                state.update(stage="collecting", child_pid=child.pid)
                atomic_json(status, state)
                exit_code = child.wait()
                if exit_code != 0:
                    raise RuntimeError(f"Collection failed with exit{exit_code}; see stderr log")
                state["stage"] = "auditing"
                atomic_json(status, state)
                subprocess.run(  # noqa: S603 - fixed local audit module
                    [sys.executable, "-m", "scripts.report_update_filtering", str(out)],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                )
        state["stage"] = "complete"
        atomic_json(status, state)
    except Exception as error:
        state.update(stage="failed", error=repr(error))
        atomic_json(status, state)
        raise


if __name__ == "__main__":
    main()
