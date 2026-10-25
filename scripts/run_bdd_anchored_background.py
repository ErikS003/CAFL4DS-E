"""Supervise content filtering and its audit with a scoped Windows idle-sleep guard."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

from cafl4ds.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Launch a new collection and release the sleep guard on success or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    status = out.parent / f"{out.name}.supervisor.json"
    if out.exists() or status.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {"stage": "starting", "supervisor_pid": os.getpid(), "idle_sleep_guard": False}
    try:
        if os.name == "nt":
            if not ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001)):
                raise RuntimeError("Windows rejected the idle-sleep guard")
            state["idle_sleep_guard"] = True
        with (out.parent / f"{out.name}.stdout.log").open("x") as stdout:
            with (out.parent / f"{out.name}.stderr.log").open("x") as stderr:
                child = subprocess.Popen(  # noqa: S603 - explicit local interpreter and module
                    [sys.executable, "-u", "-m", "scripts.run_bdd_anchored_filter", "--output", str(out)],
                    cwd=ROOT, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                state.update(stage="collecting", child_pid=child.pid)
                atomic_json(status, state)
                if child.wait() != 0:
                    raise RuntimeError("collection failed; see stderr log")
                state["stage"] = "auditing"
                atomic_json(status, state)
                subprocess.run(  # noqa: S603 - explicit local report module
                    [sys.executable, "-u", "-m", "scripts.report_bdd_anchored_filter", str(out)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, check=True,
                )
        state["stage"] = "complete"
    except Exception as error:
        state.update(stage="failed", error=repr(error))
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000000))
        state["idle_sleep_guard"] = False
        atomic_json(status, state)


if __name__ == "__main__":
    main()
