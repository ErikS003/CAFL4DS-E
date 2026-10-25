"""Supervise the replay study with persistent status and optional temporary Windows sleep prevention."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/adaptation-bdd/scope-replay/20260929"


def main() -> None:
    """Run one explicit attempt; never silently restart or overwrite an existing attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-awake", action="store_true", help="Use only with user approval; changes no power settings."
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT / "background_status.json"
    # Exclusive creation prevents duplicate supervisors from launching overlapping experiments.
    with status_path.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    status: dict[str, Any] = {
        "state": "starting",
        "supervisor_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "keep_awake_requested": args.keep_awake,
        "keep_awake_active": False,
    }

    def save() -> None:
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if args.keep_awake:
            if os.name != "nt":
                raise RuntimeError("This keep-awake helper supports Windows only.")
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Windows rejected the temporary keep-awake request.")
            status["keep_awake_active"] = True
            save()
        with (OUTPUT / "background_stdout.log").open("x", encoding="utf-8") as stdout:
            with (OUTPUT / "background_stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(  # noqa: S603 - fixed interpreter/module in this repository
                    [sys.executable, "-u", "-m", "scripts.run_scope_replay"],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                status.update(state="running", child_pid=child.pid)
                save()
                status["exit_code"] = child.wait()
        status["state"] = "completed" if status["exit_code"] == 0 else "failed"
    except Exception as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        if status["keep_awake_active"]:
            status["keep_awake_release_succeeded"] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
            status["keep_awake_active"] = False
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
