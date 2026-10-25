"""Run the recurrence study with persistent status and optional approved idle-sleep prevention."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cafl4ds.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/adaptation-bdd/recurrence-retention/20261001"


def main() -> None:
    """Start one exclusive supervised attempt and release power requests on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-awake", action="store_true", help="Requires explicit user approval")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prior-run", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "background_status.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    status: dict[str, Any] = {
        "state": "starting",
        "supervisor_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "keep_awake_requested": args.keep_awake,
        "keep_awake_active": False,
    }

    def save() -> None:
        atomic_json(path, status)

    save()
    try:
        if args.keep_awake:
            if os.name != "nt" or not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Temporary idle-sleep prevention unavailable")
            status["keep_awake_active"] = True
            save()
        with (output / "stdout.log").open("x", encoding="utf-8") as stdout:
            with (output / "stderr.log").open("x", encoding="utf-8") as stderr:
                command = [sys.executable, "-u", "-m", "scripts.run_recurrence_retention", "--output", str(output)]
                if args.prior_run is not None:
                    command.extend(["--prior-run", str(args.prior_run.resolve())])
                    status["state"] = "preflight"
                    save()
                    subprocess.run([*command, "--preflight-only"], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)  # noqa: S603
                child = subprocess.Popen(  # noqa: S603 - fixed local interpreter and module
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                status.update(state="running", child_pid=child.pid)
                save()
                status["exit_code"] = child.wait()
                if status["exit_code"] == 0 and args.prior_run is not None:
                    status["state"] = "auditing"
                    save()
                    subprocess.run(  # noqa: S603 - fixed local audit module
                        [sys.executable, "-m", "scripts.audit_recurrence_retention", str(output)],
                        cwd=ROOT, stdout=stdout, stderr=stderr, check=True,
                    )
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
