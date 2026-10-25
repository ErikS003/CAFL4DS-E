"""Verify collection/report ordering and failure handling for the new authorized study."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_memory_admission_background as supervisor


@pytest.mark.parametrize("exit_code", [0, 1])
def test_supervisor_audits_only_successful_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    """Worker failure must not produce a completion report; exclusive launch prevents duplicates."""
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["supervisor", "--output", str(out)])
    calls: list[list[str]] = []

    def launch(command: list[str], **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 - subprocess mock
        calls.append(command)
        return SimpleNamespace(pid=1, wait=lambda: exit_code)

    def execute(command: list[str], **kwargs: Any) -> None:  # noqa: ANN401 - subprocess mock
        calls.append(command)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(subprocess, "run", execute)
    if exit_code:
        with pytest.raises(RuntimeError, match="Collection failed"):
            supervisor.main()
    else:
        supervisor.main()
    state = json.loads((tmp_path / "run.supervisor.json").read_text())
    assert state["stage"] == ("failed" if exit_code else "complete")
    assert len(calls) == (1 if exit_code else 2)
    if not exit_code:
        assert "scripts.report_memory_admission" in calls[-1]
    with pytest.raises(FileExistsError):
        supervisor.main()
