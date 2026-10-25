"""Exercise actual file replacement with an injected transient Windows lock."""

import os
from pathlib import Path

import pytest

from scripts import temporal_checkpoint_retry as shim


@pytest.mark.parametrize("eligible", [True, False])
def test_retries_only_requested_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eligible: bool) -> None:
    """Retry a checkpoint lock; propagate identical errors on unrelated files."""
    source = tmp_path / "latest.partial"
    destination = tmp_path / ("latest.pt" if eligible else "unrelated.pt")
    source.write_bytes(b"complete checkpoint")
    actual_replace = os.replace
    calls = []

    def flaky_replace(src: str | Path, dst: str | Path) -> None:
        calls.append((src, dst))
        if len(calls) == 1:
            error = PermissionError("Simulated transient sharing lock")
            error.winerror = 5
            raise error
        actual_replace(src, dst)

    def execute(module: str, run_name: str) -> None:
        assert module == "scripts.run_temporal_state_development" and run_name == "__main__"
        os.replace(source, destination)

    monkeypatch.setattr(shim.os, "replace", flaky_replace)
    monkeypatch.setattr(shim.runpy, "run_module", execute)
    monkeypatch.setattr(shim.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(shim.sys, "argv", ["shim", "--output", str(tmp_path)])
    if eligible:
        shim.main()
        assert destination.read_bytes() == b"complete checkpoint" and len(calls) == 2
    else:
        with pytest.raises(PermissionError):
            shim.main()
        assert len(calls) == 1 and source.exists()
    assert os.replace is flaky_replace
