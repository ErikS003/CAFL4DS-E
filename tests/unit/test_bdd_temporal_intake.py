"""Protect bounded acquisition against full-part downloads and ZIP seek mistakes."""

from __future__ import annotations

import io
import zipfile
from types import TracebackType
from typing import Any

import pytest

from scripts.prepare_bdd_temporal_archive import SplitHTTPFile


class Response:
    """Small HTTP fixture that records whether an unsafe body was read."""

    def __init__(self, data: bytes, start: int, end: int, status: int = 206) -> None:
        """Expose an exact bounded response."""
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        self.raw = io.BytesIO(data[start:end + 1])

    def __enter__(self) -> Response:
        """Return the open fixture."""
        return self

    def __exit__(self, _kind: type[BaseException] | None, _value: BaseException | None,
                 _traceback: TracebackType | None) -> None:
        """Close without side effects."""


def remote_for(monkeypatch: pytest.MonkeyPatch, segments: list[bytes]) -> SplitHTTPFile:
    """Build an in-memory split server honoring byte ranges."""
    parts = [{"path": str(i), "size": len(data)} for i, data in enumerate(segments)]
    remote = SplitHTTPFile(parts, "pinned", 100000)

    def get(url: str, headers: dict[str, str], **_: Any) -> Response:  # noqa: ANN401 - HTTP fixture keywords
        index = int(url.rsplit("/", 1)[1])
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        return Response(segments[index], start, end)

    monkeypatch.setattr(remote.session, "get", get)
    return remote


def test_zip_directory_and_member_cross_split_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZIP directory and compressed payload must survive arbitrary split boundaries."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bdd100k/videos/100k/train/example.mov", bytes(range(256)) * 20)
    data = buffer.getvalue()
    remote = remote_for(monkeypatch, [data[:100], data[100:300], data[300:]])
    with zipfile.ZipFile(remote) as archive:
        assert archive.read("bdd100k/videos/100k/train/example.mov") == bytes(range(256)) * 20


def test_unhonored_range_is_rejected_without_reading_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server's full-part response must be rejected before consuming its body."""
    remote = remote_for(monkeypatch, [b"abcdefghij"])
    response = Response(b"abcdefghij", 0, 1, status=200)
    monkeypatch.setattr(remote.session, "get", lambda *_args, **_kwargs: response)
    with pytest.raises(ValueError, match="did not honor"):
        remote.read(2)
    assert response.raw.tell() == 0


def test_budget_rejection_precedes_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No request is made when a ZIP read would exceed its declared budget."""
    remote = remote_for(monkeypatch, [b"abcdefghij"])
    remote.budget = 1
    monkeypatch.setattr(remote.session, "get", lambda *_args, **_kwargs: pytest.fail("unexpected request"))
    with pytest.raises(ValueError, match="budget"):
        remote.read(2)
