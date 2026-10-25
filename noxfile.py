"""Nox sessions for testing across all supported Python versions.

`poe test` runs the fast single-interpreter loop with coverage + HTML/XML artifacts
(the canonical run that feeds docs and badges). This matrix instead just asserts the
suite passes on every supported interpreter, so we clear pytest's heavy ``addopts``
(coverage/html/junit) from pyproject with ``-o addopts=`` to keep it lean.

This file is the single source of truth for which Python versions are supported; CI
reads the same set rather than duplicating it.
"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)  # type: ignore[misc]
def tests(session: nox.Session) -> None:
    """Run the unit test suite on a single Python version."""
    # Install the project + dev group into nox's uv-managed venv. `--extra cpu` selects the
    # CPU torch build (the matrix runs on CPU CI runners; see pyproject's torch extras).
    session.run_install(
        "uv",
        "sync",
        "--group",
        "dev",
        "--extra",
        "cpu",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    # Matrix = pass/fail only: drop the coverage/artifact addopts defined in pyproject.
    # "tests" (not "tests/unit") so this file works unmodified both here (root's tests
    # live flat under tests/) and copied verbatim into generated projects (tests/unit/).
    session.run("pytest", "-n", "auto", "tests", "-o", "addopts=-ra")
