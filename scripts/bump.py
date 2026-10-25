# ruff: noqa: S603, S607
"""Bump the version from conventional commits and emit release state for CI.

Runs ``cz bump``; on success it writes ``bumped``/``version`` to the output file and pushes
the commit + tag, otherwise it records ``bumped=false`` so downstream release jobs no-op.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

# Commitizen exit codes meaning "no release-worthy commits" (NoneIncrementExit / NoCommitsFoundError).
# Treated as a clean no-op rather than a failure so downstream release jobs simply skip.
CZ_NO_RELEASE_CODES = (3, 21)


def release_decision(rc: int) -> Literal["bump", "skip", "fail"]:
    """Map a ``cz bump`` exit code to a release action.

    Pure function (no side effects) so the exit-code contract can be unit-tested: ``0`` means a
    version was bumped, commitizen's no-eligible-commits codes mean skip, anything else is a real
    failure that must abort the pipeline.
    """
    if rc == 0:
        return "bump"
    if rc in CZ_NO_RELEASE_CODES:
        return "skip"
    return "fail"


def resolve_push_refspec(env: Mapping[str, str]) -> str:
    """Build the git refspec for pushing the bump commit back to the default branch.

    CI runners commonly check out a detached HEAD (GitLab always; GitHub for some events), so a
    bare ``HEAD`` cannot be resolved to a remote branch and git fails with "not a full refname".
    Resolve the branch name from the CI environment -- ``BRANCH`` (set by the GitHub workflow) or
    ``CI_COMMIT_BRANCH`` / ``CI_DEFAULT_BRANCH`` (provided by GitLab) -- and return an explicit
    ``HEAD:refs/heads/<branch>`` refspec, falling back to bare ``HEAD`` when no branch is known.

    Pure function (env in, refspec out) so the resolution can be unit-tested.
    """
    branch = env.get("BRANCH") or env.get("CI_COMMIT_BRANCH") or env.get("CI_DEFAULT_BRANCH")
    return f"HEAD:refs/heads/{branch}" if branch else "HEAD"


def _explain_push_failure() -> None:
    """Print an actionable hint for a rejected release push (usually branch protection)."""
    print(
        "\n"
        "Failed to push the version bump to the default branch. The bump commit and tag\n"
        "were created locally but could not be pushed, so the release did not complete.\n"
        "This is almost always a branch-protection authorization issue, not a bug: the\n"
        "release push needs permission to update the protected default branch.\n"
        "\n"
        "Grant it one of two ways (see your project's CI/release setup docs):\n"
        "  - add a branch-protection/ruleset bypass for the CI actor "
        "(e.g. github-actions[bot]), or\n"
        "  - create a token with 'contents: write' that is allowed to bypass protection\n"
        "    and expose it to this job (GitHub: the RELEASE_TOKEN secret; GitLab: CI_REPO_ACCESS).\n",
        file=sys.stderr,
    )


def main() -> None:
    """Run ``cz bump``, push the tag on success, and write CI release state to the output file."""
    parser = argparse.ArgumentParser(description="Bump version, push, and output state for CI.")
    parser.add_argument("out", nargs="?", default="bump.env", help="File to write state variables to")
    args = parser.parse_args()

    out_file = Path(args.out)

    print("Running commitizen bump...")
    # We do not use capture_output=True so that the user/CI still sees cz's standard logs
    result = subprocess.run(["cz", "bump", "--yes"], check=False)
    rc = result.returncode
    decision = release_decision(rc)

    if decision == "bump":
        # Get the newly created tag
        tag_proc = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"], text=True, capture_output=True, check=True
        )
        new_tag = tag_proc.stdout.strip()
        version = new_tag.lstrip("v")

        print(f"Writing state to {out_file}...")
        with open(out_file, "a") as f:
            f.write("bumped=true\n")
            f.write(f"version={version}\n")

        print(f"Pushing bump commit and tag {new_tag} to origin...")
        src_refspec = resolve_push_refspec(os.environ)
        try:
            subprocess.run(["git", "push", "origin", src_refspec], check=True)
            subprocess.run(["git", "push", "origin", new_tag], check=True)
        except subprocess.CalledProcessError as exc:
            _explain_push_failure()
            sys.exit(exc.returncode)

    elif decision == "skip":
        print(f"No release-worthy commits (cz exit {rc}); skipping release.")
        with open(out_file, "a") as f:
            f.write("bumped=false\n")
    else:
        print(f"cz bump failed unexpectedly (exit {rc}).", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
