"""Run the test suite on a machine that denies ``os.scandir``.

This machine refuses to list a directory that was created with
``mode=0o700`` -- the mode pytest uses for the directories of its
``tmp_path`` fixture.  The fixture then fails at *setup*, before any
pipeline code runs, and the run reports an error that has nothing to
do with the code under test.

The workaround is to create those directories with the inherited
permissions, which the machine does allow, and to let a failed
clean-up of a stale directory pass instead of failing the run.  The
tests themselves are untouched.

Usage::

    .venv\\Scripts\\python.exe tools/run_tests.py
    .venv\\Scripts\\python.exe tools/run_tests.py tests/test_vote.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _relax_permissions() -> None:
    """Let the directories of the ``tmp_path`` fixture be listed."""
    original_mkdir = os.mkdir

    def _relaxed_mkdir(path, mode=0o777, *args, **kwargs):
        return original_mkdir(path, 0o777, *args, **kwargs)

    os.mkdir = _relaxed_mkdir

    import _pytest.pathlib as pathlib_mod

    original_rm_rf = pathlib_mod.rm_rf

    def _lenient_rm_rf(path, *args, **kwargs):
        try:
            return original_rm_rf(path, *args, **kwargs)
        except OSError:
            return None

    pathlib_mod.rm_rf = _lenient_rm_rf

    import _pytest.tmpdir as tmpdir

    tmpdir.cleanup_dead_symlinks = lambda *args, **kwargs: None


def main(argv: list[str] | None = None) -> int:
    _relax_permissions()

    import pytest

    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments:
        arguments = ["tests", "-q", "--tb=short", "-p", "no:cacheprovider"]

    # pytest's default base temp directory is ``pytest-of-<user>`` in
    # the system temp.  A copy left behind by an earlier run -- made
    # before the permissions were relaxed -- cannot be listed and
    # cannot be removed, so the run would fail on it.  A fresh
    # directory is used instead.
    if not any(argument.startswith("--basetemp") for argument in arguments):
        import time

        base = (
            Path(os.environ.get("TEMP", r"C:\Windows\Temp"))
            / f"cpd_pytest_{os.getpid()}_{int(time.time())}"
        )
        arguments.append(f"--basetemp={base}")

    return pytest.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
