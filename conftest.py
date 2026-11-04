"""Make the test suite import *this* checkout's package.

Without this, the import of ``cpd`` depends on what happens to be
installed in the environment.  That is not a theoretical problem: if
the package was installed in editable mode from a *different*
checkout, ``import cpd`` silently resolves to that other tree, and
the tests then measure somebody else's code.  The symptom is an
import-order dependency -- one test file sets up ``sys.path`` and
another does not, so the result depends on the order pytest collects
them in.

pytest imports the root ``conftest.py`` before any test module, so
putting the local ``src`` at the front of ``sys.path`` here makes the
resolution deterministic and local.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"

if SRC.is_dir():
    # Remove any earlier entry for the same directory so that the
    # local source tree comes first and appears only once.
    while str(SRC) in sys.path:
        sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))
