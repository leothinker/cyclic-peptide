"""Run the CSV round-trip check without pytest's tmp_path fixture.

``tests/test_combined_pipeline.py::test_combined_csv_round_trip`` needs
pytest's ``tmp_path``, which cannot be created inside this sandbox (the
temp directory is not writable).  This script performs the same three
assertions against a directory inside the workspace so the behaviour is
still verified rather than merely skipped.

Run from the repo root::

    .venv/Scripts/python.exe analysis/run_roundtrip_check.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Imported under an alias: this script also defines a local ``main()``, which
# would otherwise shadow the module in this namespace.
import main as pipeline  # noqa: E402

WORK = ROOT / ".roundtrip_tmp"


def run() -> int:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        out = WORK / "out.csv"
        n = pipeline._write_csv(rows, out, ("a", "b"))
        assert n == 2, f"expected 2 rows written, got {n}"
        with out.open(encoding="utf-8-sig", newline="") as fh:
            loaded = list(csv.DictReader(fh))
        assert loaded == rows, f"round trip changed the data: {loaded}"
        print("PASS: _write_csv round-trips 2 rows unchanged")
        return 0
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run())
