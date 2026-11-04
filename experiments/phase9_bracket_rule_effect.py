"""What does the dropped-chiral-bracket rule do to the delivered CSV?

The rule restores the ``]`` of a chiral atom whose closing bracket
the recogniser dropped -- ``[C@@H4`` -> ``[C@@H]4`` -- but only when
the digit is a ring-closure digit with a partner in the string.

This reports what the rule changes on the shipped CSV: how many rows
it touches, how many of those become parseable, and what the text
looks like before and after.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_bracket_rule_effect.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"


def _parses(text: str) -> bool:
    try:
        return Chem.MolFromSmiles(text) is not None
    except Exception:
        return False


def main() -> int:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    changed = []
    became_parseable = []
    broke = []
    for row in rows:
        before = (row.get("smiles") or "").strip()
        if not before:
            continue
        after = auto_repair_smiles(before)
        if after == before:
            continue
        changed.append((row["cmpd_id"], before, after))
        if _parses(before) and not _parses(after):
            broke.append(row["cmpd_id"])
        if not _parses(before) and _parses(after):
            became_parseable.append(row["cmpd_id"])

    print(f"{len(rows)} rows")
    print(f"rows the repair chain now changes: {len(changed)}")
    print(f"of those, newly parseable        : {len(became_parseable)} "
          f"{became_parseable}")
    print(f"of those, no longer parseable    : {len(broke)} {broke}")
    print()
    for compound_id, before, after in changed[:6]:
        index = 0
        for index, (a, b) in enumerate(zip(before, after)):
            if a != b:
                break
        start = max(0, index - 25)
        print(f"{compound_id}:")
        print(f"  before: {before[start:index + 30]!r}")
        print(f"  after : {after[start:index + 31]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
