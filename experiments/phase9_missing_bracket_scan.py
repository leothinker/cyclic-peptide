"""How many printed SMILES lose the closing bracket of a chiral atom?

The repair chain fixes ``[C@H(`` -> ``[C@H](`` -- a chiral atom whose
closing bracket was dropped in front of a parenthesis.  The same
recogniser failure in front of a *ring-closure digit* is not handled:
``[C@@H4`` should be ``[C@@H]4``.  A bracket atom in SMILES is
always closed before a ring-closure digit, so the form with the digit
inside the brackets is not valid SMILES and cannot be intended.

This scans the delivered CSV for the pattern and reports how many
rows it affects, so the size of the gap is a number rather than an
impression.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_missing_bracket_scan.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.chem import auto_repair_smiles  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# A bracket atom whose closing bracket is missing, followed by a
# ring-closure digit: "[C@@H4".  A digit cannot appear inside a
# bracket atom in SMILES, so this is always a dropped "]".
MISSING_BRACKET = re.compile(r"\[C@{1,2}H\d")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "data" / "processed" / "compounds_combined.csv",
    )
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    hits = []
    for row in rows:
        text = (row.get("smiles") or "").strip()
        if not text:
            continue
        match = MISSING_BRACKET.search(text)
        if match:
            hits.append((row["cmpd_id"], text, match.group(0)))

    print(f"{len(rows)} rows")
    print(f"rows with a dropped chiral bracket: {len(hits)}")
    for compound_id, text, matched in hits:
        print(f"  {compound_id}: {matched}")

    # What would the repair give, and would it parse?
    fixed_rows = []
    for compound_id, text, _matched in hits:
        repaired = re.sub(
            r"(\[C@{1,2}H)(\d)", r"\1]\2", text
        )
        parses = Chem.MolFromSmiles(repaired) is not None
        fixed_rows.append((compound_id, parses, repaired))
    if fixed_rows:
        print("\nwith the bracket restored:")
        for compound_id, parses, repaired in fixed_rows:
            print(f"  {compound_id}: parses={parses}")

    # The repair chain must not change anything that does not need it.
    changed_others = 0
    for row in rows:
        text = (row.get("smiles") or "").strip()
        if not text:
            continue
        if MISSING_BRACKET.search(text):
            continue
        if re.sub(r"(\[C@{1,2}H)(\d)", r"\1]\2", text) != text:
            changed_others += 1
    print(f"\nrows without the pattern that the rule would touch: {changed_others}")
    print(f"rows the repair chain would change at all: "
          f"{sum(1 for row in rows if auto_repair_smiles((row.get('smiles') or '').strip()) != (row.get('smiles') or '').strip())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
