"""How often is the letter ``O`` read as the digit ``0``?

``C(=O)`` read as ``C(=0)`` is not a SMILES -- a ring-closure digit
cannot stand where an atom belongs -- so such a cell cannot parse.
This counts the pattern in the shipped CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]

# A digit directly after a bond symbol, or directly after an opening
# bracket: in those places a ring-closure number is not what a
# printed SMILES means, because a ring closure has to be opened
# somewhere else in the same string.
SUSPECT = re.compile(r"[=#\-]0|(\[0)|(0\])")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "data" / "processed" / "compounds_combined.csv",
        help="The CSV to scan (default: %(default)s).",
    )
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    suspects = []
    for row in rows:
        text = (row.get("smiles") or "").strip()
        if text and SUSPECT.search(text):
            suspects.append((row["cmpd_id"], text))

    print(f"{len(rows)} rows")
    print(f"rows whose smiles holds a digit where an atom belongs: "
          f"{len(suspects)}")
    parses = 0
    for compound_id, text in suspects:
        try:
            ok = Chem.MolFromSmiles(text) is not None
        except Exception:
            ok = False
        parses += ok
        print(f"  {compound_id:>6} parses={ok!s:<5} {text[:100]}")
    print(f"\nof those, {parses} parse as they stand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
