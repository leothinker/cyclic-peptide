"""How close are the three hand-verified references, now?

Compound 1036 and 1038 have hand transcriptions that are character
exact in the delivered CSV.  Compound 1037 used to have one error --
a dropped chiral bracket -- and the repair chain now restores it.

This prints the three references side by side with the delivered
text and the edit distance, so the claim is checkable rather than
asserted.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_reference_check.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from phase7_route_verify import REFERENCES, edit_distance  # noqa: E402

CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"


def main() -> int:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["cmpd_id"]: row for row in csv.DictReader(handle)}

    total_edits = 0
    exact = 0
    for compound_id, reference in REFERENCES.items():
        text = (rows.get(compound_id, {}).get("smiles") or "").strip()
        distance = edit_distance(reference, text)
        total_edits += distance
        exact += distance == 0
        print(f"{compound_id}: edit distance {distance} "
              f"({len(text)} of {len(reference)} characters)")
        if distance:
            for index, (a, b) in enumerate(zip(reference, text)):
                if a != b:
                    print(f"    first difference at {index}: "
                          f"reference {reference[max(0,index-12):index+12]!r}")
                    print(f"                          "
                          f"delivered {text[max(0,index-12):index+13]!r}")
                    break
    print(f"\n{exact} of {len(REFERENCES)} are character exact, "
          f"{total_edits} edits in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
