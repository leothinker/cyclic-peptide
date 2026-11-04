"""Check the shipped CSV against what it is supposed to contain.

The CSV is the deliverable, so its shape and its content are checked
separately: a row count says nothing about whether the values in the
rows are right.

* **shape** -- the number of rows, the number of columns, the
  uniqueness of the compound number, and how much of each activity
  column is filled;
* **content** -- the distribution of the ionisation mode, the number
  of empty values, and whether compound 1036 is still character-exact
  against the hand transcription the researcher supplied.

Usage::

    .venv\\Scripts\\python.exe experiments/verify_csv.py
    .venv\\Scripts\\python.exe experiments/verify_csv.py --csv other.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The hand transcription of compound 1036, character for character.
REFERENCE_1036 = (
    "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C2CCCCC2)C(=O)N3[C@@H](CC3)"
    "C(=O)N(C)[C@@H]4C(=O)N(C)CC(=O)N[C@@H](CCC5=CC(F)=C(C(F)=C5)"
    "C(F)(F)F)C(=O)N6[C@@H](CCC6)C(=O)NC7(CCCC7)C(=O)N(C)[C@@H]"
    "(C8CCCC8)C(=O)N(C)[C@@H](CC(=O)N(C)[C@H](CCNC(=S)C9=C(F)C(F)="
    "C(SC4)C(F)=C9F)C(=O)N1)C(=O)N(C)C"
)

ACTIVITY_FIELDS = (
    "kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO / "data" / "processed" / "compounds_combined.csv",
        help="The CSV to check (default: %(default)s).",
    )
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0].keys())

    print(f"file               : {args.csv}")
    print(f"rows               : {len(rows)}")
    print(f"columns            : {len(columns)}")
    print(f"unique cmpd_id     : {len({row['cmpd_id'] for row in rows})}")

    print("\n-- activity field fill --")
    for field in ACTIVITY_FIELDS:
        filled = sum(1 for row in rows if (row.get(field) or "").strip())
        print(f"  {field:<14} {filled}/{len(rows)}")

    print("\n-- the ionisation mode --")
    for value, count in Counter(
        (row.get("ms_polarity") or "").strip() for row in rows
    ).most_common():
        print(f"  {value!r:<16} {count}")

    print("\n-- compound 1036 --")
    row = next(row for row in rows if row["cmpd_id"] == "1036")
    exact = row["smiles"] == REFERENCE_1036
    print(f"  character-exact against the hand transcription: {exact}")
    print(f"  length {len(row['smiles'])} against {len(REFERENCE_1036)}")
    if not exact:
        matcher = difflib.SequenceMatcher()
        matcher.set_seqs(REFERENCE_1036, row["smiles"])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            print(
                f"    {tag:<8} {REFERENCE_1036[i1:i2]!r} -> "
                f"{row['smiles'][j1:j2]!r}"
            )

    print("\n-- the structure columns --")
    print(
        "  smiles non-empty     :",
        sum(1 for row in rows if (row.get("smiles") or "").strip()),
    )
    print(
        "  canonical non-empty  :",
        sum(1 for row in rows if (row.get("canonical_smiles") or "").strip()),
    )
    print(
        "  smiles_best non-empty:",
        sum(1 for row in rows if (row.get("smiles_best") or "").strip()),
    )
    print(
        "  smiles_best_source   :",
        dict(Counter(
            (row.get("smiles_best_source") or "").strip() for row in rows
        ).most_common()),
    )
    print(
        "  structure_confidence :",
        dict(Counter(
            (row.get("structure_confidence") or "").strip() for row in rows
        ).most_common()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
