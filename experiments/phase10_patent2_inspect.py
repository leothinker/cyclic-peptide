"""Print the delivery CSV of WO2022251372 in a readable form.

The CSV is the deliverable, but a CSV is not easy to read in a
terminal.  This script prints the same content as a small table:
the compound, whether its printed SMILES parses, and the six letter
grades.

Usage::

    .venv\\Scripts\\python.exe experiments/phase10_patent2_inspect.py
    .venv\\Scripts\\python.exe experiments/phase10_patent2_inspect.py --cmpd 16
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "data" / "processed" / "WO2022251372" / "compounds_combined.csv"
GRADES = ("mdm2", "mdm4", "molm13", "mv4_11", "waga", "mkl1")


def _balanced(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmpd", default=None,
        help="Show only this compound, with its full printed SMILES.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} does not exist.")
        print("Run experiments/phase10_patent2_delivery.py first.")
        return 1

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if args.cmpd:
        for row in rows:
            if row["cmpd_id"] == args.cmpd:
                print(f"compound {row['cmpd_id']}")
                for key in GRADES:
                    print(f"  {key:<8} {row.get(key, '')}")
                text = row.get("smiles") or ""
                print(f"  parses    {row.get('is_valid')}")
                print(f"  balanced  {_balanced(text)}")
                print()
                print("printed SMILES:")
                print(text)
        return 0

    print(f"{CSV_PATH.relative_to(REPO)}")
    print(f"{len(rows)} rows, {len(rows[0])} columns\n")
    header = "cmpd  parse  bal   " + " ".join(g[:5].ljust(5) for g in GRADES)
    print(header)
    print("-" * len(header))
    for row in rows:
        text = row.get("smiles") or ""
        grades = " ".join((row.get(g) or ".").ljust(5) for g in GRADES)
        print(
            f"{row['cmpd_id']:>4}  "
            f"{'yes' if row.get('is_valid') == 'True' else 'no ':>4}  "
            f"{'yes' if _balanced(text) else 'no ':>3}   {grades}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

