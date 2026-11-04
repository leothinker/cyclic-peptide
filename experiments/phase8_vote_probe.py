"""Why does the vote not change the result of the two hard compounds?

The ``padded`` route reads compounds 1-4 of WO2022251372 exactly, and
makes one and three character errors on compounds 10 and 28.  The
``padded_vote`` route, which reads every line with a second
recogniser and lets the two readings settle, returns the same result.

This script shows why, band by band: how many bands the two
recognisers disagree on, what each of them reads, and whether the
vote can find a combination that RDKit accepts.

Usage::

    .venv\\Scripts\\python.exe experiments/phase8_vote_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO / "experiments"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PrintedSmilesOCR,
    ROUTE_PADDED,
)
from cpd.ocr.vote import vote_lines  # noqa: E402
from wo2022251372_references import REFERENCE_LINES  # noqa: E402

CELLS = (
    REPO / "data" / "audit" / "phase4" / "wo2022251372" / "table1a_cells"
)
CELL_FILES = {10: "000120_cmpd10.png", 28: "000126_cmpd28.png"}


def main() -> int:
    reader = PrintedSmilesOCR(route=ROUTE_PADDED)

    for compound_id, cell_name in CELL_FILES.items():
        path = CELLS / cell_name
        cell = cv2.imread(str(path))
        if cell is None:
            print(f"cannot read {path}")
            continue
        reference = "".join(REFERENCE_LINES[compound_id])

        primary = reader._read_cell_ink_bands(
            cell, margin=6, scale=1, engine=reader.engine, as_lines=True,
        )
        secondary = reader._read_cell_ink_bands(
            cell, margin=6, scale=1, engine=reader.second_engine,
            as_lines=True,
        )
        print(f"\n================ compound {compound_id} ================")
        print(f"reference ({len(reference)} chars): {reference!r}")
        differing = 0
        for index in range(max(len(primary), len(secondary))):
            first = primary[index] if index < len(primary) else ""
            second = secondary[index] if index < len(secondary) else ""
            mark = " " if first == second else "*"
            if first != second:
                differing += 1
            print(f"  {mark} band {index}: en_v4={first!r}")
            if first != second:
                print(f"      default_v6={second!r}")

        for limit in (1, 2, 3, 5, 10):
            result = vote_lines(primary, secondary, max_changes=limit)
            print(
                f"  vote(max_changes={limit:<2}) -> changed="
                f"{result.changed_bands} parses={result.parses} "
                f"text={result.text[:60]!r}"
            )
        print(f"  bands that differ: {differing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
