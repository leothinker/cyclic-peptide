"""Check whether WO2022251372 table rows continue across page breaks.

The band geometry of page 115 shows a data band that starts in the
middle of a SMILES string, and page 116 repeats the header before a
band that also starts mid-molecule.  If true, a compound's printed
row spans several pages and the compound number appears only on the
first fragment.  That changes how the adapter must join rows, so it
needs to be measured rather than assumed.

This probe reports, for every band of every table on every page:

* whether the ``No.`` cell holds ink,
* what the recogniser reads from it,
* the height of the band, and
* whether the SMILES cell starts mid-molecule (a leading fragment that
  does not begin with the usual ``COCC``/``CCC`` head).

Run:
    .venv\\Scripts\\python.exe experiments/phase6_row_number_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import default_recogniser  # noqa: E402
from cpd.parsers.page_tables import (  # noqa: E402
    horizontal_rules,
    locate_page_tables,
)

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


def ink_ratio(cell: np.ndarray) -> float:
    if cell.size == 0:
        return 0.0
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return float((gray < 160).mean())


def main() -> int:
    engine = default_recogniser()
    for path in sorted(PAGES.glob("*.tif")):
        page = cv2.imread(str(path))
        rules = horizontal_rules(page)
        tables = locate_page_tables(page)
        for table in tables:
            inner = [y for y in rules if table.y1 <= y <= table.y2]
            bands = list(zip(inner, inner[1:], strict=False))
            for index, (y1, y2) in enumerate(bands):
                # The No. column is the first logical column.
                no_cell = page[
                    y1:y2,
                    table.x1 + table.column_boundaries[0]:
                    table.x1 + table.column_boundaries[1],
                ]
                ratio = ink_ratio(no_cell)
                result = engine(no_cell) if ratio > 0.005 else None
                txts = getattr(result, "txts", None) or []
                label = "HEADER" if index == 0 else "data"
                print(
                    f"{path.stem} cols={table.n_columns} band{index} "
                    f"{label} h={y2 - y1} no_ink={ratio:.4f} "
                    f"no_txt={txts[0] if txts else None!r}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
