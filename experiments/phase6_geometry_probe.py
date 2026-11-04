"""Report the ruled-table geometry of every WO2022251372 page.

The row-splitting fix for full-page inputs depends on one fact: every
ruled table in these pages is a stack of horizontal bands, and the
*first* band of each stack is the header.  This script prints the
measured evidence for that claim so the fix can be based on the
measurement rather than on an assumption.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_geometry_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.parsers.page_tables import (  # noqa: E402
    horizontal_rules,
    locate_page_tables,
    vertical_boundaries,
)

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"


def main() -> int:
    for path in sorted(PAGES.glob("*.tif")):
        image = cv2.imread(str(path))
        rules = horizontal_rules(image)
        tables = locate_page_tables(image)
        print(f"=== {path.name}  size={image.shape[1]}x{image.shape[0]}")
        print(f"    page rules: {rules}")
        for table in tables:
            print(
                f"    table cols={table.n_columns} "
                f"y=[{table.y1},{table.y2}] x=[{table.x1},{table.x2}] "
                f"bounds={table.column_boundaries}"
            )
            # Re-derive the bands inside this table so the header/data
            # split can be checked band by band.
            inner = [y for y in rules if table.y1 <= y <= table.y2]
            for index, (y1, y2) in enumerate(
                zip(inner, inner[1:], strict=False)
            ):
                bounds = vertical_boundaries(image[y1:y2, :])
                print(
                    f"      band {index}: y=[{y1},{y2}] "
                    f"h={y2 - y1} cols={len(bounds) - 1}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
