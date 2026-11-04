"""Render a readable grid of the compound-number cells.

Two questions need an eye, not an OCR verdict:

1. Does a data band whose number cell OCRs as empty really contain no
   number (i.e. the row continues from the previous page)?
2. Does every band that *does* carry a number carry a readable one?

The number is printed at the top of its cell, so only the top of the
``No.`` column is cropped and it is scaled up.  Bands are laid out in
a grid, one cell per band, labelled with page and band index.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_no_number_grid.py
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

from cpd.parsers.page_tables import (  # noqa: E402
    horizontal_rules,
    locate_page_tables,
)

PAGES = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"
OUT = REPO / "data" / "audit" / "phase6" / "no_number_grid.png"

TOP_PX = 110
CELL_W = 150
CELL_H = 150


def main() -> int:
    cells: list[tuple[str, np.ndarray]] = []
    for path in sorted(PAGES.glob("*.tif")):
        page = cv2.imread(str(path))
        rules = horizontal_rules(page)
        for table in locate_page_tables(page):
            inner = [y for y in rules if table.y1 <= y <= table.y2]
            x1 = table.x1 + table.column_boundaries[0]
            x2 = table.x1 + table.column_boundaries[1]
            for index, (y1, y2) in enumerate(
                zip(inner, inner[1:], strict=False)
            ):
                strip = page[y1:min(y2, y1 + TOP_PX), x1:x2]
                strip = cv2.copyMakeBorder(
                    strip, 8, 8, 8, 8, cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
                strip = cv2.resize(
                    strip, (CELL_W - 20, CELL_H - 46),
                    interpolation=cv2.INTER_CUBIC,
                )
                cells.append(
                    (f"{path.stem} b{index} c{table.n_columns}", strip)
                )

    cols = 10
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * CELL_H, cols * CELL_W, 3), 255, np.uint8)
    for position, (label, strip) in enumerate(cells):
        row, col = divmod(position, cols)
        top, left = row * CELL_H, col * CELL_W
        canvas[top:top + strip.shape[0], left:left + strip.shape[1]] = strip
        cv2.putText(
            canvas, label, (left + 4, top + CELL_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160), 1, cv2.LINE_AA,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), canvas)
    print(f"wrote {OUT} ({len(cells)} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
