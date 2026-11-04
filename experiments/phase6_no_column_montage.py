"""Render the ``No.`` column of every WO2022251372 table band.

The number-column OCR says some data bands carry no compound number.
That would mean those bands continue the previous compound's row
across a page break.  OCR alone is not enough evidence for a rule
that decides how rows are joined, so this script renders the column
strips so the claim can be checked by eye.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_no_column_montage.py
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
OUT = REPO / "data" / "audit" / "phase6" / "no_column_montage.png"


def main() -> int:
    strips: list[np.ndarray] = []
    for path in sorted(PAGES.glob("*.tif")):
        page = cv2.imread(str(path))
        rules = horizontal_rules(page)
        for table in locate_page_tables(page):
            inner = [y for y in rules if table.y1 <= y <= table.y2]
            x1 = table.x1 + table.column_boundaries[0]
            x2 = table.x1 + table.column_boundaries[1]
            for y1, y2 in zip(inner, inner[1:], strict=False):
                strip = page[y1:y2, x1:x2]
                # Scale to a fixed strip width and label it.
                strip = cv2.copyMakeBorder(
                    strip, 6, 6, 4, 4, cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
                strip = cv2.resize(
                    strip, (120, max(40, strip.shape[0])),
                    interpolation=cv2.INTER_AREA,
                )
                label = np.full((strip.shape[0], 460, 3), 255, np.uint8)
                cv2.putText(
                    label,
                    f"{path.stem} c{table.n_columns} y{y1}",
                    (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )
                label[:strip.shape[0], 124:244] = strip
                strips.append(label)

    height = sum(s.shape[0] for s in strips)
    canvas = np.full((height, 460, 3), 255, np.uint8)
    top = 0
    for strip in strips:
        canvas[top:top + strip.shape[0]] = strip
        top += strip.shape[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), canvas)
    print(f"wrote {OUT} ({len(strips)} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
