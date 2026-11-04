"""Save the individual printed-line crops of two compounds.

The accuracy report shows that every remaining character error sits
at the first or last character of a printed line.  This script writes
those lines out as images so the cause can be seen rather than
inferred: a line whose first glyph is cut by the crop border looks
different from a line whose first glyph is simply misread.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_line_boundary_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    ink_bands,
    text_span,
)
from cpd.parsers.page_tables import locate_page_tables  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402

OUT = REPO / "data" / "audit" / "phase6" / "line_boundary"

# (compound, page stem, line index, note)
TARGETS = [
    (10, "000120", 7, "leading @ dropped by the recogniser"),
    (28, "000126", 0, "leading [ read as ]"),
    (2, "000116", 9, "trailing 1 dropped"),
]


def main() -> int:
    profile = get_profile("WO2022251372")
    OUT.mkdir(parents=True, exist_ok=True)
    pages = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"

    for compound, stem, line_index, note in TARGETS:
        path = pages / f"{stem}.tif"
        page = cv2.imread(str(path))
        if page is None:
            print(f"!! cannot read {path}")
            continue
        for table in locate_page_tables(page):
            layout_name = profile.layout_for_column_count(table.n_columns)
            if layout_name is None:
                continue
            crop = page[table.y1:table.y2, table.x1:table.x2]
            bounds = table.column_boundaries
            cell = crop[:, bounds[1]:bounds[2]]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            bands = ink_bands(gray)
            if not bands:
                continue
            for index, (top, bottom) in enumerate(bands):
                y1 = max(0, top - PAD_TOP)
                y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
                row = cell[y1:y2, :]
                row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
                x1, x2 = text_span(row_gray)
                variant = row[:, x1:x2 + 1]
                marked = variant.copy()
                # Mark the left and right border of the ink span so a
                # glyph that touches the border is easy to see.
                marked[:, 0:2] = (0, 0, 255)
                marked[:, -2:] = (0, 0, 255)
                name = f"cmpd{compound}_{stem}_{layout_name}_line{index}.png"
                cv2.imwrite(str(OUT / name), marked)
                print(
                    f"{name}: band {index} "
                    f"(ink rows {top}-{bottom}), "
                    f"span x {x1}-{x2}, width {variant.shape[1]}, "
                    f"note: {note if index == line_index else ''}"
                )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
