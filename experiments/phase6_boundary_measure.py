"""Measure how much of the printed line the column boundary removes.

The accuracy report shows that every remaining character error sits on
the first or last character of a printed line.  Two different causes
are possible and they need different fixes:

* the *column boundary* clipped the glyph, so the recogniser never saw
  the whole character;
* the glyph is complete but the recogniser read it wrong.

This script measures the first one directly.  For every printed line
it reports the distance from the *text* ink to the vertical column
rule that bounds the cell.  The rules are removed first: a rule is a
column that is dark over the whole cell height, and without removing
it every line would appear to touch the edge.

A gap of zero means the text runs into the rule, so the outermost
glyph is clipped by it.

Run:
    .venv\\Scripts\\python.exe experiments/phase6_boundary_measure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import ink_bands  # noqa: E402
from cpd.parsers.page_tables import locate_page_tables  # noqa: E402
from cpd.parsers.profiles import get_profile  # noqa: E402

OUT = REPO / "data" / "audit" / "phase6"

# Compounds with a hand transcription; the page they are printed on.
TARGETS = {
    1: "000116", 2: "000116", 3: "000117", 4: "000117",
    10: "000120", 28: "000126",
}


def main() -> int:
    profile = get_profile("WO2022251372")
    pages = REPO / "data" / "raw" / "WO2022251372" / "golden_tables"
    rows: list[dict] = []

    for compound, stem in sorted(TARGETS.items()):
        path = pages / f"{stem}.tif"
        page = cv2.imread(str(path))
        if page is None:
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

            # The vertical column rules: dark over the whole cell
            # height.  They bound the cell, they are not text.
            dark = gray < 160
            height = gray.shape[0]
            rule = dark.sum(axis=0) / max(height, 1) >= 0.8
            rule_cols = set(np.flatnonzero(rule).tolist())

            for index, (top, bottom) in enumerate(bands):
                row = gray[top:bottom + 1, :]
                text_cols = (row < 160).any(axis=0)
                # Drop the rule columns and their immediate
                # neighbours, which belong to the same rule.
                for column in list(rule_cols):
                    for dx in (-1, 0, 1):
                        if 0 <= column + dx < text_cols.shape[0]:
                            text_cols[column + dx] = False
                xs = np.flatnonzero(text_cols)
                if not xs.size:
                    continue
                # Distance from the text to the nearest rule.
                left_rule = max(
                    (c for c in rule_cols if c < int(xs[0])), default=None
                )
                right_rule = min(
                    (c for c in rule_cols if c > int(xs[-1])), default=None
                )
                left_gap = (
                    int(xs[0]) - left_rule if left_rule is not None
                    else int(xs[0])
                )
                right_gap = (
                    right_rule - int(xs[-1]) if right_rule is not None
                    else cell.shape[1] - 1 - int(xs[-1])
                )
                rows.append({
                    "cmpd_id": compound,
                    "page": stem,
                    "layout": layout_name,
                    "line_index": index,
                    "cell_width": cell.shape[1],
                    "ink_left": int(xs[0]),
                    "ink_right": int(xs[-1]),
                    "left_rule_x": left_rule if left_rule is not None else "",
                    "right_rule_x": (
                        right_rule if right_rule is not None else ""
                    ),
                    "gap_left_px": left_gap,
                    "gap_right_px": right_gap,
                    "clipped_left": left_gap <= 1,
                    "clipped_right": right_gap <= 1,
                })

    out_csv = OUT / "boundary_measure.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    clipped = sum(
        1 for r in rows if r["clipped_left"] or r["clipped_right"]
    )
    both = sum(
        1 for r in rows if r["clipped_left"] and r["clipped_right"]
    )
    print(f"printed lines measured : {n}")
    print(f"lines with ink touching a column edge : {clipped}")
    print(f"lines with ink touching BOTH edges    : {both}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
