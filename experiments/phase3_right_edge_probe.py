"""Where does the SMILES column end, and where does the ink end?

The pipeline's SMILES cell and the phase-2 text crop agree pixel for
pixel over their common columns, but the pipeline crop stops 20 px
earlier on the right.  On I100298 that clips the tail of the longest
printed rows, which is enough to make the recogniser return garbage.

Before changing anything this script measures, on the *page* image:

* where the vertical table rules are,
* where the ink of each printed row actually ends,
* how far the ink reaches past the rule the pipeline snaps to.

The answer decides whether the fix belongs in the column geometry or
in the validated route's own cropping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cpd.ocr.printed_smiles import (  # noqa: E402
    PAD_BOTTOM,
    PAD_TOP,
    ink_bands,
    text_span,
)

PAGE = (
    REPO
    / "data"
    / "raw"
    / "WO2025162428"
    / "golden_tables"
    / "PCTCN2025075389-ftappb-I100298.jpg"
)
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"


def vertical_rules(gray: np.ndarray, frac: float = 0.8) -> list[int]:
    dark = gray < 160
    ratio = dark.sum(axis=0) / max(gray.shape[0], 1)
    return np.flatnonzero(ratio >= frac).tolist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(PAGE))
    if page is None:
        print(f"missing {PAGE}")
        return 1
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    print(f"page {PAGE.name}: {page.shape[1]}x{page.shape[0]}")

    rules = vertical_rules(gray)
    print(f"vertical rules (>=80% dark): {rules}")

    # The phase-2 crop of 1038 is 854 wide; find its offset on the page
    # by template matching a distinctive band.
    cells = REPO / "data" / "audit" / "phase2" / "cells_text"
    crop = cv2.imread(str(cells / "I100298_1038_text.png"))
    if crop is None:
        print("missing phase2 crop")
        return 1

    # Match on the first band of the crop.
    band = crop[0:60, :]
    result = cv2.matchTemplate(
        gray, cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(
        f"phase2 crop first band matches page at x={max_loc[0]} "
        f"y={max_loc[1]} (score {max_val:.4f})"
    )

    x_off = max_loc[0]
    report = {
        "page": PAGE.name,
        "page_size": [page.shape[1], page.shape[0]],
        "rules": rules,
        "crop_offset_x": x_off,
        "crop_offset_y": max_loc[1],
        "rows": [],
    }

    # For every printed row of the crop, measure the ink extent on the
    # page and compare it with the pipeline's column boundary.
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for top, bottom in ink_bands(cg):
        y1 = max(0, top - PAD_TOP)
        y2 = min(crop.shape[0], bottom + PAD_BOTTOM)
        row = cg[y1:y2, :]
        x1, x2 = text_span(row)
        # Page coordinates of the same row.
        py1 = max_loc[1] + y1
        py2 = min(gray.shape[0], max_loc[1] + y2)
        page_row = gray[py1:py2, :]
        prule = vertical_rules(page_row)
        report["rows"].append(
            {
                "band_top": top,
                "band_bottom": bottom,
                "crop_ink_span": [x1, x2],
                "crop_ink_width": x2 - x1 + 1,
                "page_ink_span": [x1 + x_off, x2 + x_off],
                "page_rules_in_row": prule,
            }
        )
        print(
            f"  band {top}-{bottom}: crop ink x{x1}-{x2} "
            f"(page x{x1 + x_off}-{x2 + x_off}); "
            f"rules in row: {prule}"
        )

    (OUT / "right_edge_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'right_edge_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
