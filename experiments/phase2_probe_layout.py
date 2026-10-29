"""Read-only layout probe for the patent golden tables (Phase 2).

Instantiates ``RapidOcrTableExtractor`` without running ``__init__`` so no
RapidOCR model is loaded: every method used here is either a class
constant read or a pure-OpenCV helper. Prints the detected column layout
and row count for each page so the cell-extraction script can target the
pages that actually carry printed SMILES text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpd.parsers.table_extractor import (  # noqa: E402
    RapidOcrTableExtractor,
    default_columns,
    detect_layout_from_columns,
)

GOLDEN = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"


def probe() -> None:
    ex = RapidOcrTableExtractor.__new__(RapidOcrTableExtractor)
    for path in sorted(GOLDEN.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            print(f"{path.stem}\tUNREADABLE")
            continue
        n_cols = ex._count_columns(img)
        layout = detect_layout_from_columns(n_cols)
        rows = ex._detect_row_boundaries(img)
        first_row_layout = None
        if rows and layout in ("dense", "activity"):
            fr = ex._detect_first_row_columns(img, rows)
            if fr is not None:
                first_row_layout = [c.name for c in fr]
        print(
            f"{path.stem}\tshape={img.shape[:2]}\tncols={n_cols}"
            f"\tlayout={layout}\trows={len(rows)}"
            f"\tfirst_row={first_row_layout}"
        )


if __name__ == "__main__":
    probe()
