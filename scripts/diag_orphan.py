"""Quick diagnostic: which source pages actually wrote each cell PNG, and which layouts were picked."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from cpd.parsers.table_extractor import (
    RapidOcrTableExtractor,
    default_columns,
    detect_layout_from_columns,
)


def main() -> None:
    img_dir = Path("data/raw/WO2025162428/golden_tables")
    ex = RapidOcrTableExtractor()

    # 1. Find pages that contain cmpd_id "1038" anywhere (so we can see if any
    #    activity page later overwrote cells/1038_struct.png).
    print("Pages containing the OCR token '1038' (any column):")
    targets = sorted(img_dir.glob("*.jpg"))
    for p in targets:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Quick full-image OCR
        r = ex.engine(ex._upscale(img[0:int(img.shape[0] * 0.05), :]))
        header_has = False
        for txt in (ex._txts(r) or []):
            if "1038" in txt:
                header_has = True
                break
        if not header_has:
            # fall back: scan cmpd_id column (leftmost ~12%) for 4-digit numbers
            cid_strip = img[:, 0:int(img.shape[1] * 0.13)]
            cid_text = " ".join(ex._txts(ex.engine(ex._upscale(cid_strip))) or [])
            if "1038" in cid_text:
                header_has = True
        if header_has:
            h, w = img.shape[:2]
            n = ex._count_columns(img)
            layout = detect_layout_from_columns(n) or "(fallback)"
            print(f"   {p.name}  shape={w}x{h}  cols={n}  layout={layout}")

    # 2. Inspect I100298 specifically: how many cmpd_ids, what layout, what structures saved
    print("\nDetailed run on I100298:")
    p = next(img_dir.glob("*I100298*.jpg"))
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    n_cols = ex._count_columns(img)
    layout = detect_layout_from_columns(n_cols)
    columns = default_columns(w, layout or "activity")
    print(f"   cols={n_cols} layout={layout}")
    for c in columns:
        print(f"   col={c.name:14s} x=[{c.x1:5d},{c.x2:5d})  is_image={c.is_image}")

    rows = ex._detect_row_boundaries(img)
    print(f"   rows detected: {len(rows)}")
    cid_col = next(c for c in columns if c.name == "cmpd_id")
    for i, (y1, y2) in enumerate(rows):
        cid_cell = img[y1:y2, cid_col.x1:cid_col.x2]
        cid = ex._read_cmpd_id(cid_cell)
        print(f"   row {i}: y=[{y1},{y2}) cmpd_id={cid!r}")


if __name__ == "__main__":
    sys.exit(main())
