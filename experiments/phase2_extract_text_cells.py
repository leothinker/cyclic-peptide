"""Task 2: persist the printed-SMILES text cells (Phase 2, read-only).

Only the *layout* half of ``RapidOcrTableExtractor`` is used, and the
class is instantiated with ``__new__`` so no RapidOCR engine is built and
``main.run_pipeline`` is never touched. Every output goes to
``data/audit/phase2/cells_text/``; ``data/processed/cells`` is not read
for anything except the compound-id ordering already stored in the CSV's
``structure_image`` column, and no official CSV column is rewritten.

Row -> cmpd_id mapping uses the existing ``structure_image`` filenames
instead of re-running OCR on the id column: those crops were written by
the same row walk, so their ascending numeric order inside a page is the
row order. ``header_offset`` records how many detected slices are header
bands so the alignment stays auditable.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

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
CSV_PATH = REPO / "data" / "processed" / "compounds_combined.csv"
OUT = REPO / "data" / "audit" / "phase2" / "cells_text"
META_CSV = OUT / "cells_text_meta.csv"
META_JSONL = OUT / "cells_text_meta.jsonl"

_STRUCT_RE = re.compile(r"(I100\d+)_(\d+)_struct")


def right_table_rule(ex, img) -> int | None:
    """Right-most full-height table rule, page border included.

    ``_SMILES_FRACS`` ends the SMILES column at 0.99 of the page width,
    which on these scans lands ~9 px short of the actual table rule at
    0.994. That truncates the last glyph of every printed line, so the
    audit crops use the real rule instead of the fractional edge.
    """
    body = ex._body_slice(img)
    if body.size == 0:
        return None
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(1, int(body.shape[0] * 0.4)))
    )
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    xs = np.where(np.sum(lines, axis=0) > int(body.shape[0] * 0.4))[0]
    if len(xs) == 0:
        return None
    rules = ex._cluster(xs, gap=ex.COLUMN_GAP_PX)
    return max(rules) if rules else None


def ink_extent(img: np.ndarray) -> tuple[int, int] | None:
    """Left/right column of dark pixels in a single row slice."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = np.where((gray < 128).sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return None
    return int(cols.min()), int(cols.max())


def page_ids_from_csv() -> dict[str, list[int]]:
    """cmpd ids per golden-table page, in ascending row order."""
    import pandas as pd

    per: dict[str, list[int]] = collections.defaultdict(list)
    df = pd.read_csv(CSV_PATH, dtype=str)
    for raw in df["structure_image"].dropna():
        m = _STRUCT_RE.search(raw)
        if m:
            per[m.group(1)].append(int(m.group(2)))
    return {pg: sorted(ids) for pg, ids in per.items()}


def extract(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = RapidOcrTableExtractor.__new__(RapidOcrTableExtractor)
    ids_per_page = page_ids_from_csv()

    meta: list[dict] = []
    for image_path in sorted(GOLDEN.glob("*.jpg")):
        page = image_path.stem.split("-")[-1]
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            continue
        layout = detect_layout_from_columns(ex._count_columns(img))
        rows = ex._detect_row_boundaries(img)

        if layout == "smiles":
            columns = ex._snap_to_borders(
                img, default_columns(img.shape[1], "smiles")
            )
        elif layout in ("activity", "dense") and rows:
            # I100376 opens with one 3-column SMILES row above the dense
            # activity table; that row is the only printed SMILES there.
            first = ex._detect_first_row_columns(img, rows)
            if first is None:
                continue
            # ``_detect_first_row_columns`` returns unsnapped fractional
            # columns, which on I100376 start the crop at x=1128 while the
            # glyphs begin at x=1095 and the first characters are lost.
            columns, rows = ex._snap_to_borders(img, first), rows[:1]
            layout = "smiles_first_row"
        else:
            continue

        smiles_col = next(
            (c for c in columns if c.name == "smiles"), None
        )
        if smiles_col is None or smiles_col.x2 <= smiles_col.x1:
            continue
        rule = right_table_rule(ex, img)
        col_x2 = (
            rule
            if rule is not None and rule > smiles_col.x2
            else smiles_col.x2
        )

        struct_col = next((c for c in columns if c.name == "structure"), None)
        scan_x1 = struct_col.x2 if struct_col is not None else smiles_col.x1

        page_ids = ids_per_page.get(page, [])
        header_offset = max(0, len(rows) - len(page_ids))
        for idx, (y1, y2) in enumerate(rows):
            data_i = idx - header_offset
            status = "ok"
            cmpd_id = ""
            if data_i < 0:
                status = "header_band"
            elif data_i >= len(page_ids):
                status = "row_without_id"
            else:
                cmpd_id = str(page_ids[data_i])

            # Widen the left edge onto the first inked column inside the
            # cell. The fractional/snap boundary can land to the right of
            # the first glyph (I100376's leading row starts its text at
            # x=1095 but the column edge sits at 1128), which silently
            # drops characters from the printed string.
            band = img[y1:y2, scan_x1:col_x2]
            band_ink = ink_extent(band)
            x1 = smiles_col.x1
            x2 = col_x2
            if band_ink is not None:
                x1 = max(
                    scan_x1,
                    min(col_x2 - 1, scan_x1 + max(0, band_ink[0] - 2)),
                )
                x2 = min(col_x2, scan_x1 + band_ink[1] + 3)
            # The full-height rule detector misses the closing rule on a
            # short leading row (I100376's rule sits at x=1931 but the
            # probe stops at 1925). Recover it from the page's own right
            # margin so trailing glyphs are not clipped.
            if col_x2 >= img.shape[1] - 60:
                x2 = img.shape[1]
            cell = img[y1:y2, x1:x2]
            if cell.size == 0:
                if status == "ok":
                    status = "empty_cell"
                crop_path = ""
            else:
                name = f"{page}_{cmpd_id or 'row%d' % idx}_text.png"
                cv2.imwrite(str(out_dir / name), cell)
                crop_path = f"cells_text/{name}"
                if status == "ok" and cell.max() - cell.min() < 12:
                    status = "blank_cell"

            meta.append(
                {
                    "cmpd_id": cmpd_id,
                    "source_image": image_path.name,
                    "page": page,
                    "layout": layout,
                    "row_index": idx,
                    "header_offset": header_offset,
                    "bbox_x1": int(x1),
                    "bbox_y1": int(y1),
                    "bbox_x2": int(x2),
                    "bbox_y2": int(y2),
                    "crop_path": crop_path,
                    "crop_px": f"{cell.shape[1]}x{cell.shape[0]}"
                    if cell.size
                    else "0x0",
                    "extraction_status": status,
                }
            )

    fields = list(meta[0]) if meta else []
    with META_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(meta)
    with META_JSONL.open("w", encoding="utf-8") as fh:
        for row in meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["extraction_status"] for r in meta)
    print(f"rows written: {len(meta)}")
    print(f"pages: {len({r['page'] for r in meta})}")
    print("status:", dict(counts))
    print(f"unique cmpd_id with a crop: "
          f"{len({r['cmpd_id'] for r in meta if r['cmpd_id']})}")
    print(f"crops on disk: {len(list(out_dir.glob('*_text.png')))}")
    print(f"metadata: {META_CSV.relative_to(REPO)}  {META_JSONL.relative_to(REPO)}")
    return len(meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="output directory for crops + metadata",
    )
    args = ap.parse_args()
    extract(args.out if args.out.is_absolute() else REPO / args.out)


if __name__ == "__main__":
    main()
