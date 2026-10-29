r"""Test whether the last glyph of a printed row is clipped.

Finding that prompted this script
---------------------------------
The recogniser reads the final ``C(=O)`` of row 1 of 1039 as ``C(=0)``.
The disputed glyph is pixel-identical (similarity 1.000) to the same
position in row 1 of 1038, whose reference was verified verbatim by the
researcher and contains ``C(=O)``.  The printed character is therefore
an O, and the recogniser is wrong.

The disputed glyph is also 25 px wide where an unambiguous O in the same
row is 31 px wide.  That points at the crop: the cell crop ends before
the last glyph is complete.

This script tests that explanation by re-cropping the same rows from the
ORIGINAL PAGE IMAGE with a wider right boundary and re-recognising
them.  The crop source, the row, the recogniser and the preprocessing
stay fixed; only the right edge moves.

Read-only.  Does not touch the formal CSV or ``src/cpd``.

Run:
    .venv\Scripts\python.exe -X utf8 phase3_i100298_rightedge.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phase2_ocr_compare import edit_ops  # noqa: E402
from phase3_i100298_sweep import PRINTED_ROWS  # noqa: E402

OUT = REPO / "data" / "audit" / "phase3" / "i100298_rightedge"

GOLD = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
MANIFEST = (REPO / "data" / "audit" / "phase2" / "row_crop_audit"
            / "manifest.csv")

TARGETS = ("1038", "1039", "1040", "1041")
RIGHT_MARGINS = (0, 8, 16, 24, 32, 48)
PAD_TOP = 4
PAD_BOTTOM = 5


def _engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={})


def _recognise(engine, crop: np.ndarray) -> str:
    result = engine(crop, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None)
    if not txts:
        return ""
    return (txts[0] or "").strip()


def _segment_rows(gray: np.ndarray) -> list[tuple[int, int]]:
    """Ink-projection row bands, same method as the recognition route."""
    h, w = gray.shape
    x1 = min(10, max(1, w // 100))
    x2 = max(x1 + 1, w - 10)
    ink = (gray[:, x1:x2] < 160).astype(np.uint8)
    sums = ink.sum(axis=1)
    active = np.flatnonzero(sums > 5)
    if not len(active):
        return []
    groups: list[list[int]] = []
    start = previous = int(active[0])
    for value in active[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append([start, previous])
            start = current
        previous = current
    groups.append([start, previous])
    return [(a, b) for a, b in groups if b - a >= 8]


def _manifest_cells() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with MANIFEST.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["is_header"] == "yes" or row["extraction_status"] != "ok":
                continue
            out[row["cmpd_id"]] = {
                "page": row["page"],
                "x1": int(float(row["bbox_x1"])),
                "y1": int(float(row["bbox_y1"])),
                "x2": int(float(row["bbox_x2"])),
                "y2": int(float(row["bbox_y2"])),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = _manifest_cells()
    engine = _engine()
    rows: list[dict] = []

    for cmpd_id in TARGETS:
        cell = cells[cmpd_id]
        page_path = GOLD / f"PCTCN2025075389-ftappb-{cell['page']}.jpg"
        page = cv2.imread(str(page_path))
        if page is None:
            raise RuntimeError(f"could not read {page_path}")

        # Row bands measured on the original page, in page coordinates.
        cell_region = page[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
        gray = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
        bands = _segment_rows(gray)
        printed = PRINTED_ROWS.get(cmpd_id, [])

        for margin in RIGHT_MARGINS:
            x2 = min(page.shape[1], cell["x2"] + margin)
            parts: list[str] = []
            for index, (top, bottom) in enumerate(bands, start=1):
                y1 = cell["y1"] + max(0, top - PAD_TOP)
                y2 = cell["y1"] + min(cell_region.shape[0],
                                       bottom + PAD_BOTTOM)
                crop = page[y1:y2, cell["x1"]:x2]
                parts.append(_recognise(engine, crop))
            joined = "".join(parts)
            exp = printed[index - 1] if index <= len(printed) else ""
            edit = edit_ops(exp, parts[-1]) if exp else ""
            rows.append({
                "cmpd_id": cmpd_id,
                "right_margin": margin,
                "crop_x2": x2,
                "last_row_text": parts[-1] if parts else "",
                "last_row_edit": edit,
                "joined_length": len(joined),
                "joined_open": joined.count("("),
                "joined_close": joined.count(")"),
                "joined_text": joined,
            })

    fields = list(rows[0].keys())
    with (OUT / "right_edge_test.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=== I100298 right-edge clipping test ===",
        "",
        "Rows are re-cropped from the ORIGINAL PAGE IMAGE with the cell's",
        "right boundary moved outwards.  The row, the recogniser and the",
        "preprocessing are unchanged.",
        "",
        f"{'cmpd':<6} {'margin':>7} {'last_row_edit':>14} {'len':>5} "
        f"{'open':>5} {'close':>6}  last row",
    ]
    for row in rows:
        lines.append(
            f"{row['cmpd_id']:<6} {row['right_margin']:>7} "
            f"{str(row['last_row_edit']):>14} {row['joined_length']:>5} "
            f"{row['joined_open']:>5} {row['joined_close']:>6}  "
            f"{row['last_row_text']}"
        )

    lines.extend([
        "",
        "The last-row edit distance compares the final printed row with",
        "the printed transcription.  1038 has no transcription entry, so",
        "its cell is reported for geometry only.",
        "",
        "No candidate was selected by RDKit parseability, the mass gate,",
        "or DECIMER output.",
    ])
    io.open(OUT / "right_edge_test.txt", "w",
            encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
