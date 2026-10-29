"""Does extending the SMILES crop to the table's right rule fix it?

Measured on I100298:

* the table's right-hand vertical rule sits at page x = 1931/1932;
* the printed ink of the longest row reaches x = 1930, i.e. it stops
  one pixel before the rule;
* the pipeline's SMILES column ends at x = 1925, so the last glyph of
  that row is sliced through the middle;
* a half-cut glyph makes the CTC recogniser return garbage for the
  *whole* row, while the same row with white padding instead of the
  missing pixels decodes correctly.

The fix under test is therefore geometric: let the SMILES cell run up
to the next vertical rule instead of stopping at a fractional column
boundary.  ``text_span`` already removes rule columns before
recognition, so including the rule is harmless.

Arms:

``frac``     the current behaviour (column boundary at 0.99 width)
``rule``     the cell is extended to the next vertical rule
``rule2``    the cell is extended two pixels past the rule
``phase2``   the phase-2 crop, i.e. the whole cell up to the page edge
"""

from __future__ import annotations

import csv
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
    default_recogniser,
    ink_bands,
    text_span,
    upscale,
)

PAGE_DIR = REPO / "data" / "raw" / "WO2025162428" / "golden_tables"
CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"

COMPOUNDS = ("1038", "1039", "1040", "1041")

REFERENCES: dict[str, str] = {
    "1038": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N(C)C"
    ),
    "1039": (
        "CC[C@H](C)[C@H]1C(=O)N(C)[C@@H](C)C(=O)N2[C@@H](CC2)C(=O)N(CCC(F)(F)F)"
        "[C@H]3C(=O)N(C)CC(=O)N[C@@H](CCC4=CC(F)=C(C(F)=C4)C(F)(F)F)C(=O)N5[C@@H]"
        "(CCC5)C(=O)NC6(CCCC6)C(=O)N(C)[C@@H](C7CCCC7)C(=O)N(C)[C@@H](CC(=O)N(C)"
        "[C@H](CCNC(=S)C8=C(F)C(F)=C(SC3)C(F)=C8F)C(=O)N1)C(=O)N9CCCC9"
    ),
    "1040": (
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C"
    ),
    "1041": (
        "CCO[C@H]1CN2[C@@H](C1)C(=O)NC3(CCCC3)C(=O)N(C)[C@@H](C4CCCC4)C(=O)N(C)"
        "[C@@H](CC(=O)N(C)[C@@H]5C(=O)N[C@@H]([C@@H](C)CC)C(=O)N(C)[C@@H](C)"
        "C(=O)N6[C@@H](CC6)C(=O)N(C)[C@H](CSC7=C(F)C(F)=C(C(=S)NCC5)C(F)=C7F)"
        "C(=O)N(C)CC(=O)N[C@@H](CCC8=CC(F)=C(C(F)=C8)C(F)(F)F)C2=O)C(=O)N(C)C"
    ),
}


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def recognise(engine, image: np.ndarray) -> str:
    result = engine(image, use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def read_cell(engine, cell: np.ndarray) -> tuple[str, list[dict]]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts = []
    detail = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        strip = row[:, x1 : x2 + 1]
        if strip.size == 0:
            continue
        text = recognise(engine, upscale(strip))
        parts.append(text)
        detail.append({"span": [x1, x2], "width": int(strip.shape[1])})
    return "".join(parts), detail


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()
    page = cv2.imread(str(PAGE_DIR / "PCTCN2025075389-ftappb-I100298.jpg"))
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    rows = []
    for cmpd in COMPOUNDS:
        reference = REFERENCES[cmpd]
        crop = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
        # Each compound sits at its own y offset on the page, so the
        # crop has to be located individually.
        result = cv2.matchTemplate(
            gray,
            cv2.cvtColor(crop[0:60, :], cv2.COLOR_BGR2GRAY),
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        x_off, y_off = max_loc
        height = crop.shape[0]
        print(
            f"{cmpd}: phase2 crop anchored at x={x_off} y={y_off} "
            f"(score {max_val:.4f})"
        )
        for arm, right in (
            ("frac", 1925 - x_off),
            ("rule", 1931 - x_off),
            ("rule2", 1933 - x_off),
            ("phase2", 854),
        ):
            cell = page[y_off : y_off + height, x_off : x_off + right]
            text, detail = read_cell(engine, cell)
            edits = edit_distance(text, reference)
            rows.append(
                {
                    "cmpd_id": cmpd,
                    "arm": arm,
                    "right_page_x": x_off + right,
                    "crop_width": int(cell.shape[1]),
                    "edit": edits,
                    "cer": round(edits / len(reference), 6),
                    "band_widths": ";".join(
                        str(d["width"]) for d in detail
                    ),
                    "text": text,
                }
            )
            print(
                f"  {cmpd} {arm:7s} right_page_x={x_off + right} "
                f"w={cell.shape[1]:4d} edit={edits:3d} "
                f"cer={edits / len(reference):.5f}"
            )

    with (OUT / "rule_extension.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "rule_extension.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
