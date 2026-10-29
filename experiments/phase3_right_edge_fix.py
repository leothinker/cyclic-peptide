"""Can the SMILES column crop be widened so that no ink is clipped?

Measured on patent page I100298 (1945 x 2525):

* vertical table rules at page x = 11/12, 191/192, 1091/1092, 1931/1932;
* the printed SMILES ink of the longest row reaches x = 1930, i.e. it
  runs right up against the last rule;
* the pipeline's SMILES column stops at x = 1925, so the last few
  pixels of the row are cropped away.

That single missing sliver is enough to flip the recogniser from the
correct string to garbage: the same ink, trimmed by six pixels,
decodes to ``CO()CC((.(C)`` instead of ``CCO[C@H]1CN2[C@@H](C1)...

Two candidate fixes are compared here:

``snap_rule``   the SMILES column is snapped to the last vertical rule
                (x = 1931) instead of stopping at the fractional
                boundary, so the ink is kept;
``pad_right``   the column keeps its current boundary but the cell is
                extended with a white margin.

Only the *crop* changes; the recogniser, the preprocessing and the
route are identical.  The comparison is character level against the
phase-2 crop, which is the crop the validated route was tuned on.
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

COMPOUNDS = {
    "1038": "PCTCN2025075389-ftappb-I100298.jpg",
    "1039": "PCTCN2025075389-ftappb-I100298.jpg",
    "1040": "PCTCN2025075389-ftappb-I100298.jpg",
    "1041": "PCTCN2025075389-ftappb-I100298.jpg",
}

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


def read_cell(engine, cell: np.ndarray) -> str:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    parts = []
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
        if text:
            parts.append(text)
    return "".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()

    page = cv2.imread(str(PAGE_DIR / COMPOUNDS["1038"]))
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # Locate the phase-2 crop inside the page.
    crop = cv2.imread(str(CELLS / "I100298_1038_text.png"))
    band = crop[0:60, :]
    result = cv2.matchTemplate(
        gray, cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x_off, y_off = max_loc
    print(
        f"phase2 crop anchored at page x={x_off} y={y_off} "
        f"(score {max_val:.4f}); page {page.shape[1]}x{page.shape[0]}"
    )

    rows = []
    for cmpd in COMPOUNDS:
        reference = REFERENCES[cmpd]
        for arm in ("phase2_crop", "pipeline_crop", "snap_rule", "pad_right"):
            if arm == "phase2_crop":
                cell = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
            elif arm == "pipeline_crop":
                idx = int(cmpd) - 1037
                cell = cv2.imread(
                    str(
                        OUT.parent
                        / "i100298_validation"
                        / "cellcrops"
                        / f"pipeline_smiles_cell_{idx:02d}.png"
                    )
                )
            else:
                # Rebuild the cell from the page with a different right
                # boundary.  The row slice is the same for every arm.
                src = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
                height = src.shape[0]
                if arm == "snap_rule":
                    # extend to the last vertical rule on the page
                    right = 1931 - x_off
                    cell = page[y_off : y_off + height, x_off : x_off + right]
                else:
                    # keep the pipeline boundary, add a white margin
                    cell = cv2.copyMakeBorder(
                        page[y_off : y_off + height, x_off : x_off + 834],
                        0,
                        0,
                        0,
                        40,
                        cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )

            if cell is None:
                print(f"  {cmpd} {arm}: no crop")
                continue
            text = read_cell(engine, cell)
            edits = edit_distance(text, reference)
            rows.append(
                {
                    "cmpd_id": cmpd,
                    "arm": arm,
                    "crop_width": int(cell.shape[1]),
                    "ref_len": len(reference),
                    "ocr_len": len(text),
                    "edit": edits,
                    "cer": round(edits / len(reference), 6),
                    "text": text,
                }
            )
            print(
                f"  {cmpd} {arm:14s} w={cell.shape[1]:4d} "
                f"edit={edits:3d} cer={edits / len(reference):.5f}"
            )

    with (OUT / "right_edge_fix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "right_edge_fix.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
