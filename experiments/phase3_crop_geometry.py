"""Compare the phase-2 text crop with the pipeline's own SMILES crop.

The validated OCR route is perfect on the phase-2 crops
(edit distance 0 for 1038-1041) but mis-reads 1040/1041 when the
same route is fed the crop that ``table_extractor`` produces.

This script measures how the two crops relate: is one a shifted or
truncated copy of the other, which columns carry ink, and where the
recogniser's input differs.
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
    default_recogniser,
    ink_bands,
    text_span,
    upscale,
)

PHASE2 = REPO / "data" / "audit" / "phase2" / "cells_text"
PIPELINE = (
    REPO / "data" / "audit" / "phase3" / "i100298_validation" / "cellcrops"
)
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"


def rows_of(cell: np.ndarray):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append((top, bottom, x1, x2, row[:, x1 : x2 + 1]))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()
    report = []

    for idx, cmpd in enumerate(("1038", "1039", "1040", "1041"), start=1):
        a = cv2.imread(str(PHASE2 / f"I100298_{cmpd}_text.png"))
        b = cv2.imread(str(PIPELINE / f"pipeline_smiles_cell_{idx:02d}.png"))
        print(f"\n=== {cmpd} ===")
        print(f"  phase2   {a.shape[1]}x{a.shape[0]}")
        print(f"  pipeline {b.shape[1]}x{b.shape[0]}")

        # Is the pipeline crop a left-aligned truncation of phase2?
        common_h = min(a.shape[0], b.shape[0])
        common_w = min(a.shape[1], b.shape[1])
        sub = a[:common_h, :common_w]
        diff = cv2.absdiff(sub, b[:common_h, :common_w])
        print(
            f"  left-aligned pixel diff: max={diff.max()} "
            f"mean={diff.mean():.3f} "
            f"frac>30={float((diff.max(axis=2) > 30).mean()):.4f}"
        )

        rows_a = rows_of(a)
        rows_b = rows_of(b)
        entry = {
            "cmpd_id": cmpd,
            "phase2_shape": [a.shape[1], a.shape[0]],
            "pipeline_shape": [b.shape[1], b.shape[0]],
            "bands": [],
        }
        for i, (ra, rb) in enumerate(zip(rows_a, rows_b), start=1):
            ta, ba, xa1, xa2, sa = ra
            tb, bb, xb1, xb2, sb = rb
            ta_text = engine(upscale(sa), use_det=False, use_rec=True)
            tb_text = engine(upscale(sb), use_det=False, use_rec=True)
            ta_txt = (getattr(ta_text, "txts", [""]) or [""])[0]
            tb_txt = (getattr(tb_text, "txts", [""]) or [""])[0]
            same = sa.shape == sb.shape and bool(np.array_equal(sa, sb))
            print(
                f"  band{i}: phase2 x{xa1}-{xa2} w{sa.shape[1]} "
                f"| pipe x{xb1}-{xb2} w{sb.shape[1]} "
                f"| identical={same}"
            )
            print(f"     phase2   : {ta_txt}")
            print(f"     pipeline : {tb_txt}")
            entry["bands"].append(
                {
                    "band": i,
                    "phase2_span": [xa1, xa2],
                    "phase2_width": int(sa.shape[1]),
                    "pipeline_span": [xb1, xb2],
                    "pipeline_width": int(sb.shape[1]),
                    "identical": same,
                    "phase2_text": ta_txt,
                    "pipeline_text": tb_txt,
                }
            )
        report.append(entry)

    (OUT / "crop_geometry.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT / 'crop_geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
