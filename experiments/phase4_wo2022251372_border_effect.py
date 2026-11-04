"""Minimal reproduction of the line-end border effect.

``text_span`` trims the band to the first and last column that contain
any ink, so the final glyph of a printed line touches the crop border.
This script recognises one printed line with and without a white margin
and prints both strings next to the hand transcription, which is the
single observation that pinned the root cause.

Run:
    .venv\\Scripts\\python.exe experiments/phase4_wo2022251372_border_effect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    ModelType,
    OCRVersion,
    TaskType,
)

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.printed_smiles import ink_bands, text_span  # noqa: E402

CELL = (
    REPO
    / "data"
    / "audit"
    / "phase4"
    / "wo2022251372"
    / "table1a_cells"
    / "000116_cmpd1.png"
)

# Printed line 4 of compound 1, as transcribed by hand.
REFERENCE = "H](Cc2ccc(Cl)cc2)NC(=O)["

MARGINS = (0, 2, 4, 6, 8, 12, 16, 24)


def main() -> int:
    engine = RapidOCR(
        params={
            "Rec.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.task_type": TaskType.REC,
            "Global.text_score": 0.0,
            "Global.use_cls": False,
        }
    )

    cell = cv2.imread(str(CELL))
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bands = ink_bands(gray)
    top, bottom = bands[3]
    row = cell[max(0, top - 14) : bottom + 15, :]
    row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    x1, x2 = text_span(row_gray)
    band = row[:, x1:x2 + 1]

    print(f"reference : {REFERENCE!r}")
    for margin in MARGINS:
        padded = band
        if margin:
            padded = cv2.copyMakeBorder(
                band,
                0,
                0,
                margin,
                margin,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        result = engine(padded, use_det=False, use_rec=True)
        txts = getattr(result, "txts", None) or []
        text = (txts[0] or "").strip() if txts else ""
        flag = "  <-- exact" if text == REFERENCE else ""
        print(f"margin {margin:>2}px -> {text!r}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
