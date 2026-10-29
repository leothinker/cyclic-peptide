"""Is the printed-SMILES recogniser stable with respect to crop width?

The width sweep on I100298 showed a *non-monotonic* failure: trimming
the strip to 832 px produced garbage while 831 px and 838 px were
correct.  Two explanations are possible:

*content* -- the 6 trimmed pixels carry a glyph, and losing it
              changes the answer.  This would mean the crop is
              genuinely losing information.
*geometry* -- the recogniser's input normalisation is unstable at
              particular widths, so the same pixels decode differently
              depending on how wide the strip happens to be.

To separate the two, this script pads the *same* strip with white
paper on the right to a range of widths.  The ink is identical in
every case, so any change in the output is a geometry effect.
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

CELLS = REPO / "data" / "audit" / "phase2" / "cells_text"
OUT = REPO / "data" / "audit" / "phase3" / "width_sensitivity"

COMPOUNDS = ("1038", "1039", "1040", "1041")


def strips_of(cell: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    out = []
    for top, bottom in ink_bands(gray):
        y1 = max(0, top - PAD_TOP)
        y2 = min(cell.shape[0], bottom + PAD_BOTTOM)
        row = cell[y1:y2, :]
        row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
        x1, x2 = text_span(row_gray)
        out.append(row[:, x1 : x2 + 1])
    return out


def recognise(engine, strip: np.ndarray) -> str:
    result = engine(upscale(strip), use_det=False, use_rec=True)
    txts = getattr(result, "txts", None) or []
    return (txts[0] or "").strip() if txts else ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = default_recogniser()

    widths = list(range(820, 861))

    rows = []
    for cmpd in COMPOUNDS:
        cell = cv2.imread(str(CELLS / f"I100298_{cmpd}_text.png"))
        strips = strips_of(cell)
        for band, strip in enumerate(strips, start=1):
            base = strip
            for width in widths:
                # Cropping would remove ink, so only widths that keep
                # every pixel are used: the ink stays constant across
                # all rows of the sweep.
                if width < base.shape[1]:
                    continue
                padded = cv2.copyMakeBorder(
                    base,
                    0,
                    0,
                    0,
                    width - base.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
                text = recognise(engine, padded)
                rows.append(
                    {
                        "cmpd_id": cmpd,
                        "band": band,
                        "ink_width": int(base.shape[1]),
                        "padded_width": width,
                        "height": int(base.shape[0]),
                        "text": text,
                    }
                )

    with (OUT / "pad_width_sweep.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for cmpd in COMPOUNDS:
        for band in range(1, 8):
            sel = [
                r
                for r in rows
                if r["cmpd_id"] == cmpd and r["band"] == band
            ]
            if not sel:
                continue
            counts: dict[str, int] = {}
            for r in sel:
                counts[r["text"]] = counts.get(r["text"], 0) + 1
            modal = max(counts, key=lambda k: counts[k])
            outliers = [
                {"padded_width": r["padded_width"], "text": r["text"]}
                for r in sel
                if r["text"] != modal
            ]
            summary.append(
                {
                    "cmpd_id": cmpd,
                    "band": band,
                    "n_widths": len(sel),
                    "modal_text": modal,
                    "modal_count": counts[modal],
                    "distinct_outputs": len(counts),
                    "outlier_widths": [o["padded_width"] for o in outliers],
                    "outliers": outliers,
                }
            )

    (OUT / "pad_width_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for entry in summary:
        print(
            f"{entry['cmpd_id']} band{entry['band']}: "
            f"{entry['modal_count']}/{entry['n_widths']} widths agree, "
            f"{entry['distinct_outputs']} distinct outputs, "
            f"outlier widths {entry['outlier_widths']}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
